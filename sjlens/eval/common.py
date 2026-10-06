"""Shared helpers for the eval scripts: rank correlation, context loading, model loading."""
import torch


def spearman(a, b):
    """Spearman rank correlation of two 1-d tensors (ties broken by position, as in dev_checks.py)."""
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def wikitext_contexts(tok, n, T=128, seed=0, min_chars=600):
    """n wikitext-103 records of exactly T tokens (the lens loader's filter), as [T] long tensors on CPU."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    out = []
    for r in ds:
        if len(r["text"]) < min_chars:
            continue
        ids = tok(r["text"], return_tensors="pt").input_ids[0, :T]
        if ids.numel() == T:
            out.append(ids)
        if len(out) == n:
            break
    return out


def load_hf(model_name, dtype=torch.float32, device="cuda"):
    """Frozen HF model with eager attention (functorch-safe) plus its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="eager").to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m


DTYPES = {"fp32": (torch.float32, None), "bf16": (torch.bfloat16, None), "bf16-mixed": (torch.bfloat16, torch.float32)}


def load_q(model_name, dtype="fp32", device="cuda"):
    """(tok, hf model, Qwen3Min) for a --dtype string: fp32 | bf16 (pure) | bf16-mixed (bf16 weights, fp32
    activations via act_dtype; G2 measured mixed = fp32 exactly for Qwen3 checkpoints, at half the weight memory)."""
    from ..model.qwen3_min import Qwen3Min
    wd, ad = DTYPES[dtype]
    tok, m = load_hf(model_name, wd, device)
    return tok, m, Qwen3Min(m, act_dtype=ad)


def background_activations(q, contexts, layer, mask):
    """[N, d] float32 residuals at `layer` over the masked positions of every context."""
    with torch.no_grad():
        return torch.cat([q.resid(c.view(1, -1), layer)[0, mask] for c in contexts]).float()


def sample_activations(H, n, seed=0):
    """n activations drawn uniformly over all (context, position) pairs of H (background_activations concatenates
    context by context, so H[:n] would be n adjacent positions of one record)."""
    g = torch.Generator().manual_seed(seed)
    return H[torch.randperm(H.shape[0], generator=g)[:n].to(H.device)]
