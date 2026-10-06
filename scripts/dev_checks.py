"""Week-1 dev checks on a real model (GPU). Not runnable in this sandbox; each step maps to a Hypothesis id.

python scripts/dev_checks.py --model Qwen/Qwen3-14B --lens qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt --layers 16 24 32
"""
import argparse, json, math, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from sjlens.model.qwen3_min import Qwen3Min, pseudo_logits
from sjlens.lens import jlens
from sjlens.lens.backward import sj_vectors
from sjlens.lens.forward import ContextBank, jvp_exact
from sjlens.lens.score import z_bg, hutchinson_sigma, conformal_threshold, z_fw
from sjlens.beam.jbeam import JBeam, Cfg


def wikitext_contexts(tok, n, T=128, seed=0):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True).shuffle(seed=seed, buffer_size=10000)
    out = []
    for r in ds:
        if len(r["text"]) < 600: continue
        ids = tok(r["text"], return_tensors="pt").input_ids[0, :T]
        if ids.numel() == T: out.append(ids)
        if len(out) == n: break
    return out


def main(a):
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32 if a.fp32 else torch.bfloat16, attn_implementation="eager").to(dev).eval()
    for p in m.parameters(): p.requires_grad_(False)
    q = Qwen3Min(m)
    J, meta = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens))
    ctx = [c.to(dev) for c in wikitext_contexts(tok, a.n_ctx)]
    bg = [c.to(dev) for c in wikitext_contexts(tok, 200, seed=1)]
    log = {}
    for l in a.layers:
        D = jlens.token_matrix(q.w, J[l].to(dev).to(q.w.lm.dtype)).float()
        bank = ContextBank(q, ctx, skip_first=16)
        with torch.no_grad():
            H = torch.cat([q.resid(c.view(1, -1), l)[0, bank.mask] for c in bg]).float()
        mu = H.mean(0); Hc = H - mu
        hn = H.norm(dim=-1).mean().item()
        # day 2: side result (identity 4) vs released lens, Spearman over top-100
        h = H[0]
        with torch.no_grad():
            eps = 0.1 * hn / h.norm().item()
            m_ = bank.mask.view(1, -1).expand(bank.C, -1)
            hp, _ = q.prefill(bank.ids, inject=(l, m_, eps * h.to(q.w.emb.dtype)))
            hm, _ = q.prefill(bank.ids, inject=(l, m_, -eps * h.to(q.w.emb.dtype)))
            fd = (pseudo_logits(q.w, hp[:, bank.mask]) - pseudo_logits(q.w, hm[:, bank.mask])).sum(1).mean(0).float() / (2 * eps * bank.nT)
        exact = D @ h
        top = torch.topk(exact, 100).indices
        sp = torch.corrcoef(torch.stack([fd[top].argsort().argsort().float(), exact[top].argsort().argsort().float()]))[0, 1].item()
        # day 3: forward vs backward on one (h, prefix)
        pf = [int(torch.topk(exact, 1).indices)]
        v = sj_vectors(q, pf + [int(torch.topk(exact, 2).indices[1])], ctx[:8], [l], 16)[l].to(dev)
        cp, cm = bank.perturbed(l, h.to(q.w.emb.dtype), eps)
        num = bank.step_numerators(pf, cp, cm, eps).float()
        # day 6: gap profile is expensive on a 14B; sample 4 contexts x 64 output dims
        log[l] = dict(resid_norm=hn, side_result_spearman=sp, step_num_top=[int(i) for i in torch.topk(num, 10).indices],
                      z_fw_vocab=z_fw(D.shape[0]))
        print(l, log[l])
    json.dump(log, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B"); p.add_argument("--lens", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[16, 24, 32]); p.add_argument("--n_ctx", type=int, default=32)
    p.add_argument("--fp32", action="store_true"); p.add_argument("--out", default="runs/dev_checks.json")
    main(p.parse_args())
