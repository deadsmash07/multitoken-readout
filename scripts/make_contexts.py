"""Tokenize and save generic + background wikitext contexts so a job on a node without internet can run Phase A.

    python scripts/make_contexts.py --model Qwen/Qwen3-1.7B --T 128 --n-ctx 64 --n-bg 64 --out contexts/qwen3_T128.pt
Then: python scripts/phase_a_local.py ... --contexts contexts/qwen3_T128.pt
Qwen3 models share one tokenizer, so a file made with one Qwen3 size serves all of them."""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.eval.common import wikitext_contexts

p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B"); p.add_argument("--T", type=int, default=128)
p.add_argument("--n-ctx", type=int, default=64); p.add_argument("--n-bg", type=int, default=64); p.add_argument("--out", required=True)
p.add_argument("--seed-ctx", type=int, default=0); p.add_argument("--seed-bg", type=int, default=1)
p.add_argument("--exclude", default="", help="contexts .pt files whose records must not reappear (for a disjoint set)"); a = p.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
seen = {tuple(c.tolist()) for f in filter(None, a.exclude.split(",")) for k in ("ctx", "bg") for c in torch.load(f)[k]}
def draw(n, seed):
    out = [c for c in wikitext_contexts(tok, n + len(seen) // 4 + 16, a.T, seed=seed) if tuple(c.tolist()) not in seen][:n]
    assert len(out) == n, f"only {len(out)} disjoint records"; seen.update(tuple(c.tolist()) for c in out); return out
ctx = draw(a.n_ctx, a.seed_ctx); bg = draw(a.n_bg, a.seed_bg)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
torch.save({"model": a.model, "T": a.T, "ctx": ctx, "bg": bg, "source": f"Salesforce/wikitext wikitext-103-raw-v1 train, seeds {a.seed_ctx} (ctx) and {a.seed_bg} (bg), excluding {a.exclude or None}"}, a.out)
print(f"wrote {a.out}: {len(ctx)} contexts + {len(bg)} background, T={a.T}; first context: {tok.decode(ctx[0][:20])!r}...")
