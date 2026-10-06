"""E3 as a convergence curve: how many generic contexts does the finite-difference readout need to reproduce the
released lens? For each activation, perturb up to N contexts of T tokens by +/- eps h at the sources, keep each
context's summed target change, and evaluate the Spearman (top-100, top-1000, all tokens) and relative error of
W_U mean_{c<n} against D h for n = 1, 2, 4, ..., N."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min, pseudo_logits
from sjlens.lens import jlens
from sjlens.lens.jlens import valid_mask
from sjlens.eval.common import load_hf, spearman, background_activations
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
p.add_argument("--layer", type=int, default=11); p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=32)
p.add_argument("--chunk", type=int, default=8); p.add_argument("--n-act", type=int, default=6); p.add_argument("--rho", type=float, default=0.1)
p.add_argument("--device", default="cpu"); p.add_argument("--tag", default="e3_convergence"); a = p.parse_args()
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.tag); os.makedirs(out_dir, exist_ok=True)
log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

tok, m = load_hf(a.model, torch.float32, a.device); q = Qwen3Min(m); l = a.layer
J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file)); D = token_matrix32(q.w, J[l]); del J
ctx, bg = contexts(tok, a.n_ctx, 8, a.T, a.device); ids = torch.stack(ctx); mask = valid_mask(a.T, 16).to(a.device); nT = int(mask.sum())
H = background_activations(q, bg, l, mask)
with torch.no_grad(): hn = torch.cat([q.resid(ids[i:i + a.chunk], l)[:, mask].norm(dim=-1).flatten() for i in range(0, len(ctx), a.chunk)]).mean().item()
log(f"layer {l}: {len(ctx)} contexts x {nT} sources at T={a.T}; source norm {hn:.1f}; {H.shape[0]} background activations")
ns = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= len(ctx)]
results = {"model": a.model, "layer": l, "T": a.T, "nT": nT, "rho": a.rho, "n_ctx": len(ctx), "ns": ns, "acts": []}
for i in range(a.n_act):
    t0 = time.time(); h = H[i]; eps = a.rho * hn / h.norm().item(); per_ctx = []
    for c0 in range(0, len(ctx), a.chunk):
        b = ids[c0:c0 + a.chunk]; m_ = mask.view(1, -1).expand(b.shape[0], -1)
        with torch.no_grad():
            hp, _ = q.prefill(b, inject=(l, m_, eps * h)); hm, _ = q.prefill(b, inject=(l, m_, -eps * h))
            per_ctx.append((hp - hm)[:, mask].sum(1) / (2 * eps * nT))  # [chunk, d]: this context's estimate of J h
        del hp, hm
    per_ctx = torch.cat(per_ctx); exact = D @ h
    top = torch.topk(exact, 100).indices; top1k = torch.topk(exact, 1000).indices; row = {"h_norm": float(h.norm()), "curve": {}}
    for n in ns:
        fd = pseudo_logits(q.w, per_ctx[:n].mean(0))
        row["curve"][n] = {"sp_top100": spearman(fd[top], exact[top]), "sp_top1k": spearman(fd[top1k], exact[top1k]), "sp_all": spearman(fd, exact), "rel_err": float((fd - exact).norm() / exact.norm())}
    # spread between disjoint halves of the contexts: the noise floor of the FD estimate itself
    half = len(ctx) // 2
    if half >= 1:
        f1, f2 = pseudo_logits(q.w, per_ctx[:half].mean(0)), pseudo_logits(q.w, per_ctx[half:2 * half].mean(0))
        row["split_half"] = {"sp_top100": spearman(f1[top], f2[top]), "sp_all": spearman(f1, f2), "rel_diff": float((f1 - f2).norm() / (0.5 * (f1 + f2)).norm())}
    row["top5_lens"] = [tok.decode([t]) for t in top[:5].tolist()]; row["top5_fd"] = [tok.decode([t]) for t in torch.topk(fd, 5).indices.tolist()]
    results["acts"].append(row)
    log(f"act {i} ({time.time() - t0:.0f}s): " + " | ".join(f"n={n}: top100 {row['curve'][n]['sp_top100']:.2f} all {row['curve'][n]['sp_all']:.2f} err {row['curve'][n]['rel_err']:.2f}" for n in ns)
        + (f" | split-half top100 {row['split_half']['sp_top100']:.2f}" if "split_half" in row else "") + f" | lens {row['top5_lens']} fd {row['top5_fd']}")
    json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
agg = {n: {k: sum(r["curve"][n][k] for r in results["acts"]) / len(results["acts"]) for k in ("sp_top100", "sp_top1k", "sp_all", "rel_err")} for n in ns}
results["mean_curve"] = agg; json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
log("mean over activations: " + " | ".join(f"n={n}: top100 {v['sp_top100']:.3f} top1k {v['sp_top1k']:.3f} all {v['sp_all']:.3f} err {v['rel_err']:.3f}" for n, v in agg.items()))
