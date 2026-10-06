"""E3 with the exact lens-estimator JVP (sum over masked targets) instead of finite differences, to split the residual against the released lens into
(a) finite-difference step error and (b) context sampling. For each activation h at layer l: J h by exact
forward-mode AD over N contexts (fp32, no eps), central FD at several rho on the same contexts, and the lens's D h.
Reports Spearman (top-100 / top-1000 / all) and relative error of JVP vs lens, FD vs JVP, FD vs lens, and the
split-half agreement of the JVP over disjoint context halves (pure context-sampling noise)."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min, pseudo_logits
from sjlens.lens import jlens
from sjlens.lens.jlens import valid_mask
from sjlens.lens.forward import jvp_lens
from sjlens.eval.common import load_hf, spearman, background_activations, sample_activations
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
p.add_argument("--layer", type=int, default=11); p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=32)
p.add_argument("--chunk", type=int, default=8); p.add_argument("--n-act", type=int, default=4); p.add_argument("--rhos", default="0.02,0.05,0.1")
p.add_argument("--device", default="cpu"); p.add_argument("--contexts", default="contexts/qwen3_T128.pt"); p.add_argument("--tag", default="e3_jvp"); a = p.parse_args()
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.tag); os.makedirs(out_dir, exist_ok=True)
log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

tok, m = load_hf(a.model, torch.float32, a.device); q = Qwen3Min(m); l = a.layer
J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file)); D = token_matrix32(q.w, J[l]); del J
ctx, bg = contexts(tok, a.n_ctx, 8, a.T, a.device, path=a.contexts if os.path.exists(a.contexts) else None)
ids = torch.stack(ctx); mask = valid_mask(a.T, 16).to(a.device); nT = int(mask.sum()); H = sample_activations(background_activations(q, bg, l, mask), a.n_act)
with torch.no_grad(): hn = torch.cat([q.resid(ids[i:i + a.chunk], l)[:, mask].norm(dim=-1).flatten() for i in range(0, len(ctx), a.chunk)]).mean().item()
rhos = [float(x) for x in a.rhos.split(",")]
log(f"layer {l}: {len(ctx)} contexts x {nT} sources at T={a.T}; source norm {hn:.1f}")
def stats(x, ref):
    top = torch.topk(ref, 100).indices; top1k = torch.topk(ref, 1000).indices
    return {"sp_top100": spearman(x[top], ref[top]), "sp_top1k": spearman(x[top1k], ref[top1k]), "sp_all": spearman(x, ref), "rel_err": float((x - ref).norm() / ref.norm())}
results = {"model": a.model, "layer": l, "T": a.T, "nT": nT, "n_ctx": len(ctx), "rhos": rhos, "acts": []}
for i in range(a.n_act):
    t0 = time.time(); h = H[i]; exact_lens = D @ h
    jv = []
    for c0 in range(0, len(ctx), a.chunk):
        jv.append(jvp_lens(q, ids[c0:c0 + a.chunk], l, mask, h))  # [d] = J_lens h estimated on this chunk's contexts
    jv = torch.stack(jv); jvp_all = pseudo_logits(q.w, jv.mean(0)); half = len(jv) // 2
    row = {"h_norm": float(h.norm()), "jvp_vs_lens": stats(jvp_all, exact_lens), "seconds_jvp": time.time() - t0}
    if half:
        f1, f2 = pseudo_logits(q.w, jv[:half].mean(0)), pseudo_logits(q.w, jv[half:2 * half].mean(0))
        row["jvp_split_half"] = {**stats(f1, f2), "rel_diff": float((f1 - f2).norm() / (0.5 * (f1 + f2)).norm())}
    row["fd"] = {}
    for rho in rhos:
        eps = rho * hn / h.norm().item(); acc = torch.zeros(q.d, device=a.device)
        for c0 in range(0, len(ctx), a.chunk):
            b = ids[c0:c0 + a.chunk]; m_ = mask.view(1, -1).expand(b.shape[0], -1)
            with torch.no_grad():
                hp, _ = q.prefill(b, inject=(l, m_, eps * h)); hm, _ = q.prefill(b, inject=(l, m_, -eps * h))
                acc += (hp - hm)[:, mask].sum(1).sum(0) / (2 * eps * nT)
            del hp, hm
        fd = pseudo_logits(q.w, acc / len(ctx))
        row["fd"][str(rho)] = {"vs_jvp": stats(fd, jvp_all), "vs_lens": stats(fd, exact_lens)}
    row["top5_lens"] = [tok.decode([t]) for t in torch.topk(exact_lens, 5).indices.tolist()]; row["top5_jvp"] = [tok.decode([t]) for t in torch.topk(jvp_all, 5).indices.tolist()]
    results["acts"].append(row); json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    jl = row["jvp_vs_lens"]; sh = row.get("jvp_split_half", {})
    log(f"act {i} ({time.time() - t0:.0f}s): JVP vs lens top100 {jl['sp_top100']:.3f} top1k {jl['sp_top1k']:.3f} all {jl['sp_all']:.3f} err {jl['rel_err']:.3f}"
        + (f" | JVP split-half top100 {sh['sp_top100']:.3f} rel_diff {sh['rel_diff']:.3f}" if sh else "")
        + " | FD vs JVP: " + ", ".join(f"rho={r} err {row['fd'][str(r)]['vs_jvp']['rel_err']:.3f}" for r in rhos)
        + " | FD vs lens top100: " + ", ".join(f"rho={r} {row['fd'][str(r)]['vs_lens']['sp_top100']:.3f}" for r in rhos)
        + f" | lens {row['top5_lens']} jvp {row['top5_jvp']}")
keys = ("sp_top100", "sp_top1k", "sp_all", "rel_err"); A = results["acts"]
results["mean"] = {"jvp_vs_lens": {k: sum(r["jvp_vs_lens"][k] for r in A) / len(A) for k in keys},
                   "fd_vs_jvp": {str(r): {k: sum(x["fd"][str(r)]["vs_jvp"][k] for x in A) / len(A) for k in keys} for r in rhos},
                   "fd_vs_lens": {str(r): {k: sum(x["fd"][str(r)]["vs_lens"][k] for x in A) / len(A) for k in keys} for r in rhos}}
json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
log("mean: JVP vs lens " + " ".join(f"{k} {v:.3f}" for k, v in results["mean"]["jvp_vs_lens"].items()))
for r in rhos: log(f"mean: FD rho={r} vs JVP " + " ".join(f"{k} {v:.3f}" for k, v in results["mean"]["fd_vs_jvp"][str(r)].items()) + " | vs lens " + " ".join(f"{k} {v:.3f}" for k, v in results["mean"]["fd_vs_lens"][str(r)].items()))
