"""Reliability of the prefix-conditioned readout that J-beam's later steps use.

For an activation h at layer l and a prefix p, the step readout is W_U J(p) h with J(p) the Jacobian to the position
after p, averaged over contexts. J(empty) targets each context's own last token (which varies), while J(p) for a fixed
prefix targets the same token in every context; this measures how much that stabilises the estimate. Per-chunk exact
JVPs give independent estimates from disjoint context sets; we report the agreement (Spearman over the top-100, top-1000,
all tokens; relative difference) between two disjoint halves for each half size, per prefix length 0, 1, 2, alongside
the lens-target readout for the same contexts as the reference case."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min, pseudo_logits
from sjlens.lens import jlens
from sjlens.lens.jlens import valid_mask
from sjlens.lens.forward import jvp_exact, jvp_lens
from sjlens.eval.common import load_hf, spearman, background_activations, sample_activations
from sjlens.lens.score import hutchinson_sigma, nnomp
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
p.add_argument("--layer", type=int, default=11); p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=32)
p.add_argument("--chunk", type=int, default=8); p.add_argument("--n-act", type=int, default=3); p.add_argument("--act-offset", type=int, default=0); p.add_argument("--max-prefix", type=int, default=2)
p.add_argument("--m-null", type=int, default=0, help="null directions for a z-scored agreement (each costs one JVP per chunk and prefix)")
p.add_argument("--inject", default="h,hc", help="which vector to push through J(p): h, hc (h - background mean), hJ (NNOMP J-space projection, k=25)")
p.add_argument("--device", default="cpu"); p.add_argument("--contexts", default="contexts/qwen3_T128.pt"); p.add_argument("--tag", default="e3b_step_reliability"); a = p.parse_args()
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.tag); os.makedirs(out_dir, exist_ok=True)
log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

tok, m = load_hf(a.model, torch.float32, a.device); q = Qwen3Min(m); l = a.layer
J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file)); D = token_matrix32(q.w, J[l]); del J
ctx, bg = contexts(tok, a.n_ctx, 8, a.T, a.device, path=a.contexts if os.path.exists(a.contexts) else None)
ids = torch.stack(ctx); mask = valid_mask(a.T, 16).to(a.device); Hall = background_activations(q, bg, l, mask); H = sample_activations(Hall, a.act_offset + a.n_act)
n_chunks = len(ctx) // a.chunk
V = q.w.lm.shape[0]; classes = {"word": [], "piece": [], "punct": []}
for i in range(V):
    t = tok.decode([i])
    if t.startswith(" ") and t[1:].isalpha(): classes["word"].append(i)
    elif t.isalpha(): classes["piece"].append(i)
    elif t.strip() and not any(c.isalnum() for c in t): classes["punct"].append(i)
classes = {k: torch.tensor(v, device=a.device) for k, v in classes.items()}
mu = Hall.mean(0); Hc = Hall - mu; gnull = torch.Generator().manual_seed(1)
nulls = Hc[torch.randperm(Hc.shape[0], generator=gnull)[: a.m_null]] if a.m_null else None
log(f"layer {l}: {len(ctx)} contexts in {n_chunks} chunks of {a.chunk}; T={a.T}")

def overlap(f1, f2, k):
    """|topk(f1) & topk(f2)| / k: no selection on the mean of the halves (which biases Spearman within the selected set)."""
    return len(set(torch.topk(f1, k).indices.tolist()) & set(torch.topk(f2, k).indices.tolist())) / k

def agreement(R, name):
    """R [n_chunks, V] per-chunk readouts. Agreement between disjoint halves at each half size (in chunks)."""
    out = {}
    for k in [x for x in (1, 2, 4, 8, 16) if 2 * x <= R.shape[0]]:
        pairs = [(R[i:i + k].mean(0), R[i + k:i + 2 * k].mean(0)) for i in range(0, R.shape[0] - 2 * k + 1, 2 * k)]
        s = {"sp_top100": [], "sp_top1k": [], "sp_all": [], "rel_diff": [], "sp_top100_word": [], "sp_top100_piece": [], "sp_top100_punct": [], "ov10": [], "ov100": [], "top1_eq": []}
        for f1, f2 in pairs:
            ref = 0.5 * (f1 + f2); top = torch.topk(ref, 100).indices; top1k = torch.topk(ref, 1000).indices
            s["sp_top100"].append(spearman(f1[top], f2[top])); s["sp_top1k"].append(spearman(f1[top1k], f2[top1k])); s["sp_all"].append(spearman(f1, f2))
            s["rel_diff"].append(float((f1 - f2).norm() / ref.norm()))
            s["ov10"].append(overlap(f1, f2, 10)); s["ov100"].append(overlap(f1, f2, 100)); s["top1_eq"].append(float(int(f1.argmax()) == int(f2.argmax())))
            for cname, idx in classes.items():
                tk = idx[torch.topk(ref[idx], 100).indices]; s[f"sp_top100_{cname}"].append(spearman(f1[tk], f2[tk]))
        out[k * a.chunk] = {kk: sum(v) / len(v) for kk, v in s.items()}
    return out

results = {"model": a.model, "layer": l, "T": a.T, "n_ctx": len(ctx), "chunk": a.chunk, "acts": []}
for i in range(a.n_act):
    h0 = H[a.act_offset + i]; row = {"h_norm": float(h0.norm()), "readouts": {}}; t0 = time.time()
    # reference: the lens-target readout (sum over all targets) on the same contexts, with the raw activation
    R = torch.stack([pseudo_logits(q.w, jvp_lens(q, ids[c0:c0 + a.chunk], l, mask, h0)) for c0 in range(0, len(ctx), a.chunk)])
    row["readouts"]["lens_target"] = {"agreement": agreement(R, "lens"), "top5": [tok.decode([t]) for t in torch.topk(R.mean(0), 5).indices.tolist()]}
    for inj in a.inject.split(","):
      h = {"h": h0, "hc": h0 - mu, "hJ": nnomp(D, h0, 25)[1]}[inj]
      prefix = []
      for n in range(0, a.max_prefix + 1):
        R = torch.stack([pseudo_logits(q.w, jvp_exact(q, ids[c0:c0 + a.chunk], l, mask, h, prefix=prefix)) for c0 in range(0, len(ctx), a.chunk)])
        mean = R.mean(0); nxt = int(torch.topk(mean, 1).indices)
        row["readouts"][f"{inj}_prefix_len_{n}"] = {"prefix": list(prefix), "prefix_text": tok.decode(prefix), "agreement": agreement(R, f"p{n}"),
                                                    "top5": [tok.decode([t]) for t in torch.topk(mean, 5).indices.tolist()]}
        if nulls is not None:
            # z-scored readout per chunk: numerator over sigma from the same chunk's null readouts (what J-beam ranks by)
            N = torch.stack([torch.stack([pseudo_logits(q.w, jvp_exact(q, ids[c0:c0 + a.chunk], l, mask, gj, prefix=prefix)) for c0 in range(0, len(ctx), a.chunk)]) for gj in nulls])  # [m, chunks, V]
            def zhalf(R_, N_):  # agreement of z computed within each half
                out = {}
                for k in [x for x in (1, 2, 4, 8, 16) if 2 * x <= R_.shape[0]]:
                    pairs = []
                    for i0 in range(0, R_.shape[0] - 2 * k + 1, 2 * k):
                        zA = R_[i0:i0 + k].mean(0) / hutchinson_sigma(N_[:, i0:i0 + k].mean(1)); zB = R_[i0 + k:i0 + 2 * k].mean(0) / hutchinson_sigma(N_[:, i0 + k:i0 + 2 * k].mean(1)); pairs.append((zA, zB))
                    sref = {"sp_top100": [], "sp_all": [], "sp_top100_word": [], "ov10": [], "ov100": [], "top1_eq": [], "ov10_word": []}
                    for zA, zB in pairs:
                        ref = 0.5 * (zA + zB); top = torch.topk(ref, 100).indices; tw = classes["word"][torch.topk(ref[classes["word"]], 100).indices]
                        sref["sp_top100"].append(spearman(zA[top], zB[top])); sref["sp_all"].append(spearman(zA, zB)); sref["sp_top100_word"].append(spearman(zA[tw], zB[tw]))
                        sref["ov10"].append(overlap(zA, zB, 10)); sref["ov100"].append(overlap(zA, zB, 100)); sref["top1_eq"].append(float(int(zA.argmax()) == int(zB.argmax())))
                        wi = classes["word"]; sref["ov10_word"].append(overlap(zA[wi], zB[wi], 10))
                    out[k * a.chunk] = {kk: sum(v) / len(v) for kk, v in sref.items()}
                return out
            zfull = mean / hutchinson_sigma(N.mean(1))
            row["readouts"][f"{inj}_prefix_len_{n}"]["z_agreement"] = zhalf(R, N); row["readouts"][f"{inj}_prefix_len_{n}"]["top5_z"] = [tok.decode([t]) for t in torch.topk(zfull, 5).indices.tolist()]
            nxt = int(torch.topk(zfull, 1).indices)
        # J-beam's greedy choice from the full-context estimate extends the prefix; at n=0 use the exact lens top token instead
        prefix = prefix + [int(torch.topk(D @ h0, 1).indices) if n == 0 else nxt]
    row["seconds"] = time.time() - t0; results["acts"].append(row)
    json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    for name, r in row["readouts"].items():
        ag = r["agreement"]
        log(f"act {i} {name:12s} {r.get('prefix_text', '')!r:>14}: " + " | ".join(f"n={n}: top100 {v['sp_top100']:.2f} ov10 {v['ov10']:.2f} ov100 {v['ov100']:.2f} words {v['sp_top100_word']:.2f} all {v['sp_all']:.2f} rel {v['rel_diff']:.2f}" for n, v in ag.items()) + f" | top5 {r['top5']}")
        if "z_agreement" in r:
            log(f"act {i} {name:12s} z-scored: " + " | ".join(f"n={n}: top100 {v['sp_top100']:.2f} ov10 {v['ov10']:.2f} ov100 {v['ov100']:.2f} ov10_word {v['ov10_word']:.2f} all {v['sp_all']:.2f}" for n, v in r["z_agreement"].items()) + f" | top5 z {r['top5_z']}")
    log(f"act {i} done in {row['seconds']:.0f}s")
names = list(results["acts"][0]["readouts"]); results["mean"] = {}
for name in names:
    ks = results["acts"][0]["readouts"][name]["agreement"].keys()
    results["mean"][name] = {k: {kk: sum(r["readouts"][name]["agreement"][k][kk] for r in results["acts"]) / len(results["acts"]) for kk in ("sp_top100", "sp_top1k", "sp_all", "rel_diff", "sp_top100_word", "sp_top100_piece", "sp_top100_punct", "ov10", "ov100", "top1_eq")} for k in ks}
    if "z_agreement" in results["acts"][0]["readouts"][name]:
        kz = results["acts"][0]["readouts"][name]["z_agreement"].keys()
        results["mean"][name + "_z"] = {k: {kk: sum(r["readouts"][name]["z_agreement"][k][kk] for r in results["acts"]) / len(results["acts"]) for kk in ("sp_top100", "sp_all", "ov10", "ov100", "top1_eq", "ov10_word")} for k in kz}
json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
for name, ag in results["mean"].items():
    log(f"mean {name:14s}: " + " | ".join(f"n={n}: top100 {v['sp_top100']:.2f} ov10 {v['ov10']:.2f} ov100 {v['ov100']:.2f} top1 {v['top1_eq']:.2f} all {v['sp_all']:.2f}" for n, v in ag.items()))
