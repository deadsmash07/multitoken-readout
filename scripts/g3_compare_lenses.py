"""G3: how far apart are two fp32 lenses fitted on disjoint prompt sets, and how far is each from the released lens?

Lens A and lens B come from `phase_a_local.py --fit` on disjoint contexts (runs/<tag>/jlens.pt); the released lens is
the Neuronpedia file (bf16, 466 prompts, its own prompt filter). On held-out background activations h (disjoint from
both fits) compare the readouts D h pairwise: all-token Spearman, relative difference, overlap of the top-10 / top-100
sets and top-1 agreement (no selection on either side), and Spearman over the top-100 of a third lens (so the selected
set is independent of the two being compared). Matrix level: ||J_X - J_Y||_F / ||J_Y||_F. Token level: the lens's own
split-half reliability r_d(w) = cos(D_A[w], D_B[w]) for every token (saved per layer; G5 divides by sqrt(r_d)),
summarised by token class. With A-vs-B as pure prompt-sampling noise at n prompts, released-vs-A in excess of it is the
systematic part (bf16 fit, prompt filter, prompt count)."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min
from sjlens.lens import jlens
from sjlens.lens.jlens import valid_mask
from sjlens.eval.common import load_hf, spearman, background_activations
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download


def classes(tok, V, device):
    c = {"word": [], "piece": [], "punct": [], "digit": []}
    for i in range(V):
        s = tok.decode([i])
        if s.startswith(" ") and s[1:].isalpha(): c["word"].append(i)
        elif s.isalpha(): c["piece"].append(i)
        elif s.strip().isdigit(): c["digit"].append(i)
        elif s.strip() and not any(ch.isalnum() for ch in s): c["punct"].append(i)
    return {k: torch.tensor(v, device=device) for k, v in c.items()}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--lens-a", required=True, help="comma-separated jlens.pt files (merged by layer) for lens A")
    p.add_argument("--lens-b", required=True, help="same for lens B")
    p.add_argument("--layers", default="11,17,22"); p.add_argument("--T", type=int, default=128); p.add_argument("--n-bg", type=int, default=16)
    p.add_argument("--n-act", type=int, default=100); p.add_argument("--device", default="cuda")
    p.add_argument("--contexts", default="runs/contexts/qwen3_T128_256.pt", help="background taken from this file's bg split")
    p.add_argument("--tag", default="g3_compare"); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.tag); os.makedirs(out_dir, exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    tok, m = load_hf(a.model, torch.float32, a.device); q = Qwen3Min(m)
    lens = {"released": jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))[0]}
    for name, files in (("A", a.lens_a), ("B", a.lens_b)):
        lens[name] = {}
        for f in files.split(","): lens[name].update(jlens.load(f)[0])
    _, bg = contexts(tok, 1, a.n_bg, a.T, a.device, path=a.contexts)
    mask = valid_mask(a.T, 16).to(a.device); cls = classes(tok, q.w.lm.shape[0], a.device)
    results = {"model": a.model, "n_bg": len(bg), "lens_a": a.lens_a, "lens_b": a.lens_b, "layers": {}}
    pairs = [("A", "B", "released"), ("A", "released", "B"), ("B", "released", "A")]
    for l in [int(x) for x in a.layers.split(",")]:
        t0 = time.time(); H = background_activations(q, bg, l, mask); g = torch.Generator().manual_seed(l)
        H = H[torch.randperm(H.shape[0], generator=g)[: a.n_act].to(H.device)]
        Js = {k: v[l].float().to(a.device) for k, v in lens.items()}
        r = {"matrix_rel_diff": {f"{x}-{y}": float((Js[x] - Js[y]).norm() / Js[y].norm()) for x, y, _ in pairs}}
        R = {k: token_matrix32(q.w, Js[k]) @ H.T for k in Js}  # [V, n] readouts; D itself is formed once per lens
        R = {k: v.T for k, v in R.items()}  # [n, V]
        for x, y, z in pairs:
            st = {"sp_all": [], "rel_diff": [], "ov10": [], "ov100": [], "top1_eq": [], "sp_top100_third": []}
            for i in range(R[x].shape[0]):
                fx, fy, fz = R[x][i], R[y][i], R[z][i]
                st["sp_all"].append(spearman(fx, fy)); st["rel_diff"].append(float((fx - fy).norm() / (0.5 * (fx + fy)).norm()))
                tx, ty = torch.topk(fx, 100).indices.tolist(), torch.topk(fy, 100).indices.tolist()
                st["ov10"].append(len(set(tx[:10]) & set(ty[:10])) / 10); st["ov100"].append(len(set(tx) & set(ty)) / 100)
                st["top1_eq"].append(float(tx[0] == ty[0])); tz = torch.topk(fz, 100).indices; st["sp_top100_third"].append(spearman(fx[tz], fy[tz]))
            r[f"{x}_vs_{y}"] = {k: sum(v) / len(v) for k, v in st.items()}
            log(f"L{l} {x} vs {y}: " + " ".join(f"{k} {v:.3f}" for k, v in r[f"{x}_vs_{y}"].items()) + f" | ||J_{x}-J_{y}||/||J_{y}|| {r['matrix_rel_diff'][f'{x}-{y}']:.3f}")
        del R
        # per-token lens reliability r_d(w) = cos(D_A[w], D_B[w]), row by row in blocks (D is 1.2 GB per lens)
        rd = torch.empty(q.w.lm.shape[0], device=a.device); rrel = torch.empty_like(rd)
        for s0 in range(0, q.w.lm.shape[0], 16384):
            W = q.w.lm[s0:s0 + 16384].float(); dA, dB, dR = W @ Js["A"], W @ Js["B"], W @ Js["released"]
            rd[s0:s0 + 16384] = torch.nn.functional.cosine_similarity(dA, dB, 1)
            rrel[s0:s0 + 16384] = torch.nn.functional.cosine_similarity(0.5 * (dA + dB), dR, 1)
        r["token_reliability"] = {k: {"median_cos_A_B": float(rd[idx].median()), "p10_cos_A_B": float(rd[idx].quantile(0.1)),
                                      "median_cos_AB_released": float(rrel[idx].median()), "n": int(idx.numel())} for k, idx in cls.items()}
        r["token_reliability"]["all"] = {"median_cos_A_B": float(rd.median()), "median_cos_AB_released": float(rrel.median())}
        torch.save({"cos_A_B": rd.cpu(), "cos_AB_released": rrel.cpu(), "layer": l}, os.path.join(out_dir, f"token_reliability_L{l}.pt"))
        log(f"L{l} token reliability cos(D_A[w], D_B[w]) median by class: " + ", ".join(f"{k} {v['median_cos_A_B']:.3f} (vs released {v['median_cos_AB_released']:.3f})" for k, v in r["token_reliability"].items()))
        r["seconds"] = time.time() - t0; results["layers"][str(l)] = r
        json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
