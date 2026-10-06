"""H1a with an attenuation correction. For each token w: per-context SJ vectors v_c(w) (one VJP each) over N
contexts at T tokens; v = mean over all, v_A and v_B over disjoint halves. cos(d(w), v) is the raw H1a statistic;
cos(v_A, v_B) = r is the split-half reliability of an N/2-context estimate; the estimate being corrected is the
N-context mean, whose reliability is r_N = 2r / (1 + r) (Spearman-Brown), so corrected = cos(d, v) / sqrt(r_N) is
the noise-corrected cosine (the value expected with infinitely many contexts, under independent sampling noise).
corrected_halfrel_old = cos(d, v) / sqrt(r) is the earlier over-correcting formula, kept for comparison.
Also reports cos(d(w), v) for the first n/2 contexts only, to show the trend with N."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min
from sjlens.lens import jlens
from sjlens.lens.backward import sj_vectors
from sjlens.eval.common import load_hf
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
p.add_argument("--layer", type=int, default=11); p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=32)
p.add_argument("--tokens", default=" Professor, tutors, flavour, contact, writing, organis,er,en,a,e,o,h, “,”, \n,.[")
p.add_argument("--strata", type=int, default=0, help="if > 0, ignore --tokens and sample this many tokens per class (word_hi/mid/lo frequency by BPE rank, piece, punct, digit, cjk)")
p.add_argument("--device", default="cpu"); p.add_argument("--contexts", default="contexts/qwen3_T128.pt"); p.add_argument("--tag", default="e8_splithalf"); a = p.parse_args()
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.tag); os.makedirs(out_dir, exist_ok=True)
log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

tok, m = load_hf(a.model, torch.float32, a.device); q = Qwen3Min(m); l = a.layer
J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file)); D = token_matrix32(q.w, J[l]); del J
ctx, _ = contexts(tok, a.n_ctx, 1, a.T, a.device, path=a.contexts if os.path.exists(a.contexts) else None)
toks, strata_of = [], {}
if a.strata:
    V = q.w.lm.shape[0]; cls = {"word_hi": [], "word_mid": [], "word_lo": [], "piece": [], "punct": [], "digit": [], "cjk": []}; words = []
    for i in range(V):
        t = tok.decode([i])
        if t.startswith(" ") and t[1:].isalpha() and t[1:].isascii(): words.append(i)
        elif t.isalpha() and t.isascii(): cls["piece"].append(i)
        elif t.strip().isdigit(): cls["digit"].append(i)
        elif t.strip() and not any(c.isalnum() for c in t) and t.isascii(): cls["punct"].append(i)
        elif any("\u4e00" <= c <= "\u9fff" for c in t): cls["cjk"].append(i)
    words.sort(); n3 = len(words) // 3  # BPE rank (token id) as the frequency proxy: earlier merges are more frequent
    cls["word_hi"], cls["word_mid"], cls["word_lo"] = words[:n3], words[n3:2 * n3], words[2 * n3:]
    gs = torch.Generator().manual_seed(0)
    for k, v in cls.items():
        pick = [v[j] for j in torch.randperm(len(v), generator=gs)[: a.strata].tolist()]
        toks += pick; strata_of.update({w: k for w in pick})
    log("strata: " + ", ".join(f"{k} {len(v)} candidates" for k, v in cls.items()))
else:
    for s in a.tokens.split(","):
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1: toks.append(ids[0])
        else: log(f"skip {s!r}: {len(ids)} tokens {ids}")
log(f"layer {l}: {len(ctx)} contexts at T={a.T}; {len(toks)} single-token targets")
cos = torch.nn.functional.cosine_similarity; results = {"model": a.model, "layer": l, "T": a.T, "n_ctx": len(ctx), "tokens": []}
for w in toks:
    t0 = time.time(); V = torch.stack([sj_vectors(q, [w], [c], [l], 16)[l] for c in ctx])  # [N, d] per-context SJ vectors
    d = D[w].double().cpu(); N = len(ctx); h = N // 2
    v, vA, vB = V.mean(0), V[:h].mean(0), V[h:2 * h].mean(0)
    row = {"token": tok.decode([w]), "id": w, "stratum": strata_of.get(w, ""), "cos_d_v": float(cos(d, v, 0)), "cos_d_vA": float(cos(d, vA, 0)), "split_half": float(cos(vA, vB, 0)),
           "cos_d_v_by_n": {n: float(cos(d, V[:n].mean(0), 0)) for n in (1, 2, 4, 8, 16, 32) if n <= N}, "per_ctx_cos_to_mean": float(cos(V, v[None], 1).mean())}
    r = max(row["split_half"], 1e-6); row["corrected"] = row["cos_d_v"] / (2 * r / (1 + r)) ** 0.5; row["corrected_halfrel_old"] = row["cos_d_v"] / r ** 0.5
    results["tokens"].append(row); json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    log(f"{row['token']!r:12s} ({time.time() - t0:.0f}s): cos(d, v_{N}) {row['cos_d_v']:.3f} | by n: " + " ".join(f"{n}:{c:.2f}" for n, c in row["cos_d_v_by_n"].items())
        + f" | split-half {row['split_half']:.3f} | corrected (Spearman-Brown) {row['corrected']:.3f} (old {row['corrected_halfrel_old']:.3f})")
R = results["tokens"]; med = lambda k: float(torch.tensor([r[k] for r in R]).median())
results["summary"] = {k: med(k) for k in ("cos_d_v", "cos_d_vA", "split_half", "corrected", "corrected_halfrel_old")}
if strata_of:
    results["by_stratum"] = {}
    for st in sorted(set(strata_of.values())):
        rs = [r for r in R if r["stratum"] == st]
        results["by_stratum"][st] = {k: float(torch.tensor([r[k] for r in rs]).median()) for k in ("cos_d_v", "split_half", "corrected", "corrected_halfrel_old")} | {"n": len(rs)}
        b = results["by_stratum"][st]; log(f"stratum {st:9s} (n={len(rs)}): median raw {b['cos_d_v']:.3f} reliability {b['split_half']:.3f} corrected (Spearman-Brown) {b['corrected']:.3f} (old {b['corrected_halfrel_old']:.3f})")
json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
log("median: " + " ".join(f"{k} {v:.3f}" for k, v in results["summary"].items()))
