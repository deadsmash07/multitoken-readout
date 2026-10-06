"""E11 / H1c: the fragment test. Does the phrase vector of a word's pieces retrieve the word's own lens vector?

Words: single-token, space-prefixed, alphabetic tokens that also admit a two-piece tokenisation
(' Estonian' -> ' Eston' + 'ian', both single tokens, same surface). For each word and layer:
  pure SJ      v_z([p1, p2])                       (lens/backward.py, mode 'z')
  hybrid       d(p1) + J(p1)^T W_U[p2]             (mode 'hybrid')
  first piece  d(p1)                               (control)
  mean pieces  (d(p1) + d(p2)) / 2                 (control)
Retrieval: cosine of each vector against the lens rows of all sampled words; rank of the true word. Reports top-1
rate, mean rank, mean cosine to the true word and to the best wrong word, per vector type. Pass (hybrid): top-1
>= 90%; kill < 50%."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min
from sjlens.lens import jlens
from sjlens.lens.backward import sj_vectors
from sjlens.eval.common import load_hf
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download


def find_splits(tok, n_words, min_len=6, seed=0):
    """(word_id, p1_id, p2_id, word) for single-token space-prefixed alphabetic words with a two-single-token split."""
    vocab = tok.get_vocab(); inv = {v: k for k, v in vocab.items()}
    g = torch.Generator().manual_seed(seed); ids = torch.randperm(len(inv), generator=g).tolist(); out = []
    for i in ids:
        s = tok.decode([i])
        if not (s.startswith(" ") and s[1:].isalpha() and s[1:].islower() and len(s) - 1 >= min_len): continue
        if tok.encode(s, add_special_tokens=False) != [i]: continue
        w = s[1:]; found = None
        for k in sorted(range(2, len(w) - 1), key=lambda k: abs(k - len(w) / 2)):
            a, b = tok.encode(" " + w[:k], add_special_tokens=False), tok.encode(w[k:], add_special_tokens=False)
            if len(a) == 1 and len(b) == 1 and tok.decode(a + b) == s: found = (a[0], b[0]); break
        if found: out.append((i, found[0], found[1], s))
        if len(out) == n_words: break
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--layers", default="11"); p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=64); p.add_argument("--n-words", type=int, default=500)
    p.add_argument("--device", default="cpu"); p.add_argument("--contexts", default="contexts/qwen3_T128.pt"); p.add_argument("--tag", default="e11_fragment"); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.tag); os.makedirs(out_dir, exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    tok, m = load_hf(a.model, torch.float32, a.device); q = Qwen3Min(m)
    J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))
    ctx, _ = contexts(tok, a.n_ctx, 1, a.T, a.device, path=a.contexts if os.path.exists(a.contexts) else None)
    words = find_splits(tok, a.n_words); log(f"{len(words)} words with two-piece splits, e.g. {[(w, tok.decode([p1]), tok.decode([p2])) for _, p1, p2, w in words[:5]]}")
    results = {"model": a.model, "n_ctx": len(ctx), "T": a.T, "n_words": len(words), "words": [(w, tok.decode([p1]), tok.decode([p2])) for _, p1, p2, w in words], "layers": {}}
    cos = torch.nn.functional.cosine_similarity
    for l in [int(x) for x in a.layers.split(",")]:
        t0 = time.time(); D = token_matrix32(q.w, J[l]); Dl = {l: D}
        W = D[[w for w, *_ in words]].double().cpu(); Wn = W / W.norm(dim=1, keepdim=True)
        # D rows sum over ~|T| targets (mean over sources), while SJ step terms are single-target Jacobians, so the raw
        # hybrid d(p1) + J(p1)^T W_U[p2] is ~(|T|-1)/2 times dominated by its first term. hybrid_normmatched rescales
        # d(p1) to the norm of the pure-SJ first term; hybrid_T scales the conditional term by (|T|-1)/2 (Cor. 3.1).
        vecs = {"pure_sj": [], "hybrid": [], "hybrid_normmatched": [], "hybrid_T": [], "cond_term_only": [], "first_piece": [], "mean_pieces": []}
        nT = 111; ratios = []
        for n, (w, p1, p2, s) in enumerate(words):
            pure = sj_vectors(q, [p1, p2], ctx, [l], 16, mode="z")[l]; hyb = sj_vectors(q, [p1, p2], ctx, [l], 16, mode="hybrid", D=Dl)[l]
            d1 = D[p1].double().cpu(); t2 = hyb - d1; t1 = pure - t2; ratios.append(float(t2.norm() / d1.norm()))
            vecs["pure_sj"].append(pure); vecs["hybrid"].append(hyb); vecs["cond_term_only"].append(t2)
            vecs["hybrid_normmatched"].append(d1 * (t1.norm() / d1.norm()) + t2); vecs["hybrid_T"].append(d1 + (nT - 1) / 2 * t2)
            vecs["first_piece"].append(D[p1].double().cpu()); vecs["mean_pieces"].append(0.5 * (D[p1] + D[p2]).double().cpu())
            if n % 25 == 0: log(f"L{l} {n}/{len(words)} {s!r}")
        r = {}
        for name, vs in vecs.items():
            Vv = torch.stack(vs); Vn = Vv / Vv.norm(dim=1, keepdim=True); S = Vn @ Wn.T  # [n_words, n_words] cosines
            true = S.diag(); ranks = (S > true[:, None]).sum(1) + 1
            wrong = S.clone(); wrong.fill_diagonal_(-2); best_wrong = wrong.max(1).values
            r[name] = {"top1": float((ranks == 1).float().mean()), "top5": float((ranks <= 5).float().mean()), "mean_rank": float(ranks.float().mean()),
                       "median_rank": float(ranks.float().median()), "cos_true_mean": float(true.mean()), "cos_best_wrong_mean": float(best_wrong.mean()), "ranks": ranks.tolist()}
            log(f"L{l} {name:12s}: top-1 {r[name]['top1']:.3f} top-5 {r[name]['top5']:.3f} median rank {r[name]['median_rank']:.0f} | cos true {r[name]['cos_true_mean']:.3f} vs best wrong {r[name]['cos_best_wrong_mean']:.3f}")
        r["cond_term_norm_over_d_p1_median"] = float(torch.tensor(ratios).median()); log(f"L{l}: ||J(p1)^T W_U[p2]|| / ||d(p1)|| median {r['cond_term_norm_over_d_p1_median']:.4f}")
        r["seconds"] = time.time() - t0; results["layers"][str(l)] = r
        json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
        log(f"L{l}: hybrid top-1 {r['hybrid']['top1']:.3f} (pass >= 0.9, kill < 0.5); pure SJ {r['pure_sj']['top1']:.3f}; first piece {r['first_piece']['top1']:.3f}; {r['seconds']:.0f}s")


if __name__ == "__main__":
    main()
