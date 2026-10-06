"""E7b / G4: the gap-0 term of the lens against its future term, on activations and on tokens.

For an activation h at layer l: gap0(h) = mean_t J_{tt} h (perturb one source position at a time and read the
final residual at the same position; batched over positions), future(h) = J_lens h - gap0(h) (jvp_lens). Both
read out through W_U. Reports, per activation, the Spearman and cosine between the two readouts over all tokens
and restricted to token classes (space-prefixed words, word-internal pieces, punctuation), and the norm ratio.
With --tokens, also the per-token decomposition d(w) = G_0^T W_U[w] + rest via per-target VJPs (cotangent W_U[w]
at one target position; the gradient at every source gives all gaps at once), reporting cos(G_0^T W_U[w], rest)
and the share of ||d(w)|| carried by gap 0. Today's claim: gap 0 is aligned with the future term for words, not
for pieces."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min, pseudo_logits, forward
from sjlens.lens import jlens
from sjlens.lens.jlens import valid_mask
from sjlens.lens.forward import jvp_lens, gap0_jvp
from sjlens.eval.common import load_hf, load_q, spearman, background_activations, sample_activations
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download


def token_classes(tok, V):
    cls = {"word": [], "piece": [], "punct": []}
    for i in range(V):
        s = tok.decode([i])
        if s.startswith(" ") and s[1:].isalpha(): cls["word"].append(i)
        elif s.isalpha(): cls["piece"].append(i)
        elif s.strip() and not any(c.isalnum() for c in s): cls["punct"].append(i)
    return {k: torch.tensor(v) for k, v in cls.items()}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--layers", default="11"); p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=8); p.add_argument("--n-act", type=int, default=8)
    p.add_argument("--tokens", default="", help="comma-separated token strings for the per-token VJP decomposition (GPU)")
    p.add_argument("--device", default="cpu"); p.add_argument("--contexts", default="contexts/qwen3_T128.pt"); p.add_argument("--tag", default="e7b_gap0"); p.add_argument("--dtype", default="fp32", help="fp32 | bf16 | bf16-mixed (bf16 weights, fp32 activations; = fp32 for Qwen3, half the memory)"); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.tag); os.makedirs(out_dir, exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    tok, m, q = load_q(a.model, a.dtype, a.device)
    J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))
    ctx, bg = contexts(tok, a.n_ctx, 8, a.T, a.device, path=a.contexts if os.path.exists(a.contexts) else None)
    ids = torch.stack(ctx); mask = valid_mask(a.T, 16).to(a.device); cls = {k: v.to(a.device) for k, v in token_classes(tok, q.w.lm.shape[0]).items()}
    log(f"token classes: " + ", ".join(f"{k} {len(v)}" for k, v in cls.items()))
    results = {"model": a.model, "dtype": a.dtype, "T": a.T, "n_ctx": len(ctx), "layers": {}}
    for l in [int(x) for x in a.layers.split(",")]:
        t0 = time.time(); D = token_matrix32(q.w, J[l]); H = sample_activations(background_activations(q, bg, l, mask), a.n_act, seed=l); rows = []
        for i in range(a.n_act):
            h = H[i]; g0 = gap0_jvp(q, ids, l, mask, h); full = jvp_lens(q, ids, l, mask, h); fut = full - g0
            r0, rf, rl = pseudo_logits(q.w, g0), pseudo_logits(q.w, fut), D @ h
            row = {"norm_ratio_gap0_over_future": float(g0.norm() / fut.norm()), "cos_vec": float(torch.nn.functional.cosine_similarity(g0, fut, 0)),
                   "sp_all": spearman(r0, rf), "sp_lens_vs_jvp_all": spearman(rl, r0 + rf)}
            for k, idx in cls.items():
                top = idx[torch.topk(rl[idx], min(100, len(idx))).indices]
                row[f"sp_top100_{k}"] = spearman(r0[top], rf[top]); row[f"cos_top100_{k}"] = float(torch.nn.functional.cosine_similarity(r0[top], rf[top], 0))
            rows.append(row); log(f"L{l} act {i}: |gap0|/|future| {row['norm_ratio_gap0_over_future']:.2f}, vec cos {row['cos_vec']:.2f}; readout Spearman gap0 vs future: all {row['sp_all']:.2f}, "
                                  + ", ".join(f"{k} top-100 {row[f'sp_top100_{k}']:.2f}" for k in cls))
        summ = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}; summ["rows"] = rows
        if a.tokens:
            summ["tokens"] = {}
            for s in a.tokens.split(","):
                w = tok.encode(s, add_special_tokens=False)
                if len(w) != 1: continue
                w = w[0]; srcs = mask.nonzero().flatten().tolist(); gap0v = torch.zeros(q.d, dtype=torch.float64); allv = torch.zeros(q.d, dtype=torch.float64); n0 = 0
                for c in ctx:
                    c = c.view(1, -1); src = torch.zeros(1, a.T, q.d, dtype=q.act_dtype or q.w.emb.dtype, device=a.device, requires_grad=True)
                    hL, _ = forward(q.w, c, inject=(l, None, src), act_dtype=q.act_dtype); cot = q.w.lm[w].to(hL.dtype)
                    for tp in srcs:
                        gr, = torch.autograd.grad((hL[0, tp] * cot).sum(), src, retain_graph=True)
                        gap0v += gr[0, tp].double().cpu(); allv += gr[0, mask.cpu()].double().sum(0).cpu(); n0 += 1
                gap0v /= n0; allv /= n0; rest = allv - gap0v; d = D[w].double().cpu()
                summ["tokens"][s] = {"cos_gap0_rest": float(torch.nn.functional.cosine_similarity(gap0v, rest, 0)), "share_gap0": float(gap0v.norm() / allv.norm()),
                                     "cos_d_vs_recon": float(torch.nn.functional.cosine_similarity(d, allv, 0)), "cos_gap0_vs_WU": float(torch.nn.functional.cosine_similarity(gap0v, q.w.lm[w].double().cpu(), 0))}
                log(f"L{l} token {s!r}: cos(gap0 term, rest) {summ['tokens'][s]['cos_gap0_rest']:.3f}; gap-0 share of norm {summ['tokens'][s]['share_gap0']:.2f}; cos(gap0 term, W_U[w]) {summ['tokens'][s]['cos_gap0_vs_WU']:.3f}; recon vs d(w) {summ['tokens'][s]['cos_d_vs_recon']:.3f}")
        summ["seconds"] = time.time() - t0; results["layers"][str(l)] = summ; json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
        log(f"L{l} mean: |gap0|/|future| {summ['norm_ratio_gap0_over_future']:.2f}; readout Spearman gap0 vs future all {summ['sp_all']:.2f}, words {summ['sp_top100_word']:.2f}, pieces {summ['sp_top100_piece']:.2f}, punct {summ['sp_top100_punct']:.2f}; {summ['seconds']:.0f}s")


if __name__ == "__main__":
    main()
