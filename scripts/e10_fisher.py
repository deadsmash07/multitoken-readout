"""E10 / H10: the lens Fisher metric F_l = E_c Cov_{w ~ p0(.|c)}[d_l(w)] and verbalizable information.

Per layer: F_l from the model's own next-token law at the last position of each context (true logits, through the
final norm) and the lens token matrix D_l; its spectrum and participation ratio; VI_lambda(h) = lambda^2/2 h^T F h
for background activations against norm-matched random directions; the share of NNOMP atoms (k = 25) of a few
activations that lies in the top-k eigenspace of F_l. With --center-frequency, the corpus-frequency (unigram) direction is projected out of every lens row before
forming F (subtracting a constant vector would be a no-op for a covariance)."""
import argparse, json, math, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min, logits as true_logits
from sjlens.lens import jlens
from sjlens.lens.jlens import valid_mask
from sjlens.lens.score import nnomp, vi_tokens
from sjlens.eval.common import load_hf, background_activations
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download


def fisher(D, P):
    """D [V, d], P [C, V] next-token distributions. Returns E_c [D^T (diag p_c - p_c p_c^T) D] as [d, d] float64."""
    F = torch.zeros(D.shape[1], D.shape[1], dtype=torch.float64, device=D.device)
    for p in P:
        Dp = D * p.sqrt()[:, None]; mean = (p[:, None] * D).sum(0)
        F += (Dp.T @ Dp).double() - torch.outer(mean, mean).double()
    return F / P.shape[0]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--layers", default="11"); p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=16)
    p.add_argument("--n-bg", type=int, default=8); p.add_argument("--k", type=int, default=25); p.add_argument("--center-frequency", action="store_true")
    p.add_argument("--device", default="cpu"); p.add_argument("--contexts", default="contexts/qwen3_T128.pt"); p.add_argument("--tag", default="e10_fisher"); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.tag); os.makedirs(out_dir, exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    tok, m = load_hf(a.model, torch.float32, a.device); q = Qwen3Min(m)
    J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))
    ctx, bg = contexts(tok, a.n_ctx, a.n_bg, a.T, a.device, path=a.contexts if os.path.exists(a.contexts) else None)
    ids = torch.stack(ctx); mask = valid_mask(a.T, 16).to(a.device)
    with torch.no_grad():
        hL, _ = q.prefill(ids); P = true_logits(q.w, hL[:, -1]).float().softmax(-1)  # the model's generic next-token law per context
        Pm = true_logits(q.w, hL[:, mask]).float().softmax(-1).mean((0, 1))  # corpus-frequency proxy for the centring ablation
    results = {"model": a.model, "T": a.T, "n_ctx": len(ctx), "center_frequency": a.center_frequency, "layers": {}}
    for l in [int(x) for x in a.layers.split(",")]:
        t0 = time.time(); D = token_matrix32(q.w, J[l])
        if a.center_frequency:
            # F is a covariance over tokens, so subtracting a constant from every row is a no-op; instead project the
            # frequency (unigram) direction out of every lens row.
            u = (Pm[:, None] * D).sum(0); u = u / u.norm(); D = D - torch.outer(D @ u, u)
        F = fisher(D, P); ev, U = torch.linalg.eigh(F); ev = ev.flip(0).clamp_min(0); U = U.flip(1)
        pr = float(ev.sum() ** 2 / (ev ** 2).sum()); share = ev / ev.sum()
        H = background_activations(q, bg, l, mask); hn = H.norm(dim=-1).mean(); nT = int(mask.sum())
        with torch.no_grad(): z0 = true_logits(q.w, hL[:, -1]).float()  # the model's own next-token logits per context
        quad = lambda h, lam: float(lam ** 2 / 2 * (h.double() @ F @ h.double()) / math.log(2))
        gr = torch.randn(32, q.d, generator=torch.Generator().manual_seed(0)).to(D.device)
        vi_stats = {}
        for alpha in (1e-4, 1e-3, 0.01, 0.1, 1.0):  # the exponential tilt saturates near alpha ~ 0.01 on raw activations; the quadratic form is the alpha -> 0 limit
            lam = alpha * nT
            real = torch.tensor([float(vi_tokens(z0, D @ (h / h.norm() * hn), lam)) for h in H[:32]]); rand = torch.tensor([float(vi_tokens(z0, D @ (g / g.norm() * hn), lam)) for g in gr])
            qr = torch.tensor([quad(h / h.norm() * hn, lam) for h in H[:32]]); qg = torch.tensor([quad(g / g.norm() * hn, lam) for g in gr])
            vi_stats[str(alpha)] = {"real_bits": float(real.mean()), "random_bits": float(rand.mean()), "ratio": float(real.mean() / rand.mean()),
                                    "quadratic_real_bits": float(qr.mean()), "quadratic_random_bits": float(qg.mean()), "quadratic_ratio": float(qr.mean() / qg.mean())}
        # alignment of the top eigenvectors with the frequency-weighted mean lens vector (the unigram direction)
        uni = (Pm[:, None] * D).sum(0); uni = uni / uni.norm()
        top_align = [float((U[:, k].float() @ uni).abs()) for k in range(5)]
        # overlap of NNOMP atoms with the top-k eigenspace
        overlaps = []
        for h in H[:8]:
            coefs, _ = nnomp(D, h, a.k); atoms = D[list(coefs)] if coefs else None
            if atoms is not None:
                An = atoms / atoms.norm(dim=1, keepdim=True); Uk = U[:, : a.k].float()
                overlaps.append(float(((An @ Uk) ** 2).sum(1).mean()))
        r = {"participation_ratio": pr, "top_eigenvalues": ev[:10].tolist(), "cum_share_top": {str(k): float(share[:k].sum()) for k in (5, 10, 25, 50, 100, 250)},
             "vi": vi_stats, "top5_eigvec_cos_unigram": top_align,
             "atom_overlap_topk_eigenspace": sum(overlaps) / max(len(overlaps), 1), "seconds": time.time() - t0}
        results["layers"][str(l)] = r; json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
        log(f"L{l}: participation ratio {pr:.1f} of {q.d}; top-25 eigenvalues carry {share[:25].sum():.2f} of trace, top-250 {share[:250].sum():.2f}; "
            + "; ".join(f"VI(alpha={al}) {v['real_bits']:.3g} bits vs random {v['random_bits']:.3g} (x{v['ratio']:.2f}; quadratic {v['quadratic_real_bits']:.3g})" for al, v in vi_stats.items())
            + f"; |cos(top eigvecs, unigram dir)| {[round(x, 2) for x in top_align]}; NNOMP atoms in top-{a.k} eigenspace: {r['atom_overlap_topk_eigenspace']:.2f}; {r['seconds']:.0f}s")
        torch.save({"eigenvalues": ev.cpu(), "layer": l}, os.path.join(out_dir, f"spectrum_L{l}.pt"))


if __name__ == "__main__":
    main()
