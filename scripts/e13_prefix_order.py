"""E13 / E14 (H3a, H3b): prefix sensitivity kappa(p) and the order index Omega(a, b), with split-half noise floors.

kappa(p) = ||J(p) - J(empty)||_F / ||J(empty)||_F, both Frobenius norms by Hutchinson: E_g ||A g||^2 = ||A||_F^2 for
g ~ N(0, I), with m probe directions; each probe costs one JVP with prefix p and one with the empty prefix.
Omega(a, b) = 1 - cos(v(ab), v(ba)) with v the pseudo-logit SJ vectors (one VJP per phrase and context).
--random-control runs the same on a randomly initialised copy of the architecture (the zero reference).
--model tiny uses tests/tiny.py (for a self-check).

Noise floor (split halves A, B of the contexts, N/2 each, probes shared). Per context c let X_c = J_c(p) g,
Y_c = J_c(empty) g (iid over c); X, Y are their means over all N contexts, X_A, X_B, Y_A, Y_B over the halves.
kappa_full^2 = E_g ||X - Y||^2 / E_g ||Y||^2 is what the pre-registered kappa estimates, and with S = Cov(X_c - Y_c)
    E ||X - Y||^2 = ||E X_c - E Y_c||^2 + tr S / N,       E ||(X_A - Y_A) - (X_B - Y_B)||^2 = 4 tr S / N,
so the sampling noise of the full-sample difference is a quarter of the half-difference mean square and
    kappa_debiased^2 = [E_g ||X - Y||^2 - E_g ||(X_A - Y_A) - (X_B - Y_B)||^2 / 4] / E_g ||Y||^2   (clamped at 0).
The reported floors kappa_noise_{p, 0, diff} = sqrt(E_g ||Z_A - Z_B||^2 / 2) / sqrt(E_g ||Y||^2) for Z = X, Y, X - Y
are the noise sd of a HALF-sample estimate (2 tr S_Z / N)^{1/2} relative to ||J(empty)||; the full-sample mean's
noise variance is half of that (tr S_Z / N), i.e. the subtracted term above equals kappa_noise_diff^2 / 2. The
denominator E_g ||Y||^2 = ||J(empty)||_F^2 + tr S_0 / N is inflated the same way; kappa_debiased_den also subtracts
E_g ||Y_A - Y_B||^2 / 4 from it. kappa_A, kappa_B are kappa on each half alone (each with twice the full-sample
noise variance). For Omega: omega_A, omega_B are the index on each half; omega_noise_ab = 1 - cos(v_A(ab), v_B(ab))
and omega_noise_ba likewise compare the SAME phrase across halves, i.e. the value Omega would take from
sampling noise alone (half-sample noise; the full-sample floor is roughly half of it in the small-angle regime)."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min
from sjlens.lens.jlens import valid_mask
from sjlens.lens.forward import jvp_exact
from sjlens.lens.backward import sj_vectors
from sjlens.eval.common import load_hf, DTYPES


def usable_ids(tok, V):
    """C8 / A14: ids that decode to a non-empty, non-special string (drops the ~290 unused / control ids of Qwen3)."""
    special = set(getattr(tok, "all_special_ids", []) or [])
    return [i for i in range(V) if i not in special and tok.decode([i]).strip() and not tok.decode([i]).startswith("<|")]


def sample_tokens(tok, V, n, g):
    if tok is None: return torch.randint(0, V, (n,), generator=g).tolist()
    ok = torch.tensor(usable_ids(tok, V)); return ok[torch.randint(0, len(ok), (n,), generator=g)].tolist()


def random_model(model_name, seed, dtype, device):
    """(tok, model) with random weights from the config only: never touches the pretrained checkpoint, so a 14B random
    control fits beside nothing else on one GPU (the trained model is not loaded under --random-control)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name); cfg = AutoConfig.from_pretrained(model_name); cfg._attn_implementation = "eager"
    torch.manual_seed(seed); m = AutoModelForCausalLM.from_config(cfg, dtype=dtype).to(device).eval()
    for q_ in m.parameters(): q_.requires_grad_(False)
    return tok, m


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B"); p.add_argument("--random-control", action="store_true")
    p.add_argument("--dtype", default="fp32", help="fp32 | bf16 | bf16-mixed (bf16 weights, fp32 activations; = fp32 for Qwen3, half the memory)")
    p.add_argument("--layers", default="11"); p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=32); p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--n-prefix", type=int, default=100); p.add_argument("--n-pairs", type=int, default=100); p.add_argument("--m", type=int, default=16)
    p.add_argument("--device", default="cpu"); p.add_argument("--contexts", default="contexts/qwen3_T128.pt"); p.add_argument("--tag", default="e13_prefix_order"); p.add_argument("--seed", type=int, default=0); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.tag); os.makedirs(out_dir, exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    g = torch.Generator().manual_seed(a.seed)
    if a.model == "tiny":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")); from tiny import tiny
        m = tiny(); tok = None; V = 256; a.T = min(a.T, 24); skip = 4; ad = None
        ctx = [torch.randint(0, V, (a.T,), generator=g) for _ in range(a.n_ctx)]
    else:
        skip = 16; wd, ad = DTYPES[a.dtype]
        if a.random_control:  # G9 OOM fix: tokenizer + config only, the random model is built from the config (no pretrained weights)
            tok, m = random_model(a.model, a.seed, wd, a.device); V = m.config.vocab_size
        else:
            tok, m = load_hf(a.model, wd, a.device); V = m.config.vocab_size
        from sjlens.eval.phase_a import contexts
        ctx, _ = contexts(tok, a.n_ctx, 1, a.T, a.device, path=a.contexts if os.path.exists(a.contexts) else None)
    q = Qwen3Min(m, act_dtype=ad if a.model != "tiny" else None); ids = torch.stack([c.view(-1) for c in ctx]); mask = valid_mask(a.T, skip).to(ids.device)
    N = len(ctx); nA = N // 2; halves = [(0, nA), (nA, N)]; wA, wB = nA / N, (N - nA) / N
    tokens = sample_tokens(tok, V, max(a.n_prefix, 2 * a.n_pairs), g)
    results = {"model": a.model, "dtype": a.dtype, "random_control": a.random_control, "T": a.T, "n_ctx": N, "n_half": [nA, N - nA], "m": a.m, "layers": {}}
    msq = lambda X: float((X ** 2).sum(1).mean())  # E_g ||X g||^2 over the m probes
    for l in [int(x) for x in a.layers.split(",")]:
        t0 = time.time(); probes = torch.randn(a.m, q.d, generator=g).to(ids.device)
        def J_half(prefix, u, lo, hi):  # mean over the contexts lo:hi (exact when chunks divide the half)
            return torch.stack([jvp_exact(q, ids[c0:min(c0 + a.chunk, hi)], l, mask, u, prefix=prefix) * (min(c0 + a.chunk, hi) - c0) for c0 in range(lo, hi, a.chunk)]).sum(0) / (hi - lo)
        def J_apply(prefix):  # [2, m, d]: J(p) g_j on half A and half B
            return torch.stack([torch.stack([J_half(prefix, u, lo, hi) for u in probes]) for lo, hi in halves])
        full = lambda H: wA * H[0] + wB * H[1]
        Y = J_apply([]); Yf = full(Y); den = msq(Yf); den_noise = msq(Y[0] - Y[1]) / 4  # tr S_0 / N estimate
        noise0 = (msq(Y[0] - Y[1]) / 2 / den) ** 0.5; den_deb = max(den - den_noise, 1e-30)
        log(f"L{l}: ||J(empty)||_F^2 estimate {den:.4g}, half-sample noise share {den_noise / den:.3f}; kappa_noise_0 {noise0:.3f}")
        kappas = []
        for i, w in enumerate(tokens[: a.n_prefix]):
            X = J_apply([w]); Xf = full(X); Dd = X - Y  # per-half differences (X_A - Y_A, X_B - Y_B)
            num = msq(Xf - Yf); nd = msq(Dd[0] - Dd[1]) / 4
            row = {"token": w, "kappa": (num / den) ** 0.5, "kappa_A": (msq(Dd[0]) / msq(Y[0])) ** 0.5, "kappa_B": (msq(Dd[1]) / msq(Y[1])) ** 0.5,
                   "kappa_noise_p": (msq(X[0] - X[1]) / 2 / den) ** 0.5, "kappa_noise_diff": (msq(Dd[0] - Dd[1]) / 2 / den) ** 0.5,
                   "kappa_debiased": (max(num - nd, 0.0) / den) ** 0.5, "kappa_debiased_den": (max(num - nd, 0.0) / den_deb) ** 0.5}
            kappas.append(row)
            if i % 10 == 0: log(f"L{l} kappa {i}/{a.n_prefix}: {tok.decode([w]) if tok else w!r} {row['kappa']:.3f} (noise_p {row['kappa_noise_p']:.3f}, noise_diff {row['kappa_noise_diff']:.3f}, debiased {row['kappa_debiased']:.3f} / {row['kappa_debiased_den']:.3f})")
        omegas = []
        for i in range(a.n_pairs):
            x, y = tokens[2 * i], tokens[2 * i + 1]
            ab = [sj_vectors(q, [x, y], ctx[lo:hi], [l], skip)[l] for lo, hi in halves]; ba = [sj_vectors(q, [y, x], ctx[lo:hi], [l], skip)[l] for lo, hi in halves]
            om = lambda u, v: 1 - float(torch.nn.functional.cosine_similarity(u, v, 0))
            row = {"a": x, "b": y, "omega": om(full(ab), full(ba)), "omega_A": om(ab[0], ba[0]), "omega_B": om(ab[1], ba[1]), "omega_noise_ab": om(ab[0], ab[1]), "omega_noise_ba": om(ba[0], ba[1])}
            omegas.append(row)
            if i % 10 == 0: log(f"L{l} Omega {i}/{a.n_pairs}: {row['omega']:.4f} (halves {row['omega_A']:.4f} / {row['omega_B']:.4f}; same-phrase floor {row['omega_noise_ab']:.4f} / {row['omega_noise_ba']:.4f})")
        qs = lambda rows, k: {f"{k}_median": float(torch.tensor([r[k] for r in rows]).median()), f"{k}_p10": float(torch.tensor([r[k] for r in rows]).quantile(0.1)), f"{k}_p90": float(torch.tensor([r[k] for r in rows]).quantile(0.9))}
        r = {"kappa_noise_0": noise0, "den_noise_share": den_noise / den}
        for k in ("kappa", "kappa_A", "kappa_B", "kappa_noise_p", "kappa_noise_diff", "kappa_debiased", "kappa_debiased_den"): r |= qs(kappas, k)
        for k in ("omega", "omega_A", "omega_B", "omega_noise_ab", "omega_noise_ba"): r |= qs(omegas, k)
        r |= {"kappa": [(x["token"], x["kappa"]) for x in kappas], "kappa_rows": kappas, "omega": [(x["a"], x["b"], x["omega"]) for x in omegas], "omega_rows": omegas, "seconds": time.time() - t0}
        results["layers"][str(l)] = r; json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
        log(f"L{l}: kappa median {r['kappa_median']:.3f} [p10 {r['kappa_p10']:.3f}, p90 {r['kappa_p90']:.3f}] (pass >= 0.2, kill < 0.05); Omega median {r['omega_median']:.4f} [p10 {r['omega_p10']:.4f}, p90 {r['omega_p90']:.4f}] (pass >= 0.1, kill < 0.02); {r['seconds']:.0f}s")
        log(f"L{l}: noise floors (half-sample) kappa_noise_0 {noise0:.3f}, kappa_noise_p median {r['kappa_noise_p_median']:.3f}, kappa_noise_diff median {r['kappa_noise_diff_median']:.3f}; "
            f"kappa debiased median {r['kappa_debiased_median']:.3f} (den also debiased {r['kappa_debiased_den_median']:.3f}); halves {r['kappa_A_median']:.3f} / {r['kappa_B_median']:.3f}; "
            f"Omega halves {r['omega_A_median']:.4f} / {r['omega_B_median']:.4f}, same-phrase floor {r['omega_noise_ab_median']:.4f} / {r['omega_noise_ba_median']:.4f}")


if __name__ == "__main__":
    main()
