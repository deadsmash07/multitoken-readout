"""Scores, nulls, normalizers, stopping, conformal guarantee, verbalizable information.

z_bg(v, h)   = <v, h - mu> / sqrt(Var_bg <v, g>)          background z-score (Prop. 4), N(0,1) on background h
hutchinson   : sigma^2(w) for all w from m null directions   (Prop. 5), rel. err ~ 1/sqrt(2m)
z_fw(V, q)   = Phi^{-1}(1 - q/V)                            Bonferroni family-wise threshold (Prop. 4)
extend_ok    : coherence stopping test                       (Prop. 6)
conformal    : split-conformal threshold on max-z, finite-sample FPR <= q, distribution-free
vi_tokens    : verbalizable information KL(q_h || p_0) of the first-order tilted distribution
"""
import math, torch
from torch.distributions import Normal


def z_fw(V, q=0.05):
    return float(Normal(torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64)).icdf(torch.tensor(1 - q / V, dtype=torch.float64)))


def background_stats(H):
    mu = H.mean(0); Hc = H - mu
    return mu, Hc


def z_bg(v, h, mu, Hc):
    """v [..., d], h [d], Hc [N, d] centered background. Uses the empirical null of <v, g>."""
    num = v @ (h - mu)
    den = (Hc @ v.T).std(0) if v.dim() == 2 else (Hc @ v).std()
    return num / den.clamp_min(1e-12)


def hutchinson_sigma(null_nums):
    """null_nums [m, V]: <v(w), g_j> for m null directions -> sigma(w) [V]."""
    return null_nums.pow(2).mean(0).sqrt().clamp_min(1e-12)


def lens_null_scale(D, Hc, chunk=2048):
    """sigma_1(w) [V] = rms over ALL centred background activations Hc [N, d] of (D g)[w]: the exact step-1 null
    scale (free: one matmul), a smooth per-token prior for the later steps' Hutchinson estimates."""
    s2 = torch.zeros(D.shape[0], dtype=torch.float64, device=D.device)
    for i in range(0, Hc.shape[0], chunk):
        s2 += (D @ Hc[i:i + chunk].to(D.dtype).T).double().pow(2).sum(1)
    return (s2 / Hc.shape[0]).sqrt().to(D.dtype).clamp_min(1e-12)


def shrink_sigma(nn_, prior=None, kind="mc", floor=1.0, nu=None):
    """sigma-hat [V] of a step readout from its m null readouts nn_ [m, V] and a smooth prior scale [V].
    mc:    rms_j nn_[j] (Prop. 5; relative error ~ 1/sqrt(2m), so rare tokens with a small draw win an argmax).
    floor: sqrt(s2_mc + (floor * median_w sigma_mc)^2).
    proxy: the prior rescaled to the step (median over w of s2_mc / prior^2), no MC term at all.
    eb:    empirical Bayes in log space. log s2_mc = log s2 + log(chi2_m / m) has bias psi(m/2) - log(m/2) and
           variance psi'(m/2) under Gaussian nulls; the prior log(scale * prior^2) has dispersion tau^2 =
           robust Var_w(log s2_mc - log prior^2) - psi'(m/2) (MAD-based, so unreachable tokens do not blow it
           up). The posterior is the precision-weighted mean; nu (prior pseudo-count in units of null draws)
           overrides the data-driven weight."""
    m = nn_.shape[0]; s2 = nn_.pow(2).mean(0).clamp_min(1e-24); sig = s2.sqrt()
    if kind == "mc": return sig.clamp_min(1e-12)
    if kind == "floor": return (s2 + (floor * sig.median()) ** 2).sqrt()
    h = torch.tensor(m / 2.0, dtype=torch.float64)
    bias = float(torch.special.digamma(h) - h.log()); var = float(torch.special.polygamma(1, h))
    lp2 = 2 * prior.to(s2.dtype).clamp_min(1e-12).log(); lv = s2.log() - bias; diff = lv - lp2; scale = diff.median()
    if kind == "proxy": return ((lp2 + scale) / 2).exp()
    if kind == "eb":
        dev = diff - scale; tau2 = max((1.4826 * float(dev.abs().median())) ** 2 - var, 1e-4)
        a = 1.0 / var; b = a * nu / m if nu is not None else 1.0 / tau2
        return ((a * lv + b * (lp2 + scale)) / (a + b) / 2).exp()
    raise ValueError(kind)


def extend_ok(A, Q, a, q, z_fw_val, r):
    """Prop 6: extend iff step z = a/r > z_fw and a > A (sqrt(1 + q/Q) - 1)."""
    step_z = a / max(r, 1e-12)
    coh = a > A * (math.sqrt(1 + max(q, 0.0) / max(Q, 1e-12)) - 1)
    return (step_z > z_fw_val) and coh, step_z


def conformal_threshold(scores_bg, q=0.05):
    """scores_bg [N]: the max-z statistic on N held-out background activations.
    Returns tau with P(max-z(new background) > tau) <= q, exactly, under exchangeability."""
    N = scores_bg.numel(); k = math.ceil((N + 1) * (1 - q))
    s = scores_bg.sort().values
    return float(s[min(k, N) - 1]) if k <= N else float("inf")


def vi_tokens(z0, Dh, lam):
    """z0 [C, V] clean pseudo-logits per context, Dh [V] = D_l h (first-order shift), lam = alpha*|T|.
    returns E_c KL(softmax(z0 + lam*Dh) || softmax(z0)) in bits."""
    lp0 = z0.log_softmax(-1); lp1 = (z0 + lam * Dh).log_softmax(-1)
    return ((lp1.exp() * (lp1 - lp0)).sum(-1) / math.log(2)).mean()


def nnomp(D, h, k=25, tol=0.0):
    """nonnegative OMP of h onto rows of D [V, d] (unit-normalized inside). returns coefs {idx: a}, h_J."""
    Dn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)
    r = h.clone(); act = []; a = None
    for _ in range(k):
        c = Dn @ r
        c[act] = -float("inf")
        j = int(c.argmax())
        if c[j] <= tol: break
        act.append(j)
        A = Dn[act].T
        a = nnls(A, h)
        r = h - A @ a
    if not act: return {}, torch.zeros_like(h)
    return {j: float(a[i]) for i, j in enumerate(act)}, A @ a


def nnls(A, b, iters=200):
    """projected gradient NNLS, small active sets only."""
    x = torch.zeros(A.shape[1], dtype=A.dtype, device=A.device); L = (A.T @ A).norm() + 1e-9
    for _ in range(iters):
        x = (x - (A.T @ (A @ x - b)) / L).clamp_min(0)
    return x
