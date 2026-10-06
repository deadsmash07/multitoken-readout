"""J-beam: prompt-blind phrase decoder from one activation.

read(h, layer) -> ranked phrases with z_bg scores. Step 1 is the exact J-lens over D_l; later steps use the
forward SJ-lens on cached perturbed contexts with a Hutchinson normalizer from null caches; stopping uses the
family-wise threshold and the coherence test (Props. 4, 6); phrase scores come from the step and null sums.

Scales. A lens row d(w) = W_U J_l sums over the ~|T| target positions of the lens estimator (mean over sources),
while a step numerator W_U[w] . J(p) u from ContextBank.step_set is a single-target quantity (mean over sources,
one target); measured ||J(p1)^T W_U[p2]|| / ||d(p1)|| = 0.013 / 0.008 / 0.002 at L11 / L17 / L22 of Qwen3-1.7B.
Adding them raw (Cfg.combine = "raw") lets step 1 dominate every phrase score and feeds incommensurable
quantities to the coherence test. The default, combine = "std", divides each step's numerator and its null
readouts by the step's normaliser sigma_k(w_k): the phrase numerator is A = sum_k z_k, its null on direction g_j
is sum_k <v_k, g_j> / sigma_k, and the phrase score is z = A / sqrt(Q) with Q the empirical mean square of that
null, i.e. the background z of the reweighted phrase vector sum_k v_k / sigma_k. Under independent steps Q = n
and z is Stouffer's sum_k z_k / sqrt(n); combine = "stouffer" sets Q = n by fiat. Prop. 6 then runs on
(A, Q, a = z_new, q = Q_new - Q, r = 1), so extension means the phrase z rises. Step 1 (seeds by (D h)[w] /
||d(w)||, z_1 by the null of d(w)) is the same in every mode, so single-token output equals the bench J-lens; in
"std" / "stouffer" every score is invariant to a rescaling of D.

Normaliser (Cfg.normalizer, score.shrink_sigma). The plain Hutchinson sigma from m = 16 nulls has ~18 % relative
error per token, so tokens whose sigma draw is small win the argmax (G1's ` tutors磋`). "eb" shrinks log
sigma^2_mc(w) toward a smooth per-token prior (Cfg.prior: "pooled" = the same statistic pooled over
Cfg.pool_prefixes background prefixes at this layer, computed once; "lens" = the exact step-1 null scale from the
whole background; "wu" = ||W_U[w]||; "dnorm" = ||d(w)||) with a data-driven weight; "proxy" uses the rescaled
prior alone; "floor" adds a variance floor; "mc" is the old estimator; "raw" ranks by the numerator. The
calibrated step z is num / sigma-hat in every mode but "raw" (there num / sigma_mc), and the stopping threshold
is z_fw (Cfg.stop = "zfw"), a conformal tau (Cfg.stop = "tau", Cfg.tau) or none. Defaults are the configuration
chosen on the tuning half in docs/JBEAM_TUNING.md (eb / dnorm, m 16, hc, rho .05, lens seeds, tau stop); on
Qwen3-1.7B no configuration recovers multi-token answers (exact-string recall 0), see there.
"""
from dataclasses import dataclass, field
import math, torch
from ..lens.score import z_fw, hutchinson_sigma, extend_ok, nnomp, shrink_sigma, lens_null_scale


@dataclass
class Cfg:
    seeds: int = 10
    beams: int = 10
    expand: int = 1  # children per beam and step (1: greedy argmax)
    max_len: int = 5
    eps: float = None  # absolute FD step; None: relative, eps = rho * mean source norm / ||u|| (G2: rho <= 0.05)
    rho: float = 0.05
    m_null: int = 16  # nulls used (the first m_null of those given)
    q_fw: float = 0.05
    inject: str = "hc"
    k_pursuit: int = 25
    mode: str = "z"
    normalizer: str = "eb"  # "raw" | "mc" | "floor" | "proxy" | "eb"
    prior: str = "dnorm"  # "wu" | "dnorm" | "lens" | "pooled"
    floor: float = 1.0
    nu: float = None
    support: float = 0.0  # candidates need a prior scale above this vocabulary quantile (0: off)
    pool_prefixes: int = 64
    seed_rule: str = "lens"  # "bench": (D h)[w] / ||d(w)||; "lens": (D h)[w]; "z": (D hc)[w] / sigma_1(w), sigma_1 from the whole background
    stop: str = "tau"  # "zfw" | "tau" | "none"; "tau" needs Cfg.tau (conformal max-z on background, e18) and falls back to z_fw when None
    tau: float = None
    coherence: bool = True
    combine: str = "std"  # "std" | "stouffer" | "raw", see the module docstring
    boundary: tuple = ()
    top_out: int = 10


@dataclass
class Beam:
    toks: list
    num: float  # sum over steps of the (standardised, unless combine == "raw") numerators
    null: torch.Tensor  # [m] the same sum read on the null directions
    steps: list = field(default_factory=list)  # (token, calibrated step z)
    combine: str = "std"
    done: bool = False

    def Q(s): return float(len(s.toks)) if s.combine == "stouffer" else float(s.null.pow(2).mean())
    def z(s): return s.num / max(math.sqrt(s.Q()), 1e-12)
    def z_stouffer(s): return sum(z for _, z in s.steps) / math.sqrt(len(s.steps))


class JBeam:
    def __init__(s, q, bank, D, layer, nulls, cfg=Cfg(), mu=None, Hbg=None, null_set=None, prior=None):
        """nulls: [m, d] (or a list) centred background activations; Hbg: [N, d] raw background activations for the
        "lens" / "pooled" priors and the "z" seed rule (falls back to the nulls); null_set / prior: share another
        JBeam's null caches and prior (same bank, layer, nulls, mode)."""
        s.q, s.bank, s.D, s.l, s.cfg, s.mu = q, bank, D, layer, cfg, mu
        s.Dn = D.norm(dim=1).clamp_min(1e-12); s.V = D.shape[0]
        s.nulls = nulls if torch.is_tensor(nulls) else torch.stack(list(nulls)); s.nulls = s.nulls.to(D.device)
        assert cfg.m_null <= s.nulls.shape[0], f"m_null {cfg.m_null} > {s.nulls.shape[0]} nulls given"
        with torch.no_grad(): s.hn = float(bank.resid(layer)[:, bank.mask.to(bank.ids.device)].float().norm(dim=-1).mean())
        s.null_eps = torch.tensor([s._eps(g) for g in s.nulls], dtype=torch.float64, device=D.device)
        s.null_set = null_set if null_set is not None else bank.cache_set(layer, s.nulls, s.null_eps)
        s.zfw_vocab = z_fw(s.V, cfg.q_fw)
        Hc = (Hbg.to(D.device, D.dtype) - (mu.to(D.device, D.dtype) if mu is not None else 0)) if Hbg is not None else s.nulls
        s.sig1_all = lens_null_scale(D, Hc)
        need = cfg.normalizer in ("proxy", "eb") or cfg.support > 0
        s.prior = prior if prior is not None else (s._prior(Hc) if need else None)
        s.support_ok = (s.prior >= s.prior.quantile(cfg.support)) if cfg.support > 0 else None

    def _eps(s, u):
        return s.cfg.eps if s.cfg.eps is not None else s.cfg.rho * s.hn / max(float(u.norm()), 1e-12)

    def _prior(s, Hc):
        p = s.cfg.prior
        if p == "wu": return s.q.w.lm.to(s.D.dtype).norm(dim=1).clamp_min(1e-12)
        if p == "dnorm": return s.Dn
        if p == "lens": return s.sig1_all
        if p == "pooled":  # the MC statistic pooled over the lens top-1 prefixes of sampled background activations
            g = torch.Generator().manual_seed(0); rows = Hc[torch.randperm(Hc.shape[0], generator=g)[: s.cfg.pool_prefixes].to(Hc.device)]
            toks = sorted({int(t) for t in (s.D @ rows.T).argmax(0).tolist()})
            s2 = torch.zeros(s.V, dtype=torch.float64, device=s.D.device)
            for w in toks: s2 += s.null_readout([w]).double().pow(2).mean(0)
            s.pool_tokens = toks
            return (s2 / len(toks)).sqrt().to(s.D.dtype).clamp_min(1e-12)
        raise ValueError(p)

    def null_readout(s, prefix, memo=None):
        """[m, V] step readouts of the null directions for this prefix (memo: dict shared across configs)."""
        key = ("n", s.cfg.mode, tuple(prefix))
        if memo is not None and key in memo: return memo[key]
        r = s.bank.step_set(list(prefix), s.null_set, s.cfg.mode)
        if memo is not None: memo[key] = r
        return r

    def sigma(s, nn_):
        """normaliser [V] for one step from its null readouts (the first m_null of them)."""
        nn_ = nn_[: s.cfg.m_null]
        if s.cfg.normalizer == "raw": return torch.ones(s.V, dtype=nn_.dtype, device=nn_.device)
        return shrink_sigma(nn_, s.prior, s.cfg.normalizer, s.cfg.floor, s.cfg.nu)

    def step_scores(s, prefix, uset, memo=None):
        """(num [V], nn_ [m, V], sigma-hat [V], sigma_mc [V], z [V]) of the step after `prefix` for the direction of uset."""
        cfg = s.cfg; key = ("u", cfg.inject, cfg.rho, cfg.eps, cfg.k_pursuit, cfg.mode, tuple(prefix))
        if memo is not None and key in memo: num = memo[key]
        else:
            num = s.bank.step_set(list(prefix), uset, cfg.mode)[0]
            if memo is not None: memo[key] = num
        nn_ = s.null_readout(prefix, memo)[: cfg.m_null]; sig = s.sigma(nn_); sig_mc = hutchinson_sigma(nn_)
        z = num / sig
        if s.support_ok is not None: z = z.masked_fill(~s.support_ok, -float("inf"))
        return num, nn_, sig, sig_mc, z

    def read(s, h, log=None, memo=None):
        cfg = s.cfg; h = h.to(device=s.D.device, dtype=s.D.dtype); rawc = cfg.combine == "raw"; rawn = cfg.normalizer == "raw"
        # z-scores use the background-centred activation (Prop. 4); seeds keep the bench ranking by D h. mu = None: uncalibrated.
        hc = h - (s.mu if s.mu is not None else 0)
        coefs, hJ = nnomp(s.D, hc, cfg.k_pursuit)
        u = {"h": hc, "hJ": hJ, "hc": hc}[cfg.inject]
        Dh = s.D @ h; Dhc = s.D @ hc; Dg = (s.D @ s.nulls.T).T[: cfg.m_null]  # [V], [V], [m, V]
        s1 = {"bench": Dh / s.Dn, "lens": Dh, "z": Dhc / s.sig1_all}[cfg.seed_rule]
        sig1_mc = hutchinson_sigma(Dg); sig1 = s.sigma(Dg)
        top = torch.topk(s1, cfg.seeds).indices.tolist()
        eps_u = s._eps(u); uset = s.bank.cache_set(s.l, u.view(1, -1), [eps_u])
        thresh = {"zfw": s.zfw_vocab, "tau": cfg.tau if cfg.tau is not None else s.zfw_vocab, "none": -float("inf")}[cfg.stop]
        beams = []
        for w in top:
            sc = 1.0 if rawc else float(sig1[w])
            beams.append(Beam([w], float(Dhc[w]) / sc, Dg[:, w] / sc, [(w, float(Dhc[w] / (sig1_mc[w] if (rawc or rawn) else sig1[w])))], cfg.combine))
        finished = []
        for step in range(1, cfg.max_len):
            cand = []
            for b in beams:
                if b.done: finished.append(b); continue
                num, nn_, sig, sig_mc, zs = s.step_scores(b.toks, uset, memo)
                ws = torch.topk(zs, cfg.expand).indices.tolist()
                A, Q = b.num, b.Q()
                for i, w in enumerate(ws):
                    sc = 1.0 if rawc else float(sig[w])
                    a = float(num[w]) / sc; newnull = b.null + nn_[:, w] / sc; r = float(sig_mc[w]) / sc if (rawc or rawn) else 1.0  # step z = a / r
                    qinc = (1.0 if cfg.combine == "stouffer" else float(newnull.pow(2).mean())) - Q
                    ok, zstep = extend_ok(A, Q, a, qinc, thresh, r); coh = a > A * (math.sqrt(1 + max(qinc, 0.0) / max(Q, 1e-12)) - 1)
                    ok = (zstep > thresh) and (coh or not cfg.coherence)
                    if log is not None:
                        log.append(dict(prefix=list(b.toks), w=w, A=A, Q=Q, a=a, qinc=qinc, r=r, zstep=zstep, coh=coh, ok=ok,
                                        top=[(int(j), float(zs[j])) for j in torch.topk(zs, 5).indices]))
                    if (not ok) or (w in cfg.boundary):
                        if i == 0: b.done = True; finished.append(b)
                        continue
                    cand.append(Beam(b.toks + [w], A + a, newnull, b.steps + [(w, zstep)], cfg.combine))
            if not cand: break
            cand.sort(key=lambda b: -b.z()); beams = cand[: cfg.beams]
        finished += [b for b in beams if not b.done]
        seen, out = set(), []
        for b in sorted(finished, key=lambda b: -b.z()):
            t = tuple(b.toks)
            if t in seen: continue
            seen.add(t); out.append((b.toks, b.z(), b.z_stouffer(), b.steps))
        # P3 / audit A16: under the conformal stop the certificate also gates what is emitted (phrase z > tau);
        # single-token phrases are gated too, so a read may return nothing.
        thr = cfg.tau if (cfg.stop == "tau" and cfg.tau is not None) else None
        out = [o for o in out if thr is None or o[1] > thr]
        return out[: cfg.top_out], coefs
