"""Forward SJ-lens: J_l(p) u for all vocabulary tokens at once, via cached central differences or exact JVP.

ContextBank holds C equal-length generic contexts, their source mask and the clean K/V cache. A perturbed cache
is one prefill with +eps*u (and one with -eps*u) at the sources from layer l on; every prefix extension then costs
|p| tokens x C contexts x (number of caches). Null caches (directions g_j) are built once per layer and reused.

Batched path (CacheSet). Injection after block l leaves blocks 0..l untouched, so their K/V are the clean cache's
and a prefix's residual after block l is the same for every perturbed cache: cache_set() prefills only blocks
l+1.. from the stored clean residual, for n directions x 2 signs batched along the batch axis, and extend_set()
runs one shared pass through blocks 0..l and one batched pass through the rest. Memory per cache drops from 28 to
(28 - l - 1) layers and the m-null step readout is one forward call.

Modes. "z": pseudo-logit numerators W_U[w] . J(p) u (bench convention, linear in the residual, so the difference is
taken on the residuals before W_U). "lp": the derivative of log p(w | c, p) through the final norm, i.e. the object
backward.sj_vectors(mode="lp") differentiates (Prop. 2 with the norm Jacobian inserted); before 25 Sep the forward
"lp" was a softmax-centred pseudo-logit change, a different quantity (audit A5).
"""
import torch, torch.nn.functional as F
from ..model.qwen3_min import forward, pseudo_logits, logits
from .jlens import valid_mask


def _lsm(w, h):
    z = logits(w, h); z = z.float() if z.dtype in (torch.bfloat16, torch.float16) else z
    return F.log_softmax(z, -1)


class CacheSet:
    """2n partial K/V caches (blocks > layer) for directions U [n, d] at steps eps [n]: entry j*C + c of sign s."""
    def __init__(s, layer, cache, U, eps, C):
        s.layer, s.cache, s.U, s.eps, s.n, s.C = layer, cache, U, eps, U.shape[0], C


class ContextBank:
    def __init__(s, q, contexts, skip_first=16):
        s.q = q; s.ids = torch.stack([c.view(-1) for c in contexts]); s.C, s.tau = s.ids.shape
        s.mask = valid_mask(s.tau, skip_first); s.nT = int(s.mask.sum()); s._x = {}
        with torch.no_grad():
            s.hL_clean, s.cache_clean = q.prefill(s.ids)

    def perturbed(s, layer, u, eps):
        m = s.mask.view(1, -1).expand(s.C, -1)
        with torch.no_grad():
            _, cp = s.q.prefill(s.ids, inject=(layer, m, eps * u))
            _, cm = s.q.prefill(s.ids, inject=(layer, m, -eps * u))
        return cp, cm

    def extend_last(s, prefix, cache):
        if len(prefix) == 0:
            return s._last_from_cache(cache)
        ids = torch.tensor(prefix, dtype=torch.long, device=s.ids.device).view(1, -1).expand(s.C, -1)
        with torch.no_grad():
            return s.q.extend(ids, cache)[:, -1]

    def _last_from_cache(s, cache):
        raise ValueError("empty prefix: use the exact D_l readout at step 1")

    def direction_estimate(s, prefix, cp, cm, eps):
        """u_hat = J_l(p) u  (mean over contexts, divided by |T|), from the two perturbed caches."""
        hp, hm = s.extend_last(prefix, cp), s.extend_last(prefix, cm)
        return ((hp - hm) / (2 * eps * s.nT)).mean(0), (hp, hm)

    def step_numerators(s, prefix, cp, cm, eps, mode="z", w_clean=None):
        """num(w) = W_U[w] . J(p) u for all w ('z'), or d log p(w | c, p) / d(eps) through the final norm ('lp')."""
        hp, hm = s.extend_last(prefix, cp), s.extend_last(prefix, cm)
        if mode == "lp":
            dz = _lsm(s.q.w, hp) - _lsm(s.q.w, hm)
        else:
            dz = pseudo_logits(s.q.w, hp) - pseudo_logits(s.q.w, hm)
        return (dz / (2 * eps * s.nT)).mean(0)

    # batched path
    def resid(s, layer):
        """clean residual after block `layer` for every context, [C, T, d] (computed once)."""
        if layer not in s._x:
            with torch.no_grad(): s._x[layer] = s.q.resid(s.ids, layer)
        return s._x[layer]

    def cache_set(s, layer, U, eps, chunk=8):
        """CacheSet for directions U [n, d] with FD steps eps [n] (2n partial prefills, `chunk` directions per call)."""
        q, w = s.q, s.q.w; x = s.resid(layer); n = U.shape[0]; eps = torch.as_tensor(eps, dtype=torch.float64, device=U.device).view(-1)
        m = s.mask.to(s.ids.device).view(1, -1, 1); cache = [None] * w.n_layers
        with torch.no_grad():
            for j0 in range(0, n, chunk):
                js = list(range(j0, min(j0 + chunk, n))); nb = len(js)
                d = (eps[js].view(nb, 1, 1) * U[js].view(nb, 1, -1)).to(x.dtype)  # [nb, 1, d]
                d = torch.cat([d, -d], 0).expand(2 * nb, s.tau, -1) * m  # sign-major: [+ js..., - js...]
                delta = d.repeat_interleave(s.C, 0)  # [2 nb C, T, d]: entry (sign, j, c)
                ids = s.ids.repeat(2 * nb, 1)
                _, new = forward(w, ids, inject=(layer, None, delta), resume=(layer, x.repeat(2 * nb, 1, 1)), act_dtype=q.act_dtype)
                for i in range(layer + 1, w.n_layers):
                    k, v = new[i]
                    if cache[i] is None:
                        cache[i] = tuple(torch.empty((2 * n * s.C, *t.shape[1:]), dtype=t.dtype, device=t.device) for t in (k, v))
                    for si in range(2):
                        blk = slice(si * n * s.C + j0 * s.C, si * n * s.C + (j0 + nb) * s.C)
                        cache[i][0][blk] = k[si * nb * s.C:(si + 1) * nb * s.C]; cache[i][1][blk] = v[si * nb * s.C:(si + 1) * nb * s.C]
                del new
        return CacheSet(layer, cache, U, eps, s.C)

    def extend_set(s, prefix, cs):
        """last-position final residual of the prefix against every cache of cs: [2, n, C, d] (sign, direction, context)."""
        q, w = s.q, s.q.w; l = cs.layer
        ids = torch.tensor(prefix, dtype=torch.long, device=s.ids.device).view(1, -1).expand(s.C, -1)
        with torch.no_grad():
            xl, _ = forward(w, ids, None, s.cache_clean, stop_layer=l, act_dtype=q.act_dtype, return_cache=False)  # shared blocks 0..l
            h, _ = forward(w, ids.repeat(2 * cs.n, 1), None, cs.cache, act_dtype=q.act_dtype, resume=(l, xl.repeat(2 * cs.n, 1, 1)), return_cache=False)
        return h[:, -1].view(2, cs.n, s.C, -1)

    def step_set(s, prefix, cs, mode="z", per_context=False):
        """numerators for every direction of cs: [n, V] (mean over contexts) or [n, C, V] with per_context."""
        h = s.extend_set(prefix, cs); den = (2 * cs.eps * s.nT).view(-1, 1, 1).to(h.dtype)
        if mode == "lp":
            out = torch.stack([_lsm(s.q.w, h[0, j]) - _lsm(s.q.w, h[1, j]) for j in range(cs.n)]) / den  # [n, C, V]
            return out if per_context else out.mean(1)
        dh = (h[0] - h[1]) / den  # [n, C, d]
        if per_context: return pseudo_logits(s.q.w, dh)
        return pseudo_logits(s.q.w, dh.mean(1))


def jvp_exact(q, ids, layer, mask, u, prefix=(), per_context=False):
    """exact directional derivative of the last-position final residual, all sources perturbed at once."""
    if len(prefix):
        ids = torch.cat([ids, torch.tensor(prefix, dtype=torch.long, device=ids.device).view(1, -1).expand(ids.shape[0], -1)], 1)
    m = torch.zeros(ids.shape[1], dtype=torch.bool, device=ids.device); m[: mask.numel()] = mask.to(ids.device)
    m = m.view(1, -1).expand(ids.shape[0], -1)
    f = lambda e: forward(q.w, ids, inject=(layer, m, e * u), act_dtype=getattr(q, 'act_dtype', None))[0][:, -1]
    z = torch.zeros((), dtype=u.dtype, device=u.device)
    hL, t = torch.func.jvp(f, (z,), (torch.ones((), dtype=u.dtype, device=u.device),))
    if per_context: return hL, t / int(mask.sum())
    return t.mean(0) / int(mask.sum())


def jvp_step(q, ids, layer, mask, u, prefix=(), mode="z"):
    """exact step numerators [V]: W_U J(p) u ('z') or mean_c d/d(eps) log p(w | c, p) through the final norm ('lp')."""
    hL, t = jvp_exact(q, ids, layer, mask, u, prefix, per_context=True)
    if mode == "lp":
        _, tz = torch.func.jvp(lambda h: _lsm(q.w, h), (hL,), (t,))
        return tz.mean(0)
    return pseudo_logits(q.w, t.mean(0))


def jvp_lens(q, ids, layer, mask, u):
    """exact directional derivative of the released lens estimator: sum over the masked target positions of the
    final residual, mean over sources and contexts, i.e. J_lens u (identity 4 of tests/toy_identities.py)."""
    m = mask.to(ids.device).view(1, -1).expand(ids.shape[0], -1)
    f = lambda e: forward(q.w, ids, inject=(layer, m, e * u), act_dtype=getattr(q, 'act_dtype', None))[0][:, mask.to(ids.device)].sum(1)
    z = torch.zeros((), dtype=u.dtype, device=u.device)
    _, t = torch.func.jvp(f, (z,), (torch.ones((), dtype=u.dtype, device=u.device),))
    return t.mean(0) / int(mask.sum())


def gap0_jvp(q, ids, layer, mask, u, pos_batch=16):
    """mean over sources t of J_{tt} u (the gap-0 term of the lens estimator): perturb one source position per
    batch copy and read the final residual at that same position. ids [B, T]; batched over positions."""
    B, T = ids.shape; srcs = mask.nonzero().flatten().tolist(); acc = torch.zeros(q.d, dtype=u.dtype, device=u.device); n = 0
    for b in range(B):
        for s0 in range(0, len(srcs), pos_batch):
            ps = torch.tensor(srcs[s0:s0 + pos_batch], device=ids.device); rep = ids[b:b + 1].expand(len(ps), -1)
            m = torch.zeros(len(ps), T, dtype=torch.bool, device=ids.device); m[torch.arange(len(ps), device=ids.device), ps] = True
            f = lambda e: forward(q.w, rep, inject=(layer, m, e * u), act_dtype=getattr(q, 'act_dtype', None))[0][torch.arange(len(ps), device=ids.device), ps]
            z = torch.zeros((), dtype=u.dtype, device=u.device)
            _, t = torch.func.jvp(f, (z,), (torch.ones((), dtype=u.dtype, device=u.device),))
            acc += t.sum(0); n += len(ps)
    return acc / n
