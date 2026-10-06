"""J-lens fitting (the anthropics/jacobian-lens estimator) and artifact IO.

fit: cotangent one-hot on every valid target position at once, gradient at each valid source, mean over sources,
mean over prompts -> J[layer] = mean_t sum_{t'>=t} dh_L[t'] / dh_l[t]. Also fits per-gap Jacobians G_g and the
last-target-only J(empty) for Proposition 3 diagnostics. Token vectors D_l = W_U J_l, rows d_l(w).
"""
import math, torch
from ..model.qwen3_min import forward


def valid_mask(T, skip_first=16):
    m = torch.zeros(T, dtype=torch.bool); m[skip_first:T - 1] = True
    return m


def fit(q, prompts, layers, skip_first=16, dim_batch=8, target_layer=None):
    w = q.w; L = target_layer if target_layer is not None else w.n_layers - 1; dev = w.emb.device; ad = getattr(q, "act_dtype", None)
    d = q.d; J = {l: torch.zeros(d, d, dtype=torch.float64) for l in layers}
    for ids in prompts:
        ids = ids.view(1, -1); T = ids.shape[1]; mask = valid_mask(T, skip_first)
        for l in layers:
            src = torch.zeros(1, T, d, dtype=ad or w.emb.dtype, device=dev, requires_grad=True)
            hL, _ = forward(w, ids, inject=(l, None, src), stop_layer=L, act_dtype=ad)
            for j0 in range(0, d, dim_batch):
                js = list(range(j0, min(j0 + dim_batch, d)))
                for j in js:
                    g, = torch.autograd.grad(hL[0, mask, j].sum(), src, retain_graph=True)
                    J[l][j] += g[0, mask].double().mean(0).cpu()
    for l in layers: J[l] /= len(prompts)
    return J


def fit_gap_profile(q, prompts, layer, skip_first=16, target_layer=None):
    """G_g for each gap g (mean over pairs) and J(empty) (last position as target). Proposition 3 diagnostics."""
    w = q.w; L = target_layer if target_layer is not None else w.n_layers - 1; d = q.d; dev = w.emb.device; ad = getattr(q, "act_dtype", None)
    G, n = {}, {}
    Jempty = torch.zeros(d, d, dtype=torch.float64)
    for ids in prompts:
        ids = ids.view(1, -1); T = ids.shape[1]; mask = valid_mask(T, skip_first); srcs = mask.nonzero().flatten().tolist()
        src = torch.zeros(1, T, d, dtype=ad or w.emb.dtype, device=dev, requires_grad=True)
        hL, _ = forward(w, ids, inject=(layer, None, src), stop_layer=L, act_dtype=ad)
        for j in range(d):
            for tp in srcs + [T - 1]:
                g, = torch.autograd.grad(hL[0, tp, j], src, retain_graph=True)
                for t in srcs:
                    if t > tp: continue
                    if tp == T - 1:
                        Jempty[j] += g[0, t].double().cpu() / len(srcs)
                    if tp in srcs:
                        gap = tp - t
                        G.setdefault(gap, torch.zeros(d, d, dtype=torch.float64)); n.setdefault(gap, 0)
                        G[gap][j] += g[0, t].double().cpu()
                        if j == 0: n[gap] += 1
    for gap in G: G[gap] /= n[gap]
    return G, Jempty / len(prompts)


def token_matrix(w, J):
    """D_l = W_U J_l, rows are J-lens token vectors d(w) (bench convention: raw lm_head, no gamma)."""
    return (w.lm.double() @ J)


def save(path, J, meta):
    torch.save({"J": {l: J[l].half() for l in J}, **meta}, path)


def load(path):
    o = torch.load(path, map_location="cpu", weights_only=False)
    J = o["J"] if "J" in o else {i: o["jacobians"][i] for i in range(o["jacobians"].shape[0])}
    return {int(l): J[l].float() for l in J}, {k: v for k, v in o.items() if k not in ("J", "jacobians")}
