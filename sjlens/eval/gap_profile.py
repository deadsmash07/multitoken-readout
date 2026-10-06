"""E7 (gap profile, sampled output dims) and E8 (H1a: does the one-token SJ vector reduce to the J-lens vector?).

gap_profile_sampled: rows `dims` of every gap Jacobian G_g and of J(empty), by batched backward passes. On the
random model gap 0 dominates (rho = 1.6, cos(d, v) = 0.62); the proposal predicts a weak gap 0 on trained models.
reduction: cos(d(w), v_z(w)) over a token list, where v_z(w) = J(empty)^T W_U[w] comes from one VJP per
(token, context) and d(w) = J^T W_U[w] from the fitted or released lens, plus the Spearman of the two readouts
over the top-100 tokens for each supplied activation.
"""
import torch
from ..model.qwen3_min import forward
from ..lens.jlens import valid_mask
from ..lens.backward import sj_vectors
from .common import spearman


def gap_profile_sampled(q, prompts, layer, dims, skip_first=16, target_layer=None, dim_batch=None):
    """returns (G {gap: [nd, d]}, Jempty [nd, d]) for the sampled output dims, float64 on CPU.
    dim_batch bounds how many output dims share one vmapped backward pass (memory scales with it)."""
    w = q.w; L = target_layer if target_layer is not None else w.n_layers - 1; d = q.d; dev = w.emb.device; ad = getattr(q, "act_dtype", None)
    dims = torch.as_tensor(dims, dtype=torch.long, device=dev); nd = len(dims); db = dim_batch or nd
    G, n = {}, {}; Jempty = torch.zeros(nd, d, dtype=torch.float64, device=dev)
    for ids in prompts:
        ids = ids.view(1, -1); T = ids.shape[1]
        mask = valid_mask(T, skip_first).to(dev); srcs = mask.nonzero().flatten().tolist()
        src = torch.zeros(1, T, d, dtype=ad or w.emb.dtype, device=dev, requires_grad=True)
        hL, _ = forward(w, ids, inject=(layer, None, src), stop_layer=L, act_dtype=ad)
        eye = torch.zeros(nd, d, dtype=hL.dtype, device=dev); eye[torch.arange(nd, device=dev), dims] = 1
        for tp in srcs + [T - 1]:
            g = torch.cat([torch.autograd.grad(hL[0, tp], src, grad_outputs=eye[j0:j0 + db], retain_graph=True, is_grads_batched=True)[0][:, 0]
                           for j0 in range(0, nd, db)]).double()  # [nd, T, d]
            if tp == T - 1:
                Jempty += g[:, srcs].mean(1)
                continue
            for t in srcs:
                if t > tp:
                    continue
                gap = tp - t
                if gap not in G:
                    G[gap] = torch.zeros(nd, d, dtype=torch.float64, device=dev); n[gap] = 0
                G[gap] += g[:, t]; n[gap] += 1
    G = {gap: (G[gap] / n[gap]).cpu() for gap in G}
    return G, (Jempty / len(prompts)).cpu()


def gap_stats(G, nT):
    """Frobenius norm per gap and the gap-0 fraction rho_frob = ||G_0|| / ||(|T|-1)/2 * mean_{g>=1} G_g|| (sampled rows;
    the stationary proxy, audit A9). C16 additions: the norm-weighted future sum S = sum_{g>=1} (1 - g/|T|) ||G_g|| and
    the shares of gap 1, gaps 1-5 and gaps >= 20 in it; rho_exact = ||G_0|| / ||sum_{g>=1} (1 - g/|T|) G_g|| (the
    actual future term of Prop. 3 on the sampled rows)."""
    gaps = sorted(G)
    norms = {g: float(G[g].norm()) for g in gaps}
    Gbar = sum(G[g] for g in gaps if g > 0) / max(len(gaps) - 1, 1)
    rho = float(G[0].norm() / (((nT - 1) / 2) * Gbar).norm().clamp_min(1e-30))
    wt = {g: max(1 - g / nT, 0.0) for g in gaps if g > 0}; S = sum(wt[g] * norms[g] for g in wt)
    share = lambda sel: (sum(wt[g] * norms[g] for g in wt if sel(g)) / S) if S > 0 else None
    fut = sum(wt[g] * G[g] for g in wt) if wt else None
    return {"norms": norms, "rho_frob": rho, "S_future": S, "share_gap1": share(lambda g: g == 1), "share_gaps_1_5": share(lambda g: 1 <= g <= 5),
            "share_gaps_ge20": share(lambda g: g >= 20), "rho_exact": float(G[0].norm() / fut.norm().clamp_min(1e-30)) if fut is not None else None}


def reduction(q, D, layer, contexts, tokens, skip_first=16, acts=None, top=100):
    """cos(d(w), v_z(w)) for w in tokens. D: [V, d] lens token matrix. acts: optional [n, d] activations for the
    readout Spearman over the top-`top` tokens (by D h) restricted to the token list."""
    V = torch.stack([sj_vectors(q, [int(w)], contexts, [layer], skip_first)[layer] for w in tokens])  # [nw, d]
    Dw = D[torch.as_tensor(tokens)].double().cpu()
    cos = torch.nn.functional.cosine_similarity(Dw, V, dim=1)
    out = {"cos": cos, "V": V}
    if acts is not None:
        sps = []
        for h in acts:
            h = h.double().cpu(); a = Dw @ h; b = V @ h
            idx = torch.topk(a, min(top, len(tokens))).indices
            sps.append(spearman(a[idx], b[idx]))
        out["spearman_top"] = torch.tensor(sps)
    return out
