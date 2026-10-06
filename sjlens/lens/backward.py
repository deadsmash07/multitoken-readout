"""Backward SJ-lens: phrase vectors by one VJP per (phrase, context).

v_l(s) = mean_c mean_{t in T_c} sum_i d z_{s_i, tau+i-2} / d h_{l,t}   (pseudo-logit, bench convention)
v_lp   = same with log p_0(s_i | c, s_<i) (through the final norm; centered by the model's expectation)
hybrid: first-step term replaced by the released d_l(s_1) so the n=1 readout is exactly the bench J-lens.
        Scale: d_l(s_1) = J_l^T W_U[s_1] sums over the ~|T| lens targets (mean over sources) while the later terms
        J_l(s_<i)^T W_U[s_i] are single-target (mean over sources), so the hybrid vector is dominated by its first
        term by ~(|T|-1)/2 in the gap-stationary limit (Cor. 3.1; measured ||J(p1)^T W_U[p2]|| / ||d(p1)|| = 0.013 /
        0.008 / 0.002 at L11 / L17 / L22 of Qwen3-1.7B). Left unscaled on purpose: scripts/e11_fragment.py reports
        the norm-matched and (|T|-1)/2-scaled variants, beam/jbeam.py standardises each step by its null scale.
"""
import torch, torch.nn.functional as F
from ..model.qwen3_min import forward, logits, pseudo_logits
from .jlens import valid_mask


def sj_vectors(q, phrase, contexts, layers, skip_first=16, mode="z", D=None):
    """returns {layer: v[d]} for one phrase (list[int]) over a list of context id tensors."""
    w = q.w; n = len(phrase); out = {l: torch.zeros(q.d, dtype=torch.float64) for l in layers}; ad = getattr(q, "act_dtype", None)
    ph = torch.tensor(phrase, dtype=torch.long)
    for c in contexts:
        c = c.view(1, -1); tau = c.shape[1]; ids = torch.cat([c, ph.view(1, -1).to(c.device)], 1)
        mask = torch.zeros(ids.shape[1], dtype=torch.bool); mask[:tau] = valid_mask(tau, skip_first)
        for l in layers:
            src = torch.zeros(1, ids.shape[1], q.d, dtype=ad or w.emb.dtype, device=w.emb.device, requires_grad=True)
            hL, _ = forward(w, ids, inject=(l, None, src), act_dtype=ad)
            i0 = 1 if (mode == "hybrid" and D is not None) else 0
            if n - i0 > 0:
                pos = [tau + i - 1 for i in range(i0, n)]
                if mode == "lp":
                    obj = sum(F.log_softmax(logits(w, hL[0, p]), -1)[phrase[i]] for i, p in zip(range(i0, n), pos))
                else:
                    obj = sum(pseudo_logits(w, hL[0, p])[phrase[i]] for i, p in zip(range(i0, n), pos))
                g, = torch.autograd.grad(obj, src)
                out[l] += g[0, mask].double().mean(0).cpu()
            if i0 == 1:
                out[l] += D[l][phrase[0]].double().cpu()
    return {l: out[l] / len(contexts) for l in layers}


def prefix_jacobian(q, prefix, contexts, layer, skip_first=16):
    """exact J_l(p) as [d, d] by d backward passes (small models / diagnostics only)."""
    w = q.w; d = q.d; J = torch.zeros(d, d, dtype=torch.float64); pf = torch.tensor(prefix, dtype=torch.long); ad = getattr(q, "act_dtype", None)
    for c in contexts:
        c = c.view(1, -1); tau = c.shape[1]; ids = torch.cat([c, pf.view(1, -1).to(c.device)], 1) if len(prefix) else c
        mask = torch.zeros(ids.shape[1], dtype=torch.bool); mask[:tau] = valid_mask(tau, skip_first)
        src = torch.zeros(1, ids.shape[1], d, dtype=ad or w.emb.dtype, device=w.emb.device, requires_grad=True)
        hL, _ = forward(w, ids, inject=(layer, None, src), act_dtype=ad)
        for j in range(d):
            g, = torch.autograd.grad(hL[0, -1, j], src, retain_graph=True)
            J[j] += g[0, mask].double().mean(0).cpu()
    return J / len(contexts)
