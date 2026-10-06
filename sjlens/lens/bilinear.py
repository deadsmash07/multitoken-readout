"""Second-order (bilinear) sequence lens and the finite-scale readout R_alpha.

R_alpha(s, u; masks) = mean_c [ L_alpha(s|c) - L_0(s|c) ], L = teacher-forced pseudo-logit sum (mode 'z')
                                                         or log-likelihood (mode 'lp').
B(s)[u1, u2] = d^2 L / d alpha d beta at 0 with alpha*u1 on mask1 and beta*u2 on mask2 (forward-over-forward).
A window reader that is additive across positions is the gradient; the bilinear term is the smallest object
that can represent 'A at position t1 AND B at position t2' beyond their sum (binding).
"""
import torch, torch.nn.functional as F
from ..model.qwen3_min import forward, pseudo_logits, logits


def teacher_forced_L(q, ids, tau, phrase, hL, mode="z"):
    pos = [tau + i - 1 for i in range(len(phrase))]
    if mode == "lp":
        return sum(F.log_softmax(logits(q.w, hL[:, p]), -1)[:, phrase[i]] for i, p in enumerate(pos)).sum(0)
    return sum(pseudo_logits(q.w, hL[:, p])[:, phrase[i]] for i, p in enumerate(pos)).sum(0)


def R_alpha(q, contexts, layer, mask, phrase, u, alpha, mode="z"):
    ph = torch.tensor(phrase, dtype=torch.long).view(1, -1)
    tot = 0.0
    with torch.no_grad():
        for c in contexts:
            c = c.view(1, -1); tau = c.shape[1]; ids = torch.cat([c, ph.to(c.device)], 1)
            m = torch.zeros(1, ids.shape[1], dtype=torch.bool, device=c.device); m[0, :tau] = mask.to(c.device)
            ad = getattr(q, "act_dtype", None)
            hA, _ = forward(q.w, ids, inject=(layer, m, alpha * u), act_dtype=ad)
            h0, _ = forward(q.w, ids, act_dtype=ad)
            tot += float(teacher_forced_L(q, ids, tau, phrase, hA, mode) - teacher_forced_L(q, ids, tau, phrase, h0, mode))
    return tot / len(contexts)


def _L_of_scales(q, ids, tau, phrase, layer, m1, u1, m2, u2, mode):
    def f(ab):
        a, b = ab[0], ab[1]
        d = a * m1.to(u1.dtype)[..., None] * u1 + b * m2.to(u2.dtype)[..., None] * u2
        hL, _ = forward(q.w, ids, inject=(layer, None, d), act_dtype=getattr(q, "act_dtype", None))
        return teacher_forced_L(q, ids, tau, phrase, hL, mode)
    return f


def bilinear(q, contexts, layer, phrase, m1, u1, m2, u2, mode="z"):
    """returns (grad wrt alpha, grad wrt beta, mixed second derivative), averaged over contexts.
    m1, m2: [tau] boolean position masks inside the context; u1, u2: [d] directions."""
    ph = torch.tensor(phrase, dtype=torch.long).view(1, -1); g1 = g2 = b12 = 0.0
    for c in contexts:
        c = c.view(1, -1); tau = c.shape[1]; ids = torch.cat([c, ph.to(c.device)], 1)
        M1 = torch.zeros(1, ids.shape[1], dtype=torch.bool, device=c.device); M1[0, :tau] = m1.to(c.device)
        M2 = torch.zeros(1, ids.shape[1], dtype=torch.bool, device=c.device); M2[0, :tau] = m2.to(c.device)
        f = _L_of_scales(q, ids, tau, phrase, layer, M1, u1, M2, u2, mode)
        x0 = torch.zeros(2, dtype=u1.dtype, device=u1.device)
        H = torch.func.hessian(f)(x0); G = torch.func.jacrev(f)(x0)
        g1 += float(G[0]); g2 += float(G[1]); b12 += float(H[0, 1])
    n = len(contexts)
    return g1 / n, g2 / n, b12 / n
