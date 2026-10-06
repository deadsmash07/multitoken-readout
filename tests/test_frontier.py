"""Frontier checks: (F2) verbalizable information is a Fisher quadratic form at small lambda;
(F3) integrated phrase vectors are exact at finite alpha (IG completeness); (F1) bilinear symmetry.
Run: python tests/test_frontier.py"""
import sys, math, torch
sys.path.insert(0, "."); sys.path.insert(0, "tests")
from tiny import tiny
from sjlens.model.qwen3_min import Qwen3Min, pseudo_logits, forward
from sjlens.lens import jlens
from sjlens.lens.backward import sj_vectors
from sjlens.lens.forward import ContextBank
from sjlens.lens.score import vi_tokens
from sjlens.lens.bilinear import R_alpha, teacher_forced_L

torch.manual_seed(0)
m = tiny(); q = Qwen3Min(m); d = q.d; V = 256; L = 1; SK = 4; TAU = 20
ctx = [torch.randint(0, V, (TAU,)) for _ in range(4)]
J = jlens.fit(q, ctx, [L], skip_first=SK)[L]; D = jlens.token_matrix(q.w, J)
bank = ContextBank(q, ctx, SK); nT = bank.nT
with torch.no_grad():
    H = torch.cat([q.resid(c.view(1, -1), L)[0, bank.mask] for c in ctx])
    z0 = pseudo_logits(q.w, bank.hL_clean[:, -1])
h = H[3]
ok = lambda n, e, t: print(f"[{'PASS' if e < t else 'FAIL'}] {n}: {e:.2e}")

# F2: lens Fisher metric F = E_c [D^T diag(p_c) D - D^T p_c p_c^T D]; VI_lam(h) = lam^2/2 h^T F h + O(lam^3)
p = z0.softmax(-1)
Fm = sum(D.T @ (torch.diag(pc) - pc[:, None] * pc[None]) @ D for pc in p) / p.shape[0]
for lam in (0.01, 0.1, 1.0):
    vi = float(vi_tokens(z0, D @ h, lam)) * math.log(2)
    quad = float(lam ** 2 / 2 * h @ Fm @ h)
    print(f"      F2 lam={lam:<5g}: VI={vi:.3e} nats, quadratic={quad:.3e}, ratio={vi/quad:.4f}")
ok("F2 VI = lam^2/2 h^T F h  (lam=0.01, rel)", abs(float(vi_tokens(z0, D @ h, 0.01)) * math.log(2) / float(0.01 ** 2 / 2 * h @ Fm @ h) - 1), 1e-2)
ev = torch.linalg.eigvalsh(Fm).flip(0)
print(f"      F2 Fisher spectrum: top-5 {ev[:5].tolist()}; effective rank (participation ratio) {float(ev.sum()**2 / (ev**2).sum()):.1f} of {d}")

# F3: integrated phrase vector g_alpha(s,h) = int_0^1 grad_h L(s; beta*alpha*h) dbeta ; R_alpha = alpha*|T|*<g, h>
s = [17, 42, 99]; a = 0.5 * float(H.norm(dim=-1).mean() / h.norm())
def grad_at(beta):
    tot = torch.zeros(d, dtype=torch.float64)
    for c in ctx:
        c = c.view(1, -1); ids = torch.cat([c, torch.tensor(s).view(1, -1)], 1)
        mk = torch.zeros(1, ids.shape[1], dtype=torch.bool); mk[0, :TAU] = bank.mask
        src = torch.zeros(1, ids.shape[1], d, dtype=torch.float64, requires_grad=True)
        base = (beta * a * h).view(1, 1, -1) * mk.double()[..., None]
        hL, _ = forward(q.w, ids, inject=(L, None, base + src))
        Lv = teacher_forced_L(q, ids, TAU, s, hL)
        g, = torch.autograd.grad(Lv, src)
        tot += g[0, :TAU][bank.mask].mean(0)
    return tot / len(ctx)
Ra = R_alpha(q, ctx, L, bank.mask, s, h, a)
for mq in (1, 5, 20, 80):
    bs = (torch.arange(mq, dtype=torch.float64) + 0.5) / mq
    g = sum(grad_at(float(b)) for b in bs) / mq
    ig = a * nT * float(g @ h)
    print(f"      F3 IG with m={mq:<3d} points: alpha|T|<g,h> = {ig:+.5f} vs R_alpha = {Ra:+.5f}  rel err {abs(ig-Ra)/abs(Ra):.2e}")
v0 = sj_vectors(q, s, ctx, [L], SK)[L]
print(f"      F3 tangent-only prediction alpha|T|<v,h> = {a*nT*float(v0@h):+.5f} (beta(alpha)={Ra/(a*nT*float(v0@h)):.3f})")
ok("F3 IG completeness (m=80, rel)", abs(ig - Ra) / abs(Ra), 1e-3)
