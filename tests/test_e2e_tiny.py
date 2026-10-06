"""End-to-end test of the sjlens package on a tiny random Qwen3ForCausalLM (float64, CPU).
Run: python tests/test_e2e_tiny.py"""
import sys, math, torch
sys.path.insert(0, "."); sys.path.insert(0, "tests")
from tiny import tiny
from sjlens.model.qwen3_min import Qwen3Min, pseudo_logits
from sjlens.lens import jlens
from sjlens.lens.backward import sj_vectors, prefix_jacobian
from sjlens.lens.forward import ContextBank, jvp_exact
from sjlens.lens.score import z_bg, hutchinson_sigma, conformal_threshold, vi_tokens, nnomp, z_fw
from sjlens.lens.bilinear import bilinear, R_alpha
from sjlens.beam.jbeam import JBeam, Cfg

torch.manual_seed(0)
m = tiny(); q = Qwen3Min(m); d = q.d; V = 256; L = 1; SK = 4; TAU = 20
ctx = [torch.randint(0, V, (TAU,)) for _ in range(6)]
res = []
def chk(name, err, tol): res.append((name, err, err < tol)); print(f"[{'PASS' if err < tol else 'FAIL'}] {name}: {err:.2e}")

J = jlens.fit(q, ctx, [L], skip_first=SK)[L]
G, Jempty = jlens.fit_gap_profile(q, ctx, L, skip_first=SK)
nT = TAU - SK - 1
Jre = sum((1 - g / nT) * G[g] for g in G)
chk("Prop3 J_lens = sum_g (1-g/|T|) G_g (real Qwen3 arch)", (J - Jre).abs().max().item(), 1e-9)
Jp0 = prefix_jacobian(q, [], ctx, L, SK)
chk("J(empty) from gap profile = prefix_jacobian([])", (Jempty - Jp0).abs().max().item(), 1e-9)
D = jlens.token_matrix(q.w, J)
print(f"      gap profile ||G_g||_F: " + ", ".join(f"g{g}={G[g].norm():.2f}" for g in sorted(G)[:6]) + " ...")
rho = (G[0].T @ q.w.lm.double().T).norm(dim=0) / ((nT - 1) / 2 * (sum(G[g] for g in G if g > 0) / (len(G) - 1)).T @ q.w.lm.double().T).norm(dim=0)
print(f"      gap-0 fraction rho_w median = {rho.median():.3f}; median cos(d(w), v(w)) = {torch.nn.functional.cosine_similarity(D, (Jp0.T @ q.w.lm.double().T).T, dim=1).median():.3f}")

s = [17, 42, 99]
v = sj_vectors(q, s, ctx, [L], SK)[L]
vsum = sum(prefix_jacobian(q, s[:i], ctx, L, SK).T @ q.w.lm.double()[s[i]] for i in range(3))
chk("Prop2 v(s) = sum_i J(s_<i)^T W_U[s_i]", (v - vsum).abs().max().item(), 1e-9)
vh = sj_vectors(q, [s[0]], ctx, [L], SK, mode="hybrid", D={L: D})[L]
chk("hybrid n=1 equals the J-lens row", (vh - D[s[0]]).abs().max().item(), 1e-12)
vab, vba = sj_vectors(q, [17, 42], ctx, [L], SK)[L], sj_vectors(q, [42, 17], ctx, [L], SK)[L]
print(f"      order index Omega(17,42) = {1 - torch.nn.functional.cosine_similarity(vab, vba, 0):.4f} (random model; expected near 0)")

bank = ContextBank(q, ctx, SK)
h = torch.randn(d, dtype=torch.float64) * 2
mask = jlens.valid_mask(TAU, SK)
ex = jvp_exact(q, bank.ids, L, mask, h, prefix=[17])
chk("jvp_exact = J(p) h from backward", (ex - prefix_jacobian(q, [17], ctx, L, SK) @ h).abs().max().item(), 1e-9)
with torch.no_grad(): hn = q.resid(bank.ids, L)[:, mask].norm(dim=-1).mean().item()
print(f"      mean source residual norm {hn:.3f}, |h| = {h.norm():.2f}")
for rho in (0.01, 0.05, 0.1, 0.3, 1.0):
    eps = rho * hn / h.norm().item()
    cp, cm = bank.perturbed(L, h, eps)
    uhat, _ = bank.direction_estimate([17], cp, cm, eps)
    print(f"      central FD rho={rho:g}: rel err vs exact = {(uhat - ex).norm() / ex.norm():.2e}")
eps = 0.05 * hn / h.norm().item()
cp, cm = bank.perturbed(L, h, eps)
num = bank.step_numerators([17], cp, cm, eps)
chk("step numerators = W_U J(p) h (rho=0.05)", (num - q.w.lm.double() @ ex).abs().max().item() / num.abs().max().item(), 2e-2)
chk("forward = backward phrase score", (v @ h - sum(float((q.w.lm.double() @ jvp_exact(q, bank.ids, L, mask, h, prefix=s[:i]))[s[i]]) for i in range(3))), 1e-8)

Jp = prefix_jacobian(q, [17], ctx, L, SK); exact_norm = (Jp.T @ q.w.lm.double().T).norm(dim=0)
for mnull in (8, 32, 128):
    g = torch.randn(mnull, d, dtype=torch.float64)
    nn_ = torch.stack([q.w.lm.double() @ (Jp @ gi) for gi in g])
    sig = hutchinson_sigma(nn_)
    print(f"      Hutchinson m={mnull}: median |sigma/||v|| - 1| = {(sig / exact_norm - 1).abs().median():.3f} (predicted ~{1/math.sqrt(2*mnull):.3f}); Spearman(proxy ||d(w)||, true) = {torch.corrcoef(torch.stack([D.norm(dim=1).argsort().argsort().double(), exact_norm.argsort().argsort().double()]))[0,1]:.3f}")

bg_ctx = [torch.randint(0, V, (TAU,)) for _ in range(120)]
with torch.no_grad():
    Hbg = torch.cat([q.resid(c.view(1, -1), L)[0, mask] for c in bg_ctx])
stats, rest = Hbg[:600], Hbg[600:]
mu = stats.mean(0); Hs = stats - mu
cal, test = rest[:600], rest[600:]
Dn = D / D.norm(dim=1, keepdim=True)
maxz = lambda X: torch.stack([z_bg(Dn, x, mu, Hs).max() for x in X])
tau = conformal_threshold(maxz(cal), 0.1)
fpr = float((maxz(test) > tau).float().mean())
print(f"      conformal: tau_0.1 = {tau:.2f} on max-z over {V} tokens; empirical FPR on {len(test)} fresh background = {fpr:.3f} (guarantee <= 0.1); Bonferroni z_fw(256) = {z_fw(256):.2f}")
Hc = Hbg - mu
zb = z_bg(Dn, Hbg[0], mu, Hs); print(f"      background z on a background activation: max {zb.max():.2f}, std {zb.std():.2f} (N(0,1)-scale)")

with torch.no_grad():
    z0 = pseudo_logits(q.w, bank.hL_clean[:, -1])
for al in (0.3, 1.0):
    print(f"      verbalizable information alpha={al}: real activation {vi_tokens(z0, D @ Hbg[0], al * nT):.4f} bits vs norm-matched random direction {vi_tokens(z0, D @ (torch.randn(d, dtype=torch.float64) * Hbg[0].norm() / math.sqrt(d)), al * nT):.4f} bits")

u = Hbg[0]
for a in (1e-3, 1e-2, 0.1, 0.3, 1.0):
    Ra = R_alpha(q, ctx[:3], L, mask, s, u, a)
    lin = a * nT * float(sj_vectors(q, s, ctx[:3], [L], SK)[L] @ u)
    print(f"      Prop8 alpha={a:<5g}: R_alpha={Ra:+.4f}, linear={lin:+.4f}, beta(alpha)={Ra/lin:.3f}")

m1 = torch.zeros(TAU, dtype=torch.bool); m1[SK:10] = True
m2 = torch.zeros(TAU, dtype=torch.bool); m2[10:TAU - 1] = True
u1, u2 = torch.randn(d, dtype=torch.float64), torch.randn(d, dtype=torch.float64)
g1, g2, b12 = bilinear(q, ctx[:2], L, s, m1, u1, m2, u2)
e = 1e-5
def Lab(a, b):
    tot = 0
    for c in ctx[:2]:
        c = c.view(1, -1); ids = torch.cat([c, torch.tensor(s).view(1, -1)], 1)
        M = torch.zeros(1, ids.shape[1], d, dtype=torch.float64); M[0, :TAU] = a * m1[:, None] * u1 + b * m2[:, None] * u2
        from sjlens.model.qwen3_min import forward
        hL, _ = forward(q.w, ids, inject=(L, None, M))
        tot += sum(float(pseudo_logits(q.w, hL[0, TAU + i - 1])[s[i]]) for i in range(3))
    return tot / 2
fd12 = (Lab(e, e) - Lab(e, -e) - Lab(-e, e) + Lab(-e, -e)) / (4 * e * e)
chk("bilinear mixed derivative = FD mixed difference", abs(b12 - fd12) / max(abs(fd12), 1e-9), 1e-4)
print(f"      first-order |g1|,|g2| = {abs(g1):.3f}, {abs(g2):.3f}; bilinear B[u1,u2] = {b12:.4f}; ratio |B|/(|g1||g2|)^(1/2) = {abs(b12)/math.sqrt(abs(g1*g2)+1e-12):.3f}")

nulls = [Hc[i] for i in range(16)]
jb = JBeam(q, bank, D, L, nulls, Cfg(seeds=4, beams=4, max_len=4, eps=0.05 * hn / Hbg[5].norm().item(), m_null=16, inject="h", k_pursuit=8), mu)
log = []
out, coefs = jb.read(Hbg[5], log)
print(f"      J-beam on a real activation: {len(out)} phrases; top: {[(t, round(z, 2)) for t, z, *_ in out[:5]]}; pursuit atoms {len(coefs)}")
print(f"      steps logged: {len(log)}; first: {log[0]['top'][:3] if log else None}")
print("\nSUMMARY:", sum(ok for *_, ok in res), "/", len(res), "checks passed")
