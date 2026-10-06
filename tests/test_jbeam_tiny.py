"""J-beam step combination on the tiny random Qwen3 (float64, CPU): scale invariance of the standardised phrase
scores, agreement with the exact backward phrase vector, step 1 = bench J-lens, and an end-to-end run with the
default Cfg. The identity checks pin the Hutchinson (mc) normaliser and the bench seed rule explicitly since the
Cfg defaults moved to the tuned configuration (eb / dnorm, lens seeds, tau stop; docs/JBEAM_TUNING.md, 25 Sep).
Run: python tests/test_jbeam_tiny.py"""
import sys, math, torch
sys.path.insert(0, "."); sys.path.insert(0, "tests")
from tiny import tiny
from sjlens.model.qwen3_min import Qwen3Min
from sjlens.lens import jlens
from sjlens.lens.backward import prefix_jacobian
from sjlens.lens.forward import ContextBank
from sjlens.beam.jbeam import JBeam, Cfg

torch.manual_seed(0)
m = tiny(); q = Qwen3Min(m); d = q.d; V = 256; L = 1; SK = 4; TAU = 20
ctx = [torch.randint(0, V, (TAU,)) for _ in range(6)]
res = []
def chk(name, err, tol): res.append((name, err, err < tol)); print(f"[{'PASS' if err < tol else 'FAIL'}] {name}: {err:.2e}")

J = jlens.fit(q, ctx, [L], skip_first=SK)[L]; D = jlens.token_matrix(q.w, J); mask = jlens.valid_mask(TAU, SK)
bank = ContextBank(q, ctx, SK)
with torch.no_grad():
    Hbg = torch.cat([q.resid(c.view(1, -1), L)[0, mask] for c in [torch.randint(0, V, (TAU,)) for _ in range(40)]])
    hn = q.resid(bank.ids, L)[:, mask].norm(dim=-1).mean().item()
mu = Hbg.mean(0); Hc = Hbg - mu; nulls = [Hc[i] for i in range(16)]; h = Hbg[5]; hc = h - mu  # z-scores and injection use hc (Prop. 4)
# q_fw = 80 puts z_fw(256) at 0.5 so that beams extend on a random model (z_fw(256, 0.05) = 3.55 stops every beam at length 1)
base = dict(seeds=4, beams=4, max_len=4, eps=0.05 * hn / h.norm().item(), m_null=16, inject="h", k_pursuit=8, q_fw=80.0, normalizer="mc", seed_rule="bench", stop="zfw")
run = lambda Dm, **kw: JBeam(q, bank, Dm, L, nulls, Cfg(**base, **kw), mu).read(h)[0]
flat = lambda out: torch.tensor([x for t, z, zs, st in out for x in (z, zs, *[v for _, v in st])])

out = {c: run(D * c, combine="std") for c in (1.0, 55.0, 1 / 55)}
print(f"      std phrases (c=1): {[(t, round(z, 3)) for t, z, *_ in out[1.0]]}")
chk("std: beams extend past one token (max length >= 2, -1 if not)", -max(len(t) for t, *_ in out[1.0]), -1)
for c in (55.0, 1 / 55):
    chk(f"std: phrases identical under D -> {c:g} D", float(([t for t, *_ in out[c]] != [t for t, *_ in out[1.0]]) or (flat(out[c]) - flat(out[1.0])).abs().max()), 1e-9)
outs = {c: run(D * c, combine="stouffer") for c in (1.0, 55.0)}
chk("stouffer: phrases identical under D -> 55 D", float(([t for t, *_ in outs[55.0]] != [t for t, *_ in outs[1.0]]) or (flat(outs[55.0]) - flat(outs[1.0])).abs().max()), 1e-9)
chk("stouffer: z = sum_k z_k / sqrt(n)", max(abs(z - sum(v for _, v in st) / math.sqrt(len(st))) for t, z, zs, st in outs[1.0]), 1e-12)
outr = {c: run(D * c, combine="raw") for c in (1.0, 55.0)}
print(f"      raw phrases (c=1): {[(t, round(z, 3)) for t, z, *_ in outr[1.0]]}; (c=55): {[(t, round(z, 3)) for t, z, *_ in outr[55.0]]}")
chk("raw: multi-token scores change under D -> 55 D (control; -max diff)", -float((flat(outr[55.0]) - flat(outr[1.0])).abs().max()) if [t for t, *_ in outr[55.0]] == [t for t, *_ in outr[1.0]] else -1.0, -1e-3)

# step 1 = bench J-lens: seeds are the top tokens of (D h) / ||d(w)||; single-token z is the null-based z of d(w)
Dh, Dhc, Dg = D @ h, D @ hc, torch.stack([D @ g for g in nulls]); seeds = set(torch.topk(Dh / D.norm(dim=1), base["seeds"]).indices.tolist())
chk("step 1 seeds = top-k of (D h)/||d(w)|| (missing count)", float(len(seeds - {st[0][0] for *_, st in out[1.0]})), 0.5)
chk("step 1 z = (D (h - mu))[w] / rms_j (D g_j)[w]", max(abs(st[0][1] - float(Dhc[st[0][0]] / Dg[:, st[0][0]].pow(2).mean().sqrt())) for *_, st in out[1.0]), 1e-12)

# std phrase z equals the background z (same 16 nulls) of the sigma-reweighted exact backward phrase vector sum_k v_k / sigma_k
G = torch.stack(nulls); W = q.w.lm.double(); worst = worst_step = 0.0
for t, z, zs, st in out[1.0]:
    if len(t) < 2: continue
    vk = [D[t[0]]] + [prefix_jacobian(q, t[:k], ctx, L, SK).T @ W[t[k]] for k in range(1, len(t))]
    sig = [(G @ v).pow(2).mean().sqrt() for v in vk]; vstd = sum(v / s_ for v, s_ in zip(vk, sig))
    worst = max(worst, abs(float(vstd @ hc / (G @ vstd).pow(2).mean().sqrt()) / z - 1))
    worst_step = max(worst_step, max(abs(float(v @ hc / s_) / st[k][1] - 1) for k, (v, s_) in enumerate(zip(vk, sig))))
chk("std phrase z = z_bg of sum_k v_k/sigma_k from exact VJPs (rel; FD rho=0.05)", worst, 2e-2)
chk("step z_k = <v_k, h>/sigma_k from exact VJPs (rel)", worst_step, 2e-2)

# end to end with the default Cfg (hJ injection, MC normalizer, std combination), small sizes
log = []
outd, coefs = JBeam(q, bank, D, L, nulls, Cfg(seeds=3, beams=3, max_len=3, eps=base["eps"], k_pursuit=8), mu).read(h, log)
print(f"      default Cfg end to end: {len(outd)} phrases; top: {[(t, round(z, 2)) for t, z, *_ in outd[:3]]}; pursuit atoms {len(coefs)}; steps logged {len(log)}")
chk("default Cfg returns phrases", -float(len(outd)), -0.5)
jb = JBeam(q, bank, D, L, nulls, Cfg(seeds=3, beams=3, max_len=3, k_pursuit=8), mu); outr_ = jb.read(h)[0]  # default relative eps (rho = 0.05)
chk("relative eps: rho * mean source norm / ||u|| gives rho = 0.05 for the nulls", max(abs(e * float(g.norm()) / jb.hn - 0.05) for g, e in zip(nulls, jb.null_eps)), 1e-12)

print("\nSUMMARY:", sum(ok for *_, ok in res), "/", len(res), "checks passed")
