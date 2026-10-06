"""Agent-D fixes on the tiny random Qwen3 (CPU): A5 forward/backward "lp" agreement (FD and exact JVP), the batched
CacheSet path against the per-cache path, A6 mixed mode (bf16 weights, fp32 activations) equal to fp32 on
sj_vectors and the lens fit, A7 wsbench method construct + read, shrink_sigma on synthetic nulls, and J-beam
normaliser variants end to end. Run: python tests/test_fixes_tiny.py"""
import sys, os, copy, math, tempfile, torch
sys.path.insert(0, "."); sys.path.insert(0, "tests")
from tiny import tiny
from sjlens.model.qwen3_min import Qwen3Min
from sjlens.lens import jlens
from sjlens.lens.backward import sj_vectors
from sjlens.lens.forward import ContextBank, jvp_step, jvp_exact
from sjlens.lens.score import shrink_sigma, hutchinson_sigma, lens_null_scale
from sjlens.eval import fit_fast
from sjlens.beam.jbeam import JBeam, Cfg

torch.manual_seed(0)
res = []
def chk(name, err, tol): res.append((name, err, err < tol)); print(f"[{'PASS' if err < tol else 'FAIL'}] {name}: {err:.2e}")
rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-30))

m = tiny(); q = Qwen3Min(m); d = q.d; V = 256; L = 1; SK = 4; TAU = 20
ctx = [torch.randint(0, V, (TAU,)) for _ in range(5)]
J = jlens.fit(q, ctx, [L], skip_first=SK)[L]; D = jlens.token_matrix(q.w, J); mask = jlens.valid_mask(TAU, SK)
bank = ContextBank(q, ctx, SK); ids = torch.stack(ctx)
with torch.no_grad(): Hbg = torch.cat([q.resid(c.view(1, -1), L)[0, mask] for c in [torch.randint(0, V, (TAU,)) for _ in range(30)]])
mu = Hbg.mean(0); Hc = Hbg - mu; u = Hc[3]; nulls = Hc[:16]

# --- A5: forward "lp" = backward "lp" (derivative of log p(w | c, p) through the final norm) ---
prefix = [7]; eps = 1e-4
cp, cm = bank.perturbed(L, u, eps)
for mode in ("z", "lp"):
    fd = bank.step_numerators(prefix, cp, cm, eps, mode)  # [V]
    ex = jvp_step(q, ids, L, mask, u, prefix, mode)
    bw = torch.stack([(sj_vectors(q, prefix + [w], ctx, [L], SK, mode=mode)[L] - sj_vectors(q, prefix, ctx, [L], SK, mode=mode)[L]) @ u.double() for w in range(0, V, 16)])
    chk(f"A5 {mode}: forward FD (eps 1e-4, f64) = backward step term . u (rel)", rel(fd[::16].double(), bw), 1e-6)
    chk(f"A5 {mode}: forward exact JVP = backward step term . u (rel)", rel(ex[::16].double(), bw), 1e-10)
    bset = bank.step_set(prefix, bank.cache_set(L, u.view(1, -1), [eps]), mode)[0]
    chk(f"A5 {mode}: batched step_set = step_numerators (rel)", rel(bset, fd), 1e-10)

# --- batched CacheSet path = per-cache path, for several directions and a 2-token prefix ---
U = torch.stack([u, nulls[0], nulls[1]]); epss = [0.05 * bank.resid(L)[:, mask].norm(dim=-1).mean().item() / g.norm().item() for g in U]
cs = bank.cache_set(L, U, epss, chunk=2)
for pf in ([7], [7, 42]):
    per = torch.stack([bank.step_numerators(pf, *bank.perturbed(L, g, e), e) for g, e in zip(U, epss)])
    chk(f"CacheSet: step_set = per-cache step_numerators, prefix {pf} (rel)", rel(bank.step_set(pf, cs), per), 1e-10)
pc = bank.step_set([7], cs, per_context=True); chk("CacheSet: per-context readouts average to the mean (rel)", rel(pc.mean(1), bank.step_set([7], cs)), 1e-12)

# --- A6: mixed mode (bf16 weights, fp32 activations) = fp32 model with bf16-representable weights ---
m32 = tiny(dtype=torch.float32)
for p in m32.parameters(): p.data = p.data.to(torch.bfloat16).to(torch.float32)
mbf = copy.deepcopy(m32).to(torch.bfloat16)
q32, qmix, qbf = Qwen3Min(m32), Qwen3Min(mbf, act_dtype=torch.float32), Qwen3Min(mbf)
s = [7, 42]
v32, vmix, vbf = (sj_vectors(qq, s, ctx, [L], SK)[L] for qq in (q32, qmix, qbf))
chk("A6: sj_vectors mixed = fp32 (rel)", rel(vmix, v32), 1e-5)
chk("A6: sj_vectors pure bf16 differs from fp32 (control; -rel)", -rel(vbf, v32), -1e-3)
J32, Jmix = (fit_fast.fit(qq, ctx[:2], [L], SK, dim_batch=16)[L] for qq in (q32, qmix))
chk("A6: fit_fast mixed = fp32 (rel)", rel(Jmix, J32), 1e-5)
vlp32, vlpmix = (sj_vectors(qq, s, ctx[:2], [L], SK, mode="lp")[L] for qq in (q32, qmix))
chk("A6: sj_vectors lp mixed = fp32 (rel)", rel(vlpmix, vlp32), 1e-5)
jm = jvp_exact(qmix, ids[:2], L, mask, u.float(), prefix=[7]); j32 = jvp_exact(q32, ids[:2], L, mask, u.float(), prefix=[7])
chk("A6: jvp_exact mixed = fp32 (rel)", rel(jm, j32), 1e-5)

# --- A7: wsbench method constructs from a make_contexts.py file and reads ---
from sjlens.wsbench_method import SJLens, Readout
tmp = tempfile.mkdtemp()
torch.save({"model": "tiny", "T": TAU, "ctx": ctx, "bg": [torch.randint(0, V, (TAU,)) for _ in range(6)], "source": "random"}, os.path.join(tmp, "ctx.pt"))
jlens.save(os.path.join(tmp, "lens.pt"), {L: J.float()}, {"source_layers": [L]})
class Tok:
    def decode(s, t): return " ".join(f"t{int(x)}" for x in t)
class Backend: model = m; tokenizer = Tok()
meth = SJLens(os.path.join(tmp, "lens.pt"), os.path.join(tmp, "ctx.pt"), cfg=Cfg(seeds=3, beams=3, max_len=3, m_null=8, k_pursuit=8, q_fw=80.0), skip_first=SK)
meth.bind(Backend()); r = meth.read(Hbg[5], L)
print(f"      wsbench read: {r.samples[:3]} scores {[round(x, 2) for x in r.scores[:3]]}")
chk("A7: SJLens.read returns a Readout with samples", -float(isinstance(r, Readout) and len(r.samples) > 0 and len(r.tokens) == len(r.samples)), -0.5)
chk("A7: background derived from the bg split ([N, d] at the layer)", -float(meth.bg[L].shape == (6 * int(mask.sum()), d)), -0.5)

# --- shrink_sigma on synthetic Gaussian nulls with a known sigma ---
g = torch.Generator().manual_seed(1); Vs = 4000; mn = 16
sig_true = torch.exp(torch.randn(Vs, generator=g, dtype=torch.float64) * 1.0); prior = sig_true * torch.exp(0.3 * torch.randn(Vs, generator=g, dtype=torch.float64)) * 3.0
nn_ = torch.randn(mn, Vs, generator=g, dtype=torch.float64) * sig_true
err = lambda s: float((s.log() - sig_true.log()).pow(2).mean().sqrt())
e_mc, e_eb, e_px, e_fl = err(shrink_sigma(nn_, kind="mc")), err(shrink_sigma(nn_, prior, "eb")), err(shrink_sigma(nn_, prior, "proxy")), err(shrink_sigma(nn_, prior, "floor", 1.0))
print(f"      shrink_sigma RMS log error: mc {e_mc:.3f} eb {e_eb:.3f} proxy {e_px:.3f} floor {e_fl:.3f} (prior noise 0.3, m={mn})")
chk("shrink: eb beats mc on synthetic nulls (eb - mc < 0)", e_eb - e_mc, 0.0)
chk("shrink: eb beats the rescaled proxy alone", e_eb - e_px, 0.0)
chk("shrink: mc at m=16 has the predicted log-sd sqrt(psi'(8))/2 = 0.18 (abs diff)", abs(e_mc - math.sqrt(float(torch.special.polygamma(1, torch.tensor(8.0)))) / 2), 0.02)
hi = torch.exp(torch.randn(Vs, generator=g, dtype=torch.float64) * 1.0)  # prior scale x 1e3 with the same shape: scale-invariant
chk("shrink: eb invariant to the prior's overall scale (rel)", rel(shrink_sigma(nn_, prior * 1e3, "eb"), shrink_sigma(nn_, prior, "eb")), 1e-10)
chk("shrink: lens_null_scale = rms over the background of (D g)[w] (rel)", rel(lens_null_scale(D, Hc), (D @ Hc.T).pow(2).mean(1).sqrt()), 1e-12)

# --- J-beam normaliser variants end to end (shared null caches and memo across configs) ---
h = Hbg[5]; hn = bank.resid(L)[:, mask].norm(dim=-1).mean().item()
base = dict(seeds=3, beams=3, max_len=3, m_null=16, k_pursuit=8, q_fw=80.0)
jb0 = JBeam(q, bank, D, L, nulls, Cfg(**base), mu, Hbg=Hbg); memo = {}; n_ok = 0; cfgs = []
for kw in (dict(normalizer="eb", prior="pooled", pool_prefixes=8), dict(normalizer="eb", prior="lens"), dict(normalizer="proxy", prior="wu"), dict(normalizer="floor", floor=0.5),
           dict(normalizer="raw"), dict(normalizer="mc", m_null=8), dict(normalizer="eb", prior="lens", expand=2, beams=4), dict(normalizer="eb", prior="lens", stop="none"),
           dict(normalizer="eb", prior="lens", stop="tau", tau=0.3), dict(normalizer="eb", prior="lens", support=0.5), dict(normalizer="eb", prior="lens", seed_rule="z"), dict(mode="lp", normalizer="eb", prior="lens")):
    cfg = Cfg(**{**base, **kw}); jb = JBeam(q, bank, D, L, nulls, cfg, mu, Hbg=Hbg, null_set=jb0.null_set if cfg.mode == "z" else None)
    log = []; out, _ = jb.read(h, log, memo if cfg.mode == "z" else None); n_ok += len(out) > 0; cfgs.append((kw, [(t, round(z, 2)) for t, z, *_ in out[:2]]))
print("      variants:", "; ".join(f"{list(kw.values())}: {o}" for kw, o in cfgs[:4]))
chk("J-beam: every normaliser / stop / expand variant returns phrases", float(len(cfgs) - n_ok), 0.5)
chk("J-beam: memo shares null readouts across configs (entries > 0)", -float(len(memo)), -0.5)
outn, _ = JBeam(q, bank, D, L, nulls, Cfg(**{**base, "normalizer": "eb", "prior": "lens", "stop": "none", "coherence": False, "max_len": 4}), mu, Hbg=Hbg, null_set=jb0.null_set).read(h)
chk("J-beam: stop='none' + coherence=False runs every seed to max_len", float(min(len(t) for t, *_ in outn) != 4), 0.5)

print("\nSUMMARY:", sum(ok for *_, ok in res), "/", len(res), "checks passed")
