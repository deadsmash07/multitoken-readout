"""Pin the eval modules (sjlens/eval/*) to the reference implementations on the tiny random Qwen3 (float64, CPU).
Run: python tests/test_eval_tiny.py"""
import sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "tests")
from tiny import tiny
from sjlens.model.qwen3_min import Qwen3Min
from sjlens.lens import jlens
from sjlens.lens.backward import prefix_jacobian
from sjlens.lens.forward import ContextBank, jvp_lens, gap0_jvp
from sjlens.eval import fit_fast
from sjlens.eval.parity import check_parity
from sjlens.eval.gap_profile import gap_profile_sampled, gap_stats, reduction
from sjlens.eval.fwd_bwd import side_result, fd_curve
from sjlens.eval.common import background_activations

torch.manual_seed(0)
m = tiny(); q = Qwen3Min(m); d = q.d; V = 256; L = 1; SK = 4; TAU = 20
ctx = [torch.randint(0, V, (TAU,)) for _ in range(6)]
res = []
def chk(name, err, tol): res.append((name, err, err < tol)); print(f"[{'PASS' if err < tol else 'FAIL'}] {name}: {err:.2e}")

# E1 parity against the HF model: logits, residuals after blocks 0..2, 8-token extension, injection after block 1.
# Both sides build RoPE tables in float32, so the floor is ~1e-8 relative even in float64.
ids = torch.stack(ctx[:3]); mask = torch.zeros(3, TAU, dtype=torch.bool); mask[:, SK:TAU - 1] = True
delta = torch.randn(d, dtype=torch.float64)
par = check_parity(m, q, ids, layers=[0, 1, 2], n_ext=8, inject_layer=1, delta=delta, mask=mask)
for k, v in par.items():
    chk(f"E1 parity {k} (rel)", v["rel"], 1e-6)

# batched fit equals the reference estimator
J_ref = jlens.fit(q, ctx, [L], skip_first=SK)[L]
J_fast = fit_fast.fit(q, ctx, [L], skip_first=SK, dim_batch=24)[L]
chk("fit_fast = jlens.fit", (J_fast - J_ref).abs().max().item(), 1e-12)
hh = torch.randn(d, dtype=torch.float64)
chk("jvp_lens = J_lens h (identity 4, same contexts)", (jvp_lens(q, torch.stack(ctx), L, jlens.valid_mask(TAU, SK), hh) - J_ref @ hh).abs().max().item(), 1e-9)

# sampled gap profile equals the full one on the sampled rows
G_full, Je_full = jlens.fit_gap_profile(q, ctx, L, skip_first=SK)
dims = [0, 5, 17, 42, 63]
G_s, Je_s = gap_profile_sampled(q, ctx, L, dims, skip_first=SK)
chk("gap_profile_sampled rows = fit_gap_profile", max((G_s[g] - G_full[g][dims]).abs().max().item() for g in G_s), 1e-12)
chk("gap_profile_sampled J(empty) rows", (Je_s - Je_full[dims]).abs().max().item(), 1e-12)
G_c, Je_c = gap_profile_sampled(q, ctx, L, dims, skip_first=SK, dim_batch=2)
chk("gap_profile_sampled dim_batch=2 chunking", max((G_c[g] - G_s[g]).abs().max().item() for g in G_s) + (Je_c - Je_s).abs().max().item(), 1e-12)
nT = TAU - SK - 1
g0 = gap0_jvp(q, torch.stack(ctx), L, jlens.valid_mask(TAU, SK), hh, pos_batch=5)
chk("gap0_jvp = G_0 h", (g0 - G_full[0] @ hh).abs().max().item(), 1e-9)
fut = jvp_lens(q, torch.stack(ctx), L, jlens.valid_mask(TAU, SK), hh) - g0
chk("jvp_lens - gap0 = sum_{g>=1} (1-g/|T|) G_g h", (fut - sum((1 - g / nT) * G_full[g] @ hh for g in G_full if g > 0)).abs().max().item(), 1e-9)
st = gap_stats(G_s, nT); st_full = gap_stats(G_full, nT)
print(f"      rho_frob sampled {st['rho_frob']:.3f} vs full {st_full['rho_frob']:.3f}; ||G_0|| {st['norms'][0]:.2f}, ||G_1|| {st['norms'][1]:.2f}")

# reduction: v_z(w) from sj_vectors equals J(empty)^T W_U[w] (identity 5a); cosine reproduces the e2e number's scale
D = jlens.token_matrix(q.w, J_ref)
Jp0 = prefix_jacobian(q, [], ctx, L, SK)
toks = list(range(0, 256, 8))
H = background_activations(q, ctx, L, jlens.valid_mask(TAU, SK))
red = reduction(q, D, L, ctx, toks, skip_first=SK, acts=H[:5])
V_ref = (Jp0.T @ q.w.lm.double()[toks].T).T
chk("reduction v_z(w) = J(empty)^T W_U[w]", (red["V"] - V_ref).abs().max().item(), 1e-12)
print(f"      median cos(d(w), v_z(w)) over {len(toks)} tokens = {red['cos'].median():.3f} (e2e reported 0.620 over all 256); readout Spearman top-32 = {red['spearman_top'].mean():.3f}")

# E3 gate on the tiny model with the lens fitted on the same contexts: Spearman should be ~1 (only FD error)
bank = ContextBank(q, ctx, SK)
e3 = side_result(q, bank, D.float(), H[:8], L, rho=0.05, top=64); sp, err = e3["top"], e3["err"]
chk("E3 side result Spearman >= 0.95 (1 - min)", 1 - sp.min().item(), 0.05)
print(f"      E3: Spearman top-64 mean {sp.mean():.4f}, min {sp.min():.4f}; all-token {e3['all'].mean():.4f}; FD-vs-exact rel err mean {err.mean():.2e}")

# E4 curve reproduces the e2e numbers (rho=0.05 -> 1.74e-3 there, with a random h of norm 2*sqrt(d))
h = torch.randn(d, dtype=torch.float64) * 2
curve, hn = fd_curve(q, q, ctx, ctx, L, h, prefix=[17], rhos=[0.01, 0.05, 0.1, 0.3])
for r, e in curve.items(): print(f"      E4 rho={r:g}: rel err {e:.2e}")
chk("E4 central FD rho=0.05 rel err < 1e-2", curve[0.05], 1e-2)

# mixed precision: bf16 weights with float64 activations; differs from the float64 model only by weight rounding (~1e-2)
import copy
m16 = copy.deepcopy(m).to(torch.bfloat16); q16 = Qwen3Min(m16, act_dtype=torch.float64); q16.w.lm = q16.w.lm  # bf16 weights
with torch.no_grad():
    a64 = q.resid(ids, L); a16m = q16.resid(ids, L); a16 = Qwen3Min(m16).resid(ids, L).double()
chk("mixed mode (bf16 weights, f64 acts) within 3% of f64", ((a16m - a64).norm() / a64.norm()).item(), 3e-2)
print(f"      mixed-mode rel diff {((a16m - a64).norm() / a64.norm()).item():.2e} vs pure bf16 {((a16 - a64).norm() / a64.norm()).item():.2e}; activations dtype {a16m.dtype}")
jm = jvp_lens(q16, torch.stack(ctx), L, jlens.valid_mask(TAU, SK), hh)
chk("mixed-mode jvp_lens within 5% of f64", ((jm - J_ref @ hh).norm() / (J_ref @ hh).norm()).item(), 5e-2)

print("\nSUMMARY:", sum(ok for *_, ok in res), "/", len(res), "checks passed")
