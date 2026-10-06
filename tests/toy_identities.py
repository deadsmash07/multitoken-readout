"""Numerical verification of the sequence J-lens (SJ-lens) identities on a small
random pre-norm transformer. Everything runs in float64 on CPU so exact
identities are checked to ~1e-10; the finite-difference sweep also runs the
model in bf16 to show the precision floor.

Conventions follow anthropics/jacobian-lens and camilablank/workspace-bench:
  h_{l,t}   residual after block l at position t
  hL_t      residual after the last block (pre final-norm)
  z_{w,t}   = W_U[w] . hL_t   pseudo-logit (bench convention, no final norm)
  J-lens    J_l = mean_{t in T} sum_{t' in T, t'>=t} d hL_{t'} / d h_{l,t}
  SJ vector v_l(s) = mean_c mean_{t in T_c} sum_i d z_{s_i, tau+i-2} / d h_{l,t}
"""
import math, torch, torch.nn.functional as F
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
D, H, V, NL, FF = 32, 4, 64, 3, 64
TAU, SKIP = 12, 2                     # context length, attention-sink skip
LAYER = 0                             # read layer (output of block 0)

class Block(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.n1 = torch.nn.Parameter(torch.ones(D)); s.n2 = torch.nn.Parameter(torch.ones(D))
        s.qkv = torch.nn.Linear(D, 3 * D, bias=False); s.o = torch.nn.Linear(D, D, bias=False)
        s.g = torch.nn.Linear(D, FF, bias=False); s.u = torch.nn.Linear(D, FF, bias=False); s.d = torch.nn.Linear(FF, D, bias=False)
    def forward(s, x):
        B, T, _ = x.shape
        y = rms(x, s.n1)
        q, k, v = s.qkv(y).split(D, -1)
        q, k, v = [a.view(B, T, H, D // H).transpose(1, 2) for a in (q, k, v)]
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // H)
        att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")).softmax(-1)
        x = x + s.o((att @ v).transpose(1, 2).reshape(B, T, D))
        y = rms(x, s.n2)
        return x + s.d(F.silu(s.g(y)) * s.u(y))

def rms(x, g): return g * x / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()

class Model(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = torch.nn.Embedding(V, D); s.blocks = torch.nn.ModuleList(Block() for _ in range(NL))
        s.gamma = torch.nn.Parameter(1 + 0.1 * torch.randn(D)); s.WU = torch.nn.Parameter(torch.randn(V, D) / math.sqrt(D))
    def forward(s, ids, inject=None):
        """inject = (layer, delta[B,T,D]) adds delta to the residual after that block.
        returns hL (pre-norm final residual)."""
        x = s.emb(ids)
        for l, b in enumerate(s.blocks):
            x = b(x)
            if inject is not None and inject[0] == l: x = x + inject[1]
        return x
    def from_layer(s, x, l0):
        for l in range(l0 + 1, NL): x = s.blocks[l](x)
        return x

m = Model()
for p in m.parameters(): p.requires_grad_(False)
WU = m.WU.detach()
B = 4
ctx = torch.randint(0, V, (B, TAU))
src = torch.arange(SKIP, TAU - 1)           # valid source positions (skip sinks, drop last), |T|=9
NS = len(src)

def resid_at_layer(ids):
    x = m.emb(ids)
    for l in range(LAYER + 1): x = m.blocks[l](x)
    return x

def hL_given_resid(xl, ids_full):
    """continue from layer-LAYER residual xl for the first TAU positions, embedding-only for appended tokens is NOT
    valid (later tokens need their own lower layers), so instead we run the full model with an injection."""
    raise NotImplementedError

def run(ids, delta=None):
    """full forward; delta [B,T,D] added after block LAYER (zeros where no injection)."""
    inj = None if delta is None else (LAYER, delta)
    return m(ids, inj)

def append(ids, phrase):
    return torch.cat([ids, torch.tensor(phrase).expand(ids.shape[0], -1)], 1)

def mask_delta(ids_len, vec):
    """delta that adds vec at every valid context source position (positions in src), zero elsewhere."""
    d = torch.zeros(B, ids_len, D); d[:, src] = vec; return d

# ---------- backward: SJ phrase vector ----------
def sj_vector(phrase):
    """v(s) = mean_c mean_{t in src} sum_i d z_{s_i, TAU+i-2} / d h_{LAYER,t}"""
    ids = append(ctx, phrase); n = len(phrase)
    delta = torch.zeros(B, ids.shape[1], D, requires_grad=True)
    hL = run(ids, delta)
    obj = sum(hL[:, TAU + i - 1] @ WU[phrase[i]] for i in range(n)).sum()   # position TAU+i-1 (0-indexed) predicts s_i
    g, = torch.autograd.grad(obj, delta)
    return g[:, src].mean(1).mean(0)            # mean over sources, mean over contexts

# ---------- prefix-conditioned Jacobian via forward mode ----------
def jvp_prefix(prefix, vec):
    """J(p) vec = mean_c mean_t d hL_end(c.p) / d h_{LAYER,t} . vec  (all sources perturbed at once)."""
    ids = append(ctx, prefix) if prefix else ctx
    dlt = mask_delta(ids.shape[1], vec)
    f = lambda e: run(ids, e * dlt)[:, -1]
    _, tang = torch.func.jvp(f, (torch.tensor(0.0),), (torch.tensor(1.0),))
    return tang.mean(0) / NS

def J_prefix_full(prefix):
    """exact J(p) as a d x d matrix by d backward passes (only for the toy)."""
    ids = append(ctx, prefix) if prefix else ctx
    rows = []
    for j in range(D):
        delta = torch.zeros(B, ids.shape[1], D, requires_grad=True)
        hL = run(ids, delta)
        g, = torch.autograd.grad(hL[:, -1, j].sum(), delta)
        rows.append(g[:, src].mean(1).mean(0))
    return torch.stack(rows)                    # [j, k] = d hL_end[j] / d h[k]

def jlens_matrix():
    """repo estimator: cotangent at every valid target t' (same mask), gradient at source t, mean over sources,
    which equals mean_t sum_{t'>=t} dhL_{t'}/dh_t because of causality."""
    rows = []
    for j in range(D):
        delta = torch.zeros(B, TAU, D, requires_grad=True)
        hL = run(ctx, delta)
        g, = torch.autograd.grad(hL[:, src, j].sum(), delta)
        rows.append(g[:, src].mean(1).mean(0))
    return torch.stack(rows)

h = torch.randn(D) * 3                         # a probe activation
ok = lambda name, a, b, tol=1e-9: print(f"[{'PASS' if (a-b).abs().max() < tol else 'FAIL'}] {name}: max|diff|={(a-b).abs().max():.2e}")

phrase = [7, 19, 3]
v = sj_vector(phrase)

# (1) linearity decomposition: v(s) = sum_i J(s_<i)^T W_U[s_i]
vsum = sum(J_prefix_full(phrase[:i]).T @ WU[phrase[i]] for i in range(len(phrase)))
ok("(1) v(s) = sum_i J(s_<i)^T W_U[s_i]", v, vsum)

# (2) forward = backward: <v(s),h> = sum_i W_U[s_i] . J(s_<i) h   via JVP
fwd = sum(WU[phrase[i]] @ jvp_prefix(phrase[:i], h) for i in range(len(phrase)))
ok("(2) <v(s),h> = sum_i W_U[s_i].JVP(s_<i,h)", v @ h, fwd)

# (3) simultaneous perturbation = sum of per-source JVPs
per_src = torch.zeros(D)
ids = ctx
for t in src:
    d = torch.zeros(B, TAU, D); d[:, t] = h
    f = lambda e: run(ids, e * d)[:, -1]
    per_src += torch.func.jvp(f, (torch.tensor(0.0),), (torch.tensor(1.0),))[1].mean(0)
ok("(3) all-source JVP = sum of per-source JVPs", jvp_prefix([], h) * NS, per_src)

# (4) empty prefix, change summed over all masked targets = |T| * J_lens h  (repo estimator)
JL = jlens_matrix()
f = lambda e: run(ctx, e * mask_delta(TAU, h))[:, src].sum(1)
tot = torch.func.jvp(f, (torch.tensor(0.0),), (torch.tensor(1.0),))[1].mean(0)
ok("(4) sum_{t' in T} dhL_{t'} = |T| J_lens h", tot, NS * (JL @ h))

# (5) n=1 reduction: v(w) equals the LAST-TARGET Jacobian transpose row, not the J-lens row (different target weighting)
w = 7
v1 = sj_vector([w]); Jlast = J_prefix_full([])
ok("(5a) v(w) = J(empty)^T W_U[w]  (target = last position only)", v1, Jlast.T @ WU[w])
dl = JL.T @ WU[w]
print(f"      (5b) cosine(v(w), J-lens vector d(w)) = {F.cosine_similarity(v1, dl, 0):.3f}  (not an identity; H1 tests this empirically)")

# (6) order: v(ab) != v(ba)
vab, vba = sj_vector([7, 19]), sj_vector([19, 7])
print(f"      (6) cosine(v(ab), v(ba)) = {F.cosine_similarity(vab, vba, 0):.3f}; sum-of-tokens cosine to v(ab) = {F.cosine_similarity(sj_vector([7])+sj_vector([19]), vab, 0):.3f}")

# (7) log-prob gradient = logit gradient with the p-weighted mean unembedding subtracted (per context, per step)
def sj_vector_logprob(phrase):
    ids = append(ctx, phrase); delta = torch.zeros(B, ids.shape[1], D, requires_grad=True)
    hL = run(ids, delta)
    obj = sum(F.log_softmax(hL[:, TAU + i - 1] @ WU.T, -1)[:, phrase[i]] for i in range(len(phrase))).sum()
    return torch.autograd.grad(obj, delta)[0][:, src].mean(1).mean(0)
def sj_vector_centered(phrase):
    ids = append(ctx, phrase); delta = torch.zeros(B, ids.shape[1], D, requires_grad=True)
    hL = run(ids, delta); obj = 0
    for i in range(len(phrase)):
        p = (hL[:, TAU + i - 1] @ WU.T).softmax(-1).detach()          # model's own next-token dist, held fixed
        obj = obj + (hL[:, TAU + i - 1] * (WU[phrase[i]] - p @ WU)).sum()  # (W_U[s_i] - E_p W_U) . hL, per context
    return torch.autograd.grad(obj, delta)[0][:, src].mean(1).mean(0)
ok("(7) grad log p(s) = grad of (W_U[s_i] - E_p W_U).hL, p held fixed", sj_vector_logprob(phrase), sj_vector_centered(phrase))

# (8) null z-score under isotropic Gaussian nulls = sqrt(d) * cosine(v,h)   (i.e. the magnitude-free score / (|h|/sqrt d))
M = 200000; g = torch.randn(M, D) * (h.norm() / math.sqrt(D))
z = (v @ h) / (g @ v).std()
print(f"      (8) null z = {z:.4f}; sqrt(d)*cos(v,h) = {math.sqrt(D) * F.cosine_similarity(v, h, 0):.4f}  (MC, m={M})")

# (9) matched-filter view: under an isotropic null the magnitude-free score S=<v,h>/|v| has the same law for every phrase length
for n in (1, 2, 3):
    vn = sj_vector(phrase[:n]); S = (g @ vn) / vn.norm()
    print(f"      (9) n={n}: std of S under null = {S.std():.4f}  (expected |h|/sqrt(d) = {h.norm()/math.sqrt(D):.4f})")

# (10) finite differences vs exact JVP: fp64 and bf16, relative error as a function of eps (relative to source norm)
exact = jvp_prefix([], h)
hnorm = resid_at_layer(ctx)[:, src].norm(dim=-1).mean().item()
print(f"      (10) mean source residual norm at layer {LAYER}: {hnorm:.2f}; |h|={h.norm():.2f}")
for dt in (torch.float64, torch.bfloat16):
    mm = m.to(dt); hh = h.to(dt); ids = ctx
    with torch.no_grad():
        clean = mm(ids)[:, -1].double()
        for rho in (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 1.0):
            eps = rho * hnorm / h.norm().item()
            d = torch.zeros(B, TAU, D, dtype=dt); d[:, src] = eps * hh
            pert = mm(ids, (LAYER, d))[:, -1].double(); pertm = mm(ids, (LAYER, -d))[:, -1].double()
            fd1 = ((pert - clean) / eps).mean(0) / NS; fd2 = ((pert - pertm) / (2 * eps)).mean(0) / NS
            e1 = (fd1 - exact).norm() / exact.norm(); e2 = (fd2 - exact).norm() / exact.norm()
            print(f"           {str(dt):15s} rho={rho:<6g} fwd-diff rel err={e1:.3e}  central rel err={e2:.3e}  spearman(WU fd, WU exact)={torch.corrcoef(torch.stack([(WU.double()@fd2).argsort().argsort().double(), (WU.double()@exact).argsort().argsort().double()]))[0,1]:.3f}")
    m.to(torch.float64)

# ---------- Formal framework additions ----------
m.to(torch.float64)
JL = jlens_matrix()   # recompute: the bf16 sweep rounded the weights
# (11) Prop 3: J_lens = sum_g (1 - g/|T|) G_g   (G_g = mean over pairs at gap g of dh_{t+g}/dh_t)
def gap_jacobian(g):
    srcs = [t for t in src.tolist() if t + g in src.tolist()]
    acc = torch.zeros(D, D)
    for t in srcs:
        for j in range(D):
            delta = torch.zeros(B, TAU, D, requires_grad=True)
            hL = run(ctx, delta)
            gr, = torch.autograd.grad(hL[:, t + g, j].sum(), delta)
            acc[j] += gr[:, t].mean(0)
    return acc / len(srcs)
JL_rebuilt = sum((1 - g / NS) * gap_jacobian(g) for g in range(NS))
ok("(11) Prop 3: J_lens = sum_g (1-g/|T|) G_g", JL, JL_rebuilt, 1e-9)

# (12) Prop 7: v(ab) - v(a) - v(b) = (J(a) - J(empty))^T W_U[b]
a, b = 7, 19
lhs = sj_vector([a, b]) - sj_vector([a]) - sj_vector([b])
rhs = (J_prefix_full([a]) - J_prefix_full([])).T @ WU[b]
ok("(12) Prop 7: v(ab)-v(a)-v(b) = Delta_a^T W_U[b]", lhs, rhs, 1e-9)

# (13) Prop 6: exact extension condition  a > A_n (sqrt(1 + q/Q_n) - 1)  <=>  z_{n+1} > z_n, random instances
torch.manual_seed(1); bad = 0
for _ in range(2000):
    Vn, vn1, x = torch.randn(D), torch.randn(D), torch.randn(D)
    A, Q = Vn @ x, Vn @ Vn
    if A <= 0: continue
    a_, q_ = vn1 @ x, (Vn + vn1) @ (Vn + vn1) - Q
    zn, zn1 = A / Q.sqrt(), (A + a_) / (Q + q_).sqrt()
    cond = a_ > A * ((1 + q_ / Q).sqrt() - 1)
    bad += int(cond != (zn1 > zn))
print(f"[{'PASS' if bad == 0 else 'FAIL'}] (13) Prop 6 stopping condition matches z_(n+1) > z_n on 2000 random instances: mismatches={bad}")
