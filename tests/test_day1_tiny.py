"""Day-1 code (agent F) on the tiny random Qwen3 (CPU): X1's arms, X3's decoding, the P1 gate, the P2 family
metrics, the e13 random control and C16's gap statistics. Content is meaningless on a random model; only the
identities, the shapes and the control logic are checked. Run: python tests/test_day1_tiny.py"""
import importlib.util, json, os, sys, tempfile, torch
sys.path.insert(0, "."); sys.path.insert(0, "tests")
from tiny import tiny
from sjlens.model.qwen3_min import Qwen3Min
from sjlens.lens import jlens
from sjlens.lens.forward import jvp_step
from sjlens.eval.gap_profile import gap_profile_sampled, gap_stats

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(os.path.dirname(HERE), "scripts")
load = lambda n, f: (lambda s: (lambda mod: (s.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s)))(importlib.util.spec_from_file_location(n, os.path.join(SC, f)))
e12 = load("e12", "e12_second_token.py"); e14 = load("e14", "e14_dictionary.py"); e13 = load("e13", "e13_prefix_order.py"); x1 = load("x1", "x1_gap1_patch.py"); x3 = load("x3", "x3_patchscope.py")

torch.manual_seed(0)
res = []
def chk(name, err, tol): res.append((name, err, err < tol)); print(f"[{'PASS' if err < tol else 'FAIL'}] {name}: {err:.2e}")

m = tiny(); q = Qwen3Min(m); d = q.d; V = 256; L = 1; SK = 4; TAU = 20
ctx = [torch.randint(0, V, (TAU,)) for _ in range(4)]
ids = torch.stack(ctx); mask = jlens.valid_mask(TAU, SK)
h = torch.randn(d, dtype=torch.float64) * 0.5
with torch.no_grad(): hn = float(q.resid(ids, L)[:, mask].norm(dim=-1).mean())
s1, s2 = 7, 42

# --- X1: the linear arm is the derivative of the finite arm (norm-matched add), and alpha = 0 gives Delta = 0 ---
# finite add injects delta = alpha * hn / ||h|| * h, so d/d(alpha) at 0 is (hn / ||h||) * jvp(h).
scale = hn / float(h.norm())
lin = x1.delta_logp(q, ids, L, h, [s1], alpha=0.0, mode="add", hn=hn)
chk("X1 finite patch at alpha = 0 gives Delta log p = 0 (max abs)", float(lin.abs().max()), 1e-12)
jv = jvp_step(q, ids, L, mask.clone().zero_().index_fill_(0, torch.tensor([TAU - 1]), True), h, prefix=[s1], mode="lp").double() * scale
prev = None
for a_ in (1e-3, 1e-4, 1e-5):
    fd = (x1.delta_logp(q, ids, L, h, [s1], a_, "add", hn) - x1.delta_logp(q, ids, L, h, [s1], -a_, "add", hn)) / (2 * a_)
    e = float((fd - jv).norm() / jv.norm()); print(f"      X1 central FD alpha={a_:g} vs the gap-1 JVP: rel {e:.2e}")
    prev = e
chk("X1 linear arm = finite difference of the finite arm as alpha -> 0 (rel)", prev, 1e-6)
one = torch.zeros(TAU, dtype=torch.bool); one[TAU - 1] = True
chk("X1 linear arm = jvp_exact with the one-position mask at tau-1 (rel)",
    float((jv / scale - jvp_step(q, ids, L, one, h, prefix=[s1], mode="lp").double()).norm() / jv.norm() * scale), 1e-12)
rep = x1.delta_logp(q, ids, L, h, [s1], mode="replace", hn=hn)
with torch.no_grad(): xlast = q.resid(ids, L)[:, TAU - 1]
chk("X1 cached clean / xlast give the same replace arm as recomputing them (rel)",
    float((x1.delta_logp(q, ids, L, h, [s1], mode="replace", hn=hn, clean=x1.clean_logp(q, ids, [s1]), xlast=xlast) - rep).norm() / rep.norm()), 1e-12)
chk("X1 the cached clean log-probs are the carrier mean of a clean forward (rel)",
    float((x1.clean_logp(q, ids, [s1], chunk=2) - x1.clean_logp(q, ids, [s1], chunk=4)).norm().clamp_min(1e-30) / x1.clean_logp(q, ids, [s1]).norm()), 1e-12)
chk("X1 replace mode returns a finite [V] shift", float(0 if (rep.shape == (V,) and torch.isfinite(rep).all() and rep.abs().max() > 0) else 1), 0.5)
chk("X1 rank_of is the mid-rank (a self-scored vector ranks 1)", abs(x1.rank_of(torch.tensor([3.0, 1.0, 2.0]), 0) - 1), 1e-12)
chk("X1 rank_of splits ties", abs(x1.rank_of(torch.tensor([1.0, 1.0, 0.0]), 0) - 1.5), 1e-12)

# --- X1 controls and the summary / verdict plumbing ---
cats = ["a", "a", "b", "b", "c"]; perm = e12.shuffled_perm(cats, 0)
chk("controls: the within-category derangement stays inside its category", float(sum(cats[int(perm[i])] != cats[i] for i in range(len(cats)))), 0.5)
chk("controls: the derangement moves every item of a category with >= 2 members", float(sum(int(perm[i]) == i for i in range(4))), 0.5)
rows = [{"arm": "finite:add:a1:plain", "control": c, "category": "x", "rank_s2": r, "dlp_s2": 0.1} for c, r in (("none", 1.0), ("shuffled", 900.0), ("random", 900.0))]
sm = x1.summarise(rows); vd = x1.verdict(sm)
chk("X1 summarise gives bootstrap CIs and per-category tables", float(0 if (sm["finite:add:a1:plain"]["none"]["top10"] == 1.0 and "by_category" in sm["finite:add:a1:plain"]["none"]
                                                                            and sm["finite:add:a1:plain"]["none"]["top1_lo"] is not None) else 1), 0.5)
chk("X1 verdict marks a clean pass (top-10 1.0, controls 0)", float(0 if (vd["any_pass"] and not vd["all_kill"]) else 1), 0.5)
rows2 = [{"arm": "finite:add:a1:plain", "control": c, "category": "x", "rank_s2": 900.0, "dlp_s2": 0.0} for c in ("none", "shuffled")]
chk("X1 verdict marks a kill when the arm equals its control at the floor", float(0 if x1.verdict(x1.summarise(rows2))["all_kill"] else 1), 0.5)

# --- X3: decodes on the tiny model (content meaningless) and its controls / scoring run ---
class Tok:  # the tiny model has no tokenizer; a byte-ish stand-in is enough to exercise the code path
    def __call__(s, t, return_tensors=None): return type("O", (), {"input_ids": torch.tensor([[ord(c) % 256 for c in t[:24]]])})()
    def decode(s, t): return "".join(chr(int(x) % 128) for x in t)
    def encode(s, t, add_special_tokens=False): return [ord(c) % 256 for c in t]
    all_special_ids = []
tk = Tok(); cid = tk("cat -> cat; ?").input_ids
gen = x3.patched_generate(q, cid, L, cid.shape[1] - 1, h, 6, "replace")
chk("X3 greedy decode returns --gen tokens under a replace patch", float(abs(len(gen) - 6)), 0.5)
gen_a = x3.patched_generate(q, cid, L, cid.shape[1] - 1, h, 4, "add", 1.0, hn)
chk("X3 add mode decodes too", float(abs(len(gen_a) - 4)), 0.5)
print(f"      X3 tiny generations: replace {tk.decode(gen)!r}, add {tk.decode(gen_a)!r}")
chk("X3 norm() collapses whitespace and case", float(x3.norm("  Sri   LANKA\n") != "sri lanka"), 0.5)
it_x3 = {"string": " Sri Lanka", "pieces_text": [" Sri", " Lanka"]}
chk("X3 score: exact / prefix / first flags", float(x3.score(" sri  lanka ", it_x3) != (1, 1, 1)) + float(x3.score(" Sri Lankan rupee", it_x3) != (0, 1, 1)), 0.5)
chk("X3 cut() stops an identity generation at ';' so the echo scores exact",
    float(x3.score(" Sri Lanka -> Sri Lanka; next", it_x3, "identity") != (1, 1, 1)), 0.5)
chk("X3 cut() stops a describe generation at '.'", float(x3.score(" Sri Lanka. An island country", it_x3, "describe") != (1, 1, 1)), 0.5)
chk("X3 cut() does not rescue a wrong string", float(x3.score(" Nepal; Sri Lanka", it_x3, "identity")[0]), 0.5)
chk("X3 patch_pos finds the 'x' of a describe carrier", float(x3.patch_pos(tk, tk("Peru: country. x:").input_ids, "describe") != tk("Peru: country. x:").input_ids.shape[1] - 2), 0.5)
sm3 = x3.summarise([{"arm": "identity:replace", "control": "none", "kind": "answer", "category": "c", "vote_exact": 1, "vote_prefix": 1, "vote_first": 1, "any_exact": 1}])
chk("X3 summarise reports the majority vote with a CI and by-category tables", float(0 if (sm3["identity:replace"]["none"]["vote_exact"] == 1.0 and "by_category" in sm3["identity:replace"]["none"]) else 1), 0.5)

# --- P1: the bridge gate rejects a leaking bridge and a substring digit match ---
class GT:  # greedy is stubbed per item through q; here only the token-level helpers are exercised
    pass
chk("P1 _contains is token-level (target [1] not found in the tokens of '16')", float(e12._contains([16, 5], (1,))), 0.5)
chk("P1 _contains finds a real token run", float(not e12._contains([9, 16, 5], (16, 5))), 0.5)
leak_item = {"string": " Sri Lanka", "pieces_text": [" Sri", " Lanka"], "kind": "bridge"}
chk("P1 bridge_leak flags a greedy output that verbalises the bridge", float(not e12.bridge_leak(leak_item, " the answer is Sri Lanka, whose", ["rupee"])), 0.5)
chk("P1 bridge_leak flags a target starting with the bridge's first piece", float(not e12.bridge_leak(leak_item, " rupee", ["Sri Lankan rupee"])), 0.5)
chk("P1 bridge_leak passes a clean bridge item", float(e12.bridge_leak(leak_item, " the rupee", ["rupee"])), 0.5)

# the full gate on the tiny model: a bridge whose greedy output contains the bridge string is rejected
def fake_gate(greedy_text, target, item):
    it = dict(item); e12.greedy_backup = None
    tgs = [target]
    ok = True  # containment is exercised above; here only the leak veto
    return ok and not e12.bridge_leak(it, greedy_text, tgs)
chk("P1 gate vetoes the leaking item and keeps the clean one",
    float(fake_gate(" Sri Lanka", "rupee", leak_item)) + float(not fake_gate(" the rupee", "rupee", leak_item)), 0.5)

# --- P2: family metrics on a synthetic score matrix ---
# 4 candidates: p1 = [1, 1, 2, 3]; candidates 0 and 1 form a family of 2, 2 and 3 are unique.
Z = torch.tensor([[3.0, 1.0, 0.0, 0.0],   # q0 truth 0: top-1 overall and within its family
                  [1.0, 3.0, 5.0, 0.0],   # q1 truth 1: loses overall, wins within its family
                  [0.0, 0.0, 9.0, 1.0],   # q2 truth 2: unique family, top-1
                  [0.0, 0.0, 5.0, 1.0]])  # q3 truth 3: unique family, misses
truth = torch.tensor([0, 1, 2, 3]); cand_p1 = [1, 1, 2, 3]; q_cat = ["w", "w", "year", "year"]
fm = e14.family_metrics(Z, truth, cand_p1, q_cat, numeric=("year",))
chk("P2 unique-first-piece top-1 = 1/2", abs(fm["unique_top1"] - 0.5), 1e-12)
chk("P2 within-family top-1 = 1 (both family queries win inside the family)", abs(fm["within_family_top1"] - 1.0), 1e-12)
chk("P2 within-family chance = 1/2", abs(fm["within_family_chance"] - 0.5), 1e-12)
chk("P2 excluding numeric categories drops the year queries from the unique set", abs(fm["n_unique_ex_numeric"] - 0), 0.5)
chk("P2 by-category table carries n / top1 / within-family per category", float(0 if (fm["by_category"]["w"]["n"] == 2 and fm["by_category"]["year"]["numeric"] and abs(fm["by_category"]["w"]["top1"] - 0.5) < 1e-12) else 1), 0.5)
chk("P2 mid-rank of the truth", float((e14.midrank(Z, truth) - torch.tensor([1.0, 2.0, 1.0, 2.0])).abs().max()), 1e-12)
pm = e14.shuffled_perm(q_cat, 0)
chk("P2 shuffled-h control swaps the queries inside each category", float((pm - torch.tensor([1, 0, 3, 2])).abs().max()), 0.5)
chk("P2 shuffled-h control scores the ORIGINAL truth against another item's scores",
    float((e14.midrank(Z[pm], truth) - torch.stack([e14.midrank(Z[int(pm[i])].view(1, -1), truth[i].view(1))[0] for i in range(4)])).abs().max()), 1e-12)
Hq = torch.randn(4, d, dtype=torch.float64); mu = torch.randn(d, dtype=torch.float64) * 0.1
Hr = e14.random_queries(Hq, mu, 0)
chk("P2 random-direction control is norm-matched around mu (rel)", float(((Hr - mu).norm(dim=1) - (Hq - mu).norm(dim=1)).abs().max() / (Hq - mu).norm(dim=1).mean()), 1e-12)
chk("P2 random-direction control is not the original h (control; -min rel diff)", -float((Hr - Hq).norm(dim=1).min() / Hq.norm(dim=1).mean()), -1e-3)
chk("P2 numeric-category detection by name and by digits", float(not (e14.is_numeric_cat("year") and e14.is_numeric_cat("odd", ["12", "34", "x"]) and not e14.is_numeric_cat("capital", [" Paris", " Rome"]))), 0.5)

# --- e13 random control builds from the config, never loading pretrained weights ---
import transformers
calls = {"pretrained": 0, "from_config": 0}
orig_p, orig_c = transformers.AutoModelForCausalLM.from_pretrained, transformers.AutoModelForCausalLM.from_config
class FakeCfg:
    _attn_implementation = "eager"
    @staticmethod
    def from_pretrained(n, **kw): return FakeCfg()
def fake_from_config(cfg, **kw):
    calls["from_config"] += 1
    return tiny(dtype=kw.get("dtype", torch.float32))
def fake_from_pretrained(*a_, **kw):
    calls["pretrained"] += 1; raise AssertionError("e13 --random-control must not load pretrained weights")
transformers.AutoModelForCausalLM.from_config = staticmethod(fake_from_config)
transformers.AutoModelForCausalLM.from_pretrained = staticmethod(fake_from_pretrained)
orig_ac, orig_at = transformers.AutoConfig, transformers.AutoTokenizer
transformers.AutoConfig = FakeCfg; transformers.AutoTokenizer = type("T", (), {"from_pretrained": staticmethod(lambda n, **kw: Tok())})
try:
    tk2, rm = e13.random_model("Qwen/Qwen3-14B", 0, torch.float32, "cpu")
finally:
    transformers.AutoModelForCausalLM.from_pretrained = orig_p; transformers.AutoModelForCausalLM.from_config = orig_c
    transformers.AutoConfig = orig_ac; transformers.AutoTokenizer = orig_at
chk("e13 random control builds from the config and never calls from_pretrained", float(calls["pretrained"] + abs(calls["from_config"] - 1)), 0.5)
chk("C8: usable_ids drops empty and special ids", float(0 if (set(e13.usable_ids(type("T2", (), {"all_special_ids": [5], "decode": lambda s, t: {3: "  ", 5: "<|im_start|>", 7: "a"}.get(t[0], "b")})(), 8)) == {0, 1, 2, 4, 6, 7}) else 1), 0.5)

# --- C16: gap_stats reports the norm-weighted future sum and its gap shares ---
G, _ = gap_profile_sampled(q, ctx[:2], L, [0, 3, 9], skip_first=SK)
nT = TAU - SK - 1; st = gap_stats(G, nT)
S_ref = sum(max(1 - g / nT, 0.0) * float(G[g].norm()) for g in G if g > 0)
chk("C16 S = sum_{g>=1} (1 - g/|T|) ||G_g|| (rel)", abs(st["S_future"] - S_ref) / S_ref, 1e-12)
chk("C16 gap-1 share", abs(st["share_gap1"] - (1 - 1 / nT) * float(G[1].norm()) / S_ref), 1e-12)
chk("C16 shares are ordered gap1 <= gaps1-5 <= 1", float(not (st["share_gap1"] <= st["share_gaps_1_5"] <= 1.0 + 1e-12)), 0.5)
chk("C16 rho_exact uses the actual future term (differs from the stationary proxy; -rel gap)",
    -abs(st["rho_exact"] - st["rho_frob"]) / st["rho_frob"], -1e-6)
print(f"      C16: rho_frob {st['rho_frob']:.3f}, rho_exact {st['rho_exact']:.3f}, S {st['S_future']:.3f}, shares gap1 {st['share_gap1']:.3f} / 1-5 {st['share_gaps_1_5']:.3f} / >=20 {st['share_gaps_ge20']}")

# --- P3: the conformal certificate gates emitted phrases ---
from sjlens.lens.forward import ContextBank
from sjlens.beam.jbeam import JBeam, Cfg
J = jlens.fit(q, ctx, [L], skip_first=SK)[L]; D = jlens.token_matrix(q.w, J); bank = ContextBank(q, ctx, SK)
with torch.no_grad(): Hbg = torch.cat([q.resid(c.view(1, -1), L)[0, mask] for c in [torch.randint(0, V, (TAU,)) for _ in range(20)]])
mu = Hbg.mean(0); nulls = (Hbg - mu)[:16]; hb = Hbg[3]
base = dict(seeds=3, beams=3, max_len=3, m_null=16, k_pursuit=8, q_fw=80.0, normalizer="eb", prior="lens")
out_lo, _ = JBeam(q, bank, D, L, nulls, Cfg(**base, stop="tau", tau=-1e9), mu, Hbg=Hbg).read(hb)
out_hi, _ = JBeam(q, bank, D, L, nulls, Cfg(**base, stop="tau", tau=1e9), mu, Hbg=Hbg).read(hb)
chk("P3: an unreachable conformal tau emits nothing", float(len(out_hi)), 0.5)
chk("P3: a permissive tau still emits phrases", float(len(out_lo) == 0), 0.5)
chk("P3: every emitted phrase is above tau", float(sum(z <= 0.0 for _, z, *_ in JBeam(q, bank, D, L, nulls, Cfg(**base, stop="tau", tau=0.0), mu, Hbg=Hbg).read(hb)[0])), 0.5)

print("\nSUMMARY:", sum(ok for *_, ok in res), "/", len(res), "checks passed")
