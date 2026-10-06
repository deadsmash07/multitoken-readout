"""Wave-2 code (agent G) on the tiny random Qwen3 (CPU): the C1 derangements, X1 v2 (target layers, absolute / clean
ranks, paired statistics, crosscat), X3 v2 (carriers, stop / first-piece rules, first-hop position), the single-item
gate, the probe's split discipline, the item-set modes. Content is meaningless on a random model; only identities,
shapes and control logic are checked. Run: python tests/test_wave2_tiny.py"""
import importlib.util, os, re, sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "tests")
from tiny import tiny
from sjlens.model.qwen3_min import Qwen3Min, logits as true_logits
from sjlens.lens import jlens

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(os.path.dirname(HERE), "scripts")
load = lambda n, f: (lambda s: (lambda mod: (s.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s)))(importlib.util.spec_from_file_location(n, os.path.join(SC, f)))
e12 = load("e12", "e12_second_token.py"); x1 = load("x1", "x1_gap1_patch.py"); x3 = load("x3", "x3_patchscope.py"); x4 = load("x4", "x4_probe.py"); mk = load("mk", "make_h2a_items.py")

torch.manual_seed(0)
res = []
def chk(name, err, tol): res.append((name, err, err < tol)); print(f"[{'PASS' if err < tol else 'FAIL'}] {name}: {err:.2e}")

# --- C1: within-category derangement never pairs an item with itself or with the same string; singletons are -1 ---
cats = ["a"] * 6 + ["b"] * 4 + ["c"]; strs = ["x", "x", "y", "z", "w", "v", "p", "p", "q", "r", "s"]
perm = e12.shuffled_perm(cats, 0, strs)
chk("C1 derangement: no item is its own partner", float(sum(int(perm[i]) == i for i in range(len(cats)))), 0.5)
chk("C1 derangement: partners share the category", float(sum(int(perm[i]) >= 0 and cats[int(perm[i])] != cats[i] for i in range(len(cats)))), 0.5)
chk("C1 derangement: partners never share the string", float(sum(int(perm[i]) >= 0 and strs[int(perm[i])] == strs[i] for i in range(len(cats)))), 0.5)
chk("C1 derangement: the singleton category is -1", float(int(perm[10]) != -1), 0.5)
chk("C1 derangement: every item of a multi-member category with a valid partner gets one", float(sum(int(perm[i]) < 0 for i in range(10))), 0.5)
p2 = e12.shuffled_perm(["a", "a"], 0, ["same", "same"])
chk("C1 derangement: two items with the same string get no partner (-1, -1)", float((p2 >= 0).sum()), 0.5)
chk("C1 derangement is a bijection on the assigned items", float(len({int(perm[i]) for i in range(10)}) != 10), 0.5)
chk("C1 derangement is seeded (same seed, same pairs)", float((e12.shuffled_perm(cats, 0, strs) != perm).any()), 0.5)
chk("C1 derangement per-category seeding: category b's pairs do not depend on category a being present",
    float((e12.shuffled_perm(cats[6:], 0, strs[6:]) != (perm[6:] - 6).clamp_min(-1)).any()), 0.5)
old_style = e12.shuffled_perm(["a", "a", "b", "b", "c"], 0)  # the Day-1 call signature still works
chk("C1 old signature: within-category derangement, singleton -1", float(int(old_style[4]) != -1 or int(old_style[0]) != 1 or int(old_style[2]) != 3), 0.5)
xp = e12.cross_perm(cats, 0, strs)
chk("crosscat: partner is from a DIFFERENT category", float(sum(cats[int(xp[i])] == cats[i] for i in range(len(cats)) if int(xp[i]) >= 0)), 0.5)
chk("crosscat: a category holding > half the items leaves exactly its excess unpaired (6 a vs 5 others -> one -1)", float((xp < 0).sum() != 1), 0.5)
xb = e12.cross_perm(["a"] * 4 + ["b"] * 4 + ["c"] * 3, 0)
chk("crosscat: every item has a partner when no category holds more than half", float((xb < 0).sum()), 0.5)
chk("crosscat: partners never share the string", float(sum(strs[int(xp[i])] == strs[i] for i in range(len(cats)) if int(xp[i]) >= 0)), 0.5)
chk("crosscat: a single category gives -1 everywhere", float((e12.cross_perm(["a", "a", "a"], 0) >= 0).sum()), 0.5)

# --- X1 v2 on the tiny model ---
m = tiny(); q = Qwen3Min(m); d = q.d; V = 256; L = 2; SK = 4; TAU = 20
ctx = [torch.randint(0, V, (TAU,)) for _ in range(4)]; ids = torch.stack(ctx); mask = jlens.valid_mask(TAU, SK)
h = torch.randn(d, dtype=torch.float64) * 0.5; s1, s2 = 7, 42
with torch.no_grad(): hn = {l: float(q.resid(ids, l)[:, mask].norm(dim=-1).mean()) for l in (0, L)}; xlast = {l: q.resid(ids, l)[:, TAU - 1] for l in (0, L)}
clean = x1.clean_logp(q, ids, [s1])
rep_old = x1.delta_logp(q, ids, L, h, [s1], mode="replace", hn=hn[L], clean=clean, xlast=xlast[L])
rep_same = x1.delta_logp(q, ids, L, h, [s1], mode="replace", hn=hn[L], clean=clean, xlast=xlast[L], target_layer=L)
chk("X1 v2: target_layer = source layer is bit-identical to the Day-1 path", float((rep_old - rep_same).abs().max()), 0.0 + 1e-300)
rep_t0, pat_t0 = x1.delta_logp(q, ids, L, h, [s1], mode="replace", hn=hn[0], clean=clean, xlast=xlast[0], target_layer=0, return_patched=True)
chk("X1 v2: an earlier target layer gives a different (finite) readout", -float((rep_t0 - rep_same).norm()), -1e-6)
chk("X1 v2: return_patched gives patched = Delta + clean (abs rank consistency)", float((pat_t0 - (rep_t0 + clean)).abs().max()), 1e-9)
chk("X1 v2: the patched mean log-prob is a distribution (logsumexp 0)", float(pat_t0.logsumexp(0).abs()) if False else float((pat_t0.exp().sum() - 1).abs()) , 1e-2)  # mean of log-probs over carriers is not exactly normalised; loose
# replace_nm with hn = ||h|| equals replace; with another hn it rescales h
nm = x1.delta_logp(q, ids, L, h, [s1], mode="replace_nm", hn=float(h.norm()), clean=clean, xlast=xlast[L])
chk("X1 v2: replace_nm with hn = ||h|| equals replace", float((nm - rep_old).abs().max()), 1e-9)
nm2 = x1.delta_logp(q, ids, L, h, [s1], mode="replace_nm", hn=2 * float(h.norm()), clean=clean, xlast=xlast[L])
nm2b = x1.delta_logp(q, ids, L, 2 * h, [s1], mode="replace", hn=hn[L], clean=clean, xlast=xlast[L])
chk("X1 v2: replace_nm rescales h to hn (equals replace of the rescaled h)", float((nm2 - nm2b).abs().max()), 1e-9)
# clean rank is the same object for every arm; abs rank uses patched
r_clean = x1.rank_of(clean, s2)
u2 = torch.randn(d, dtype=torch.float64)
_, pat_u2 = x1.delta_logp(q, ids, L, u2, [s1], mode="replace", hn=hn[L], clean=clean, xlast=xlast[L], return_patched=True)
chk("X1 v2: clean rank does not depend on the patched direction", abs(x1.rank_of(clean, s2) - r_clean), 1e-12)
chk("X1 v2: abs rank of a random direction differs from the clean rank for some token", -float(sum(x1.rank_of(pat_u2, w) != x1.rank_of(clean, w) for w in range(0, V, 8))), -0.5)
# crosscat control runs through delta_logp with a partner's h (just a different vector)
xc = x1.delta_logp(q, ids, L, u2, [s1], mode="replace", hn=hn[L], clean=clean, xlast=xlast[L])
chk("X1 v2: crosscat control (partner h) runs and is finite", float(0 if torch.isfinite(xc).all() else 1), 0.5)
# arm keys: Day-1 keys unchanged, target / piece suffixes
chk("X1 v2: same-layer arm keys are the Day-1 keys", float(x1.arm_name("replace", None, "plain", "same") != "finite:replace:rep:plain" or x1.arm_name("add", 1.0, "plain", "same") != "finite:add:a1:plain"), 0.5)
chk("X1 v2: target / piece arm keys", float(x1.arm_name("replace_nm", None, "plain", 8) != "finite:replace_nm:rep:plain:t8" or x1.arm_name("replace", None, "plain", "same", 3) != "finite:replace:rep:plain:p3"), 0.5)
chk("X1 v2: parse_targets accepts same / -1 / ints", float(x1.parse_targets("same,4,-1,8") != ["same", 4, "same", 8]), 0.5)
# summary: paired statistics and the secondary line on synthetic rows
rows = []
for i in range(40):
    real = 1.0 if i < 24 else 500.0; sh = 1.0 if i < 8 else 500.0; rd = 500.0
    for c, r in (("none", real), ("shuffled", sh), ("random", rd), ("crosscat", sh)):
        if c == "shuffled" and i >= 38: continue  # two items without a valid same-category partner
        rows.append({"id": f"i{i}", "arm": "finite:replace:rep:plain", "control": c, "category": "k" if i % 2 else "j", "rank_s2": r, "dlp_s2": 0.1,
                     "rank_s2_abs": r + 1, "rank_s2_clean": 300.0, "logp_s2_patched": -3.0, "logp_s2_clean": -5.0})
sm = x1.summarise(rows); e = sm["finite:replace:rep:plain"]; pr = e["shuffled"]["paired"]
chk("X1 v2 summary: shuffled control has n = 38 (dropped items are absent)", float(e["shuffled"]["n"] != 38), 0.5)
chk("X1 v2 summary: paired diff over the 38 items with both rows = (24 - 8) / 38", abs(pr["top10_diff"] - 16 / 38), 1e-9)
chk("X1 v2 summary: McNemar b = 16, c = 0", float(pr["top10_mcnemar"]["b"] != 16 or pr["top10_mcnemar"]["c"] != 0), 0.5)
chk("X1 v2 summary: exact McNemar p for 16 vs 0 = 2 / 2^16", abs(pr["top10_mcnemar"]["p"] - 2 / 2 ** 16), 1e-12)
chk("X1 v2 summary: paired CI brackets the point estimate", float(not (pr["top10_diff_lo"] <= pr["top10_diff"] <= pr["top10_diff_hi"])), 0.5)
chk("X1 v2 summary: abs / clean rank blocks present", float("abs_top10" not in e["none"] or "clean_top10" not in e["none"] or abs(e["none"]["clean_top10"]) > 0), 0.5)
chk("X1 v2 summary: abs top-10 uses rank_s2_abs (24/40 at rank 2)", abs(e["none"]["abs_top10"] - 24 / 40), 1e-9)
vd = x1.verdict(sm); sec = vd["arms"]["finite:replace:rep:plain"]["secondary"]
chk("X1 v2 verdict: SECONDARY line passes (gap .42 >= .20, lower >= .10, real - random .6 >= .30)", float(not sec["pass"]), 0.5)
chk("X1 v2 verdict: pre-registered R1 does not pass (control .2 > .05)", float(vd["any_pass"]), 0.5)
chk("X1 v2 mcnemar: symmetric counts give p = 1", abs(x1.mcnemar([1, 0, 1, 0], [0, 1, 0, 1])["p"] - 1.0), 1e-12)

# --- X3 v2 ---
class Tok:  # byte-ish stand-in
    def __call__(s, t, return_tensors=None): return type("O", (), {"input_ids": torch.tensor([[ord(c) % 256 for c in t]])})()
    def decode(s, t): return "".join(chr(int(x) % 128) for x in t)
    def encode(s, t, add_special_tokens=False): return [ord(c) % 256 for c in t]
    all_special_ids = []
tk = Tok()
chk("X3 v2: identity2 / describe2 carriers have 8 fixed variants each", float(len(x3.IDENTITY2) != 8 or len(x3.DESCRIBE2) != 8), 0.5)
chk("X3 v2: identity2 exemplars are multi-word or 4-digit (no single-word exemplar)", float(any(len(x.split(" -> ")[0].split()) == 1 and not x.split(" -> ")[0].isdigit() and len(x.split(" -> ")[0]) < 8 for x in x3.IDENTITY2)), 0.5)
chk("X3 v2: describe2 patches the '?' slot before ' ->' (char-level stand-in: 4th from the end)", float(x3.patch_pos(tk, tk("a -> a is b. ? ->").input_ids, "describe2") != "a -> a is b. ? ->".index("?")), 0.5)
chk("X3 v2: Day-1 carriers and stops unchanged", float(x3.STOPS["identity"] != (";", "->", "\n") or x3.IDENTITY[0] != "cat -> cat; 1135 -> 1135; hello -> hello; ?"), 0.5)
it3 = {"string": " Sri Lanka", "pieces": [50, 60], "pieces_text": [" Sri", " Lanka"], "p1_chars": 3}
chk("X3 v2: describe2 cut at ' is ' scores the name exact", float(x3.score(" Sri Lanka is an island country", it3, "describe2") != (1, 1, 1)), 0.5)
chk("X3 v2: describe2 cut does not split ' island' at ' is'", float(x3.cut(" island is big", "describe2") != " island"), 0.5)
chk("X3 v2: describe2 cut at ' was '", float(x3.cut(" Tchaikovsky was a composer", "describe2") != " Tchaikovsky"), 0.5)
chk("C2: first-piece flag is TOKEN-level: first generated id must equal s1's id", float(x3.first_piece_hit([50, 99], it3) != 1 or x3.first_piece_hit([51, 60], it3) != 0), 0.5)
class WsTok(Tok):
    def decode(s, t): return " " if list(t) == [1] else "".join(chr(int(x) % 128) for x in t)
chk("C2: a leading pure-whitespace token is skipped", float(x3.first_piece_hit([1, 50], it3, WsTok()) != 1), 0.5)
chk("C2: text prefix no longer counts (old behaviour only without gen_ids)", float(x3.score(" Sri", it3, "identity", gen_ids=[51], tok=tk)[2] != 0 or x3.score(" Sri", it3, "identity")[2] != 1), 0.5)
rows3 = [{"arm": "identity2:replace", "control": "none", "id": f"i{i}", "kind": "answer", "category": "c", "p1_chars": 1 if i < 3 else 3, "vote_exact": 1, "vote_prefix": 1, "vote_first": 1 if i < 3 else 0, "any_exact": 1} for i in range(6)]
rows3 += [{**r, "control": "shuffled", "vote_exact": 0, "any_exact": 0} for r in rows3]
s3 = x3.summarise(rows3)["identity2:replace"]
chk("X3 v2 summary: vote_first restricted to p1_chars >= 2 (0/3) beside the raw rate (3/6)", abs(s3["none"]["vote_first_p1ge2"] - 0.0) + abs(s3["none"]["vote_first"] - 0.5), 1e-9)
chk("X3 v2 summary: paired real - shuffled exact diff = 1 with McNemar 6 / 0", float(s3["shuffled"]["paired"]["vote_exact_diff"] != 1.0 or s3["shuffled"]["paired"]["vote_exact_mcnemar"]["b"] != 6), 0.5)
g_nm = x3.patched_generate(q, tk("cat -> cat; ?").input_ids, L, 4, h, 3, "replace_nm", 1.0, float(h.norm()), 0)
g_rp = x3.patched_generate(q, tk("cat -> cat; ?").input_ids, L, 4, h, 3, "replace", 1.0, None, 0)
chk("X3 v2: replace_nm with hn = ||u|| decodes exactly as replace (target layer 0)", float(g_nm != g_rp), 0.5)
chk("X3 v2: literal_carrier puts the entity in the slot of every kind", float(x3.literal_carrier("identity2", x3.IDENTITY2[0], "Uzbekistan") != x3.IDENTITY2[0][:-1] + "Uzbekistan"
    or x3.literal_carrier("describe2", x3.DESCRIBE2[0], "Uzbekistan") != x3.DESCRIBE2[0][:-4] + "Uzbekistan ->" or x3.literal_carrier("describe", x3.DESCRIBE[0], "Peru") != x3.DESCRIBE[0][:-2] + "Peru:"), 0.5)
chk("X3 v2: parse_targets", float(x3.parse_targets("same,2,4") != ["same", 2, 4] or x3.parse_targets("-1") != ["same"]), 0.5)
# first-hop position: the last token of the entity before the template tail (char tokens here)
br = {"kind": "bridge", "source": "curated", "prompt": "Fact: The continent of the country whose capital is Gaborone is"}
fp = e12.firsthop_pos(tk, br)
chk("firsthop_pos: the last token of 'Gaborone' (char-level stand-in)", float(fp != len("Fact: The continent of the country whose capital is Gaborone") - 1), 0.5)
chk("firsthop_pos: ' is the' tail", float(e12.firsthop_pos(tk, {"kind": "bridge", "source": "curated", "prompt": "Fact: The currency of the country whose capital is Colombo is the"}) != len("Fact: The currency of the country whose capital is Colombo") - 1), 0.5)
chk("firsthop_pos: None for non-curated bridges and for answer items", float(e12.firsthop_pos(tk, {**br, "source": "lens_eval"}) is not None or e12.firsthop_pos(tk, {**br, "kind": "answer"}) is not None), 0.5)
chk("firsthop_pos: agrees with the prompt ids passed in", float(e12.firsthop_pos(tk, br, tk(br["prompt"]).input_ids[0].tolist()) != fp), 0.5)
chk("track_of: A / B / S from the items file name, override wins", float(e12.track_of("data/h2a_items.json") != "A" or e12.track_of("/x/phrase_items.json") != "B" or e12.track_of("single_items.json") != "S" or e12.track_of("h2a_items.json", "B") != "B"), 0.5)

# --- single-item gating on the tiny model: a 1-piece answer item gates iff the greedy first token is the piece ---
pid = torch.randint(0, V, (1, 12))
with torch.no_grad(): top = int(true_logits(q.w, q.prefill(pid)[0][0, -1]).argmax())
it1 = {"kind": "answer", "pieces": [top], "pieces_text": ["x"], "string": "x", "target": "x"}
ok1, gated1, _ = e12.gate(q, tk, it1, pid)
it0 = {"kind": "answer", "pieces": [(top + 1) % V], "pieces_text": ["y"], "string": "y", "target": "y"}
ok0, _, _ = e12.gate(q, tk, it0, pid)
chk("single-item gate: greedy first token == the token passes, another token fails", float(not (ok1 and gated1) or ok0), 0.5)

# --- X4 probe: split discipline and a separable synthetic problem ---
n, dd, K = 60, 16, 3; g = torch.Generator().manual_seed(1)
y = torch.arange(n) % K; X = torch.randn(n, dd, generator=g) + 4 * torch.nn.functional.one_hot(y, dd).float()
folds = x4.folds_of(n, 5, 0)
chk("X4: folds partition the items (disjoint, complete)", float(sorted(sum(folds, [])) != list(range(n))), 0.5)
tr, te = list(range(0, n, 2)), list(range(1, n, 2)); cats_te = ["c"] * len(te)
blk = x4.split_probe(X, y, K, tr, te, cats_te, [1e-3, 1e-1, 10], 3, 0, log=lambda s: None, name="t")
chk("X4: train / test disjoint and sized", float(set(tr) & set(te) or blk["n_train"] != 30 or blk["n_test"] != 30), 0.5)
chk("X4: near-separable problem gives held-out top-1 >= 0.9", float(blk["heldout"]["top1"] < 0.9), 0.5)
chk("X4: shuffled-label control is at chance (< 0.7 on 3 classes)", float(blk["control_shuffled_labels"]["top1"]), 0.7)
cv = x4.cv_probe(X, y, K, ["a", "b"] * 30, [1e-3, 1e-1, 10], 3, 0, log=lambda s: None, name="cv")
chk("X4: item-level CV scores every item once and separates the synthetic classes (top-1 >= 0.9)", float(cv["cv"]["top1"] < 0.9) + float(len(cv["ranks"]) != n), 0.5)
fam = {0: [0, 1], 1: [0, 1], 2: [2]}
ev, _ = x4.evaluate(torch.eye(K)[y] * 5 + 0.01 * torch.randn(n, K, generator=g), y, 0, fam)
chk("X4: within-family top-1 counts only families with >= 2 classes", float(ev["n_family"] != 40 or ev["within_family_top1"] != 1.0), 0.5)

# --- item-set modes (--single / --phrases) with a word-level stand-in tokenizer ---
class WordTok:
    def encode(s, t, add_special_tokens=False): return [hash(p) % 100000 + 1 for p in re.findall(r" ?[A-Za-z0-9]+|[^A-Za-z0-9 ]| ", t)]
    def decode(s, ids): return "".join(s._inv[i] for i in ids)
    def __init__(s): s._inv = {}
    def _reg(s, t):
        for p in re.findall(r" ?[A-Za-z0-9]+|[^A-Za-z0-9 ]| ", t): s._inv[hash(p) % 100000 + 1] = p
wt = WordTok()
for t in (" New York", " Rimsky-Korsakov", " 1945", "por favor", " x", " New"): wt._reg(t)
chk("items: phrase mode keeps ' New York' (two word tokens) and rejects a hyphenated name", float(mk.encode_string(wt, " New York", "phrase") is None or mk.encode_string(wt, " Rimsky-Korsakov", "phrase") is not None), 0.5)
chk("items: phrase mode allows a quote-led first piece without a space ('por favor')", float(mk.encode_string(wt, "por favor", "phrase") is None), 0.5)
chk("items: single mode keeps exactly one token and rejects two", float(mk.encode_string(wt, " New", "single") is None or mk.encode_string(wt, " New York", "single") is not None), 0.5)
chk("items: multi mode (default) unchanged: >= 2 pieces, one token rejected", float(mk.encode_string(wt, " New York") is None or mk.encode_string(wt, " New") is not None), 0.5)
sp = mk.make_split([{"id": f"i{i}", "kind": "answer"} for i in range(10)] + [{"id": "b", "kind": "bridge"}], 0)
chk("items: make_split halves the answer ids only, disjointly", float(len(sp["tuning"]) != 5 or len(sp["heldout"]) != 5 or set(sp["tuning"]) & set(sp["heldout"]) or "b" in sp["tuning"] + sp["heldout"]), 0.5)
chk("items: phrase tables carry shared_s1_group via item()", float("shared_s1_group" not in (mk.item(wt, "phrase", "place", "answer", "Fact: The largest city is", " New York") or {})), 0.5)

print("\nSUMMARY:", sum(ok for *_, ok in res), "/", len(res), "checks passed")
