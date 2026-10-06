"""N0 fixes (FULL_REPORT section 9; amendments A11c / A12) on the tiny random Qwen3 (CPU, float64) and on the real item
files: (1) identity3 carriers are item-free and the check detects identity2's known overlap; (2) x8's X3 arm with a
target layer equals x3_patchscope's generation; (3) x7 --rank last / sum / mean order a synthetic prefix list as
specified, tau is optional; (4) x8 runs end to end without a tau file, writes every X3 generation and every P3 /
chain prefix, and --rescore re-ranks from the stored prefixes. Content is meaningless on a random model; identities,
formats and rules are what is checked. Run: python tests/test_n0_tiny.py"""
import importlib.util, json, os, shutil, sys, tempfile, torch
sys.path.insert(0, "."); sys.path.insert(0, "tests")
from tiny import tiny
from sjlens.model.qwen3_min import Qwen3Min
from sjlens.model.backend import MinBackend, HFHooksBackend
from sjlens import wsbench_regex as R

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(os.path.dirname(HERE), "scripts"); ROOT = os.path.dirname(HERE)
load = lambda n, f: (lambda s: (lambda mod: (s.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s)))(importlib.util.spec_from_file_location(n, os.path.join(SC, f)))
x7 = load("x7", "x7_chain_reader.py"); x8 = load("x8", "x8_wsbench_items.py"); x3 = x8.x3
res = []
def chk(name, err, tol): res.append((name, err, err < tol)); print(f"[{'PASS' if err < tol else 'FAIL'}] {name}: {err:.2e}")

# ---------------- (1) A12: identity3 exemplars vs the item files and the bench banks ----------------
chk("A12: identity3 has 8 fixed variants of 3 exemplars each (place, word, 4-digit number)", float(len(x3.IDENTITY3) != 8 or any(len(x3.exemplars_of(c)) != 3 for c in x3.IDENTITY3)
    or any(len(x3.exemplars_of(c)[0].split()) != 2 or not x3.exemplars_of(c)[2].isdigit() or len(x3.exemplars_of(c)[2]) != 4 for c in x3.IDENTITY3)), 0.5)
chk("A12: identity3 exemplars are all distinct", float(len({x3.norm(e) for c in x3.IDENTITY3 for e in x3.exemplars_of(c)}) != 24), 0.5)
chk("A12: identity3 overlap check passes on the real data (items + bench banks)", float(len(x3.carrier_overlap("identity3"))), 0.5)
chk("A12: assert_no_overlap('identity3') returns without raising", float(x3.assert_no_overlap("identity3") != []), 0.5)
ov2 = x3.carrier_overlap("identity2"); ex2 = {e for e, _, _ in ov2}
chk("A12: the check DETECTS identity2's known overlap (Kuala Lumpur, Sri Lanka, chimpanzee, 1945, Buenos Aires ...)", float(not {"Kuala Lumpur", "Sri Lanka", "chimpanzee", "1945", "Buenos Aires", "molybdenum"} <= ex2), 0.5)
clean2 = {x3.norm(e) for c in x3.IDENTITY2 for e in x3.exemplars_of(c)} - {x3.norm(e) for e in ex2}
chk("A12: identity2's clean exemplars are exactly 2024 / bibliothèque / praseodymium (21 of 24 overlap)", float(clean2 != {"2024", "bibliothèque", "praseodymium"} or len(ex2) != 21), 0.5)
try: x3.assert_no_overlap("identity2"); raised = False
except AssertionError: raised = True
chk("A12: assert_no_overlap('identity2') raises", float(not raised), 0.5)
chk("A12: identity2 is unchanged (frozen for the wave-2 runs)", float(x3.IDENTITY2[0] != "Kuala Lumpur -> Kuala Lumpur; hippopotamus -> hippopotamus; 1945 -> 1945; ?" or len(x3.IDENTITY2) != 8), 0.5)
# the check itself on a synthetic data root: equality, containment, list-valued targets, bench unit forms / match / bridge answers
tmp = tempfile.mkdtemp(); os.makedirs(os.path.join(tmp, "wsbench"))
json.dump([{"string": " Foo Bar", "target": ["Foo Bar", "FB"], "pieces_text": [" Foo", " Bar"]}, {"string": " x", "target": "x", "pieces_text": [" x"]}], open(os.path.join(tmp, "h2a_items.json"), "w"))
json.dump({"items": [{"name": "a", "target": "t", "target_alts": [], "units": [{"forms": {"en": ["the BAZ thing"]}, "match": ["q"]}], "bridges": [{"answers": ["7788"]}]}]}, open(os.path.join(tmp, "wsbench", "f.json"), "w"))
x3.CARRIERS["_t"] = ["Foo  Bar -> Foo  Bar; baz -> baz; 7788 -> 7788; ?", "Zed Qux -> Zed Qux; nothing -> nothing; 1234 -> 1234; ?"]
ovt = x3.carrier_overlap("_t", root=tmp)
chk("A12 check: equality after whitespace / case folding, containment, list targets and bench forms / bridge answers are all caught", float({(e, s) for e, s, _ in ovt} != {("Foo  Bar", "foo bar"), ("baz", "the baz thing"), ("7788", "7788")}), 0.5)
chk("A12 check: exemplars parse from the identity format", float(x3.exemplars_of("A B -> A B; c -> c; 12 -> 12; ?") != ["A B", "c", "12"]), 0.5)
del x3.CARRIERS["_t"]; shutil.rmtree(tmp, ignore_errors=True)
chk("A12: identity3 shares the identity2 patch position, stops and literal-carrier rule", float(x3.STOPS["identity3"] != x3.STOPS["identity2"] or "identity3" not in x3.IDENTITY_KINDS or x3.literal_carrier("identity3", x3.IDENTITY3[1], "Peru") != x3.IDENTITY3[1][:-1] + "Peru"), 0.5)


# ---------------- (2) x8's X3 target-layer arm == x3_patchscope.patched_generate ----------------
class Tok:  # char-level stand-in tokenizer over the tiny vocabulary (ids 6..255), with the HF surface the scripts use
    all_special_ids = [0, 1]; chat_template = None; pad_token_id = 0; eos_token_id = 1
    def encode(s, t, add_special_tokens=False): return [max(6, min(255, ord(c) - 26)) for c in t]
    def decode(s, ids, skip_special_tokens=False): return "".join(chr(int(i) + 26) if int(i) >= 6 else "" for i in ids)
    def batch_decode(s, rows, skip_special_tokens=False): return [s.decode(r) for r in rows]
    def __call__(s, t, return_tensors=None, add_special_tokens=False):
        ids = s.encode(t)
        return {"input_ids": ids} if return_tensors is None else type("O", (), {"input_ids": torch.tensor([ids])})()
    def convert_ids_to_tokens(s, ids): return [("Ġ" + s.decode([i]).strip()) if s.decode([i]).startswith(" ") else s.decode([i]) for i in ids]


tok = Tok(); m = tiny(); q = Qwen3Min(m); B = MinBackend(tok, q); HB = HFHooksBackend(m, tok); V, d, T = 256, 64, 16
cid = tok(x3.IDENTITY3[0], return_tensors="pt").input_ids; cp = cid.shape[1] - 1; h = torch.randn(d, dtype=torch.float64) * 0.5; L = 2
for tl in (1, L, 0):
    gX = x3.patched_generate(q, cid, L, cp, h, 4, "replace", 1.0, None, tl); gA = x8.x3_generate(B, cid, cp, h, tl, 4); gH = x8.x3_generate(HB, cid, cp, h, tl, 4)
    chk(f"x8 X3 arm: x3_generate(target {tl}) == x3.patched_generate (MinBackend and HF hooks)", float(gA != gX or gH != gX), 0.5)
xc = B.resid(cid, 1)[0, cp].float().clone()
chk("x8 X3 arm: a cached clean carrier residual gives the same generation", float(x8.x3_generate(B, cid, cp, h, 1, 4, xc) != x8.x3_generate(B, cid, cp, h, 1, 4)), 0.5)
chk("x8 X3 arm keys: x3_<kind> at the read layer, x3_<kind>_t<l> for an early target", float(x8.x3_arm_name("identity3", "same", 22) != "x3_identity3" or x8.x3_arm_name("identity3", 4, 22) != "x3_identity3_t4" or x8.x3_arm_name("identity2", 22, 22) != "x3_identity2"), 0.5)

# ---------------- (3) x7 --rank last | sum | mean on a synthetic prefix list; tau optional ----------------
A = {"ids": [10, 13, 14], "t": "A_meanlow", "S": 0.6, "d": [0.1, 0.5], "fr": 1}   # sum .6, last .5, mean .3
Bp = {"ids": [10, 12, 15], "t": "B_sumbest", "S": 0.8, "d": [0.6, 0.2], "fr": 1}  # sum .8, last .2, mean .4
C = {"ids": [10, 16], "t": "C_lastbest", "S": 0.7, "d": [0.7], "fr": 1}           # sum .7, last .7, mean .7
Dn = {"ids": [11, 17], "t": "D_negative", "S": -0.3, "d": [-0.3], "fr": 2}
P = [{"ids": [10], "t": "first", "S": 0.0, "d": [], "fr": 1}, {"ids": [11], "t": "second", "S": 0.0, "d": [], "fr": 2}, A, Bp, C, Dn]
chk("rank: score of a prefix = S (sum) / last step boost (last) / mean step boost (mean); single tokens score S under every rule",
    float(x7.rank_score(Bp, "sum") != 0.8 or x7.rank_score(Bp, "last") != 0.2 or abs(x7.rank_score(Bp, "mean") - 0.4) > 1e-12 or x7.rank_score(P[0], "last") != 0.0 or x7.rank_score(C, "mean") != 0.7), 0.5)
chk("rank: default is last (A11c)", float(x7.RANK_DEFAULT != "last" or x7.samples_topn(P, 2) != x7.samples_topn(P, 2, "last")), 0.5)
chk("rank last: N=10 set = the first reader's top-1 token, then C (.7) > A (.5) > B (.2) > single tokens (0) > D (-.3)", float(x7.samples_topn(P, 10, "last") != ["first", "C_lastbest", "A_meanlow", "B_sumbest", "second", "D_negative"]), 0.5)
chk("rank sum: B (0.8) > C (0.7) > A (0.6) > singles > D", float(x7.samples_topn(P, 10, "sum") != ["first", "B_sumbest", "C_lastbest", "A_meanlow", "second", "D_negative"]), 0.5)
chk("rank mean: C (0.7) > B (0.4) > A (0.3)", float(x7.samples_topn(P, 4, "mean") != ["first", "C_lastbest", "B_sumbest", "A_meanlow"]), 0.5)
chk("rank last, no tau: N=1 is the last-step-best multi-piece prefix (C)", float(x7.sample_top1(P, None, "last") != "C_lastbest"), 0.5)
chk("rank sum, no tau: N=1 is the sum-best prefix (B)", float(x7.sample_top1(P, None, "sum") != "B_sumbest"), 0.5)
chk("tau 0.65 under rank sum: B (a 0.2 step) is excluded, C remains", float(x7.sample_top1(P, 0.65, "sum") != "C_lastbest"), 0.5)
chk("tau 0.15 under rank sum: B is valid again", float(x7.sample_top1(P, 0.15, "sum") != "B_sumbest"), 0.5)
chk("tau impossible: falls to the first reader's top-1 token under every rule", float(any(x7.sample_top1(P, 1e9, r) != "first" for r in x7.RANKS)), 0.5)
try: x7.rank_score(C, "max"); bad = False
except ValueError: bad = True
chk("rank: an unknown rule is rejected", float(not bad), 0.5)
row = {"id": "i", "category": "c", "key": x7.normexact("C_lastbest"), "reader": "p3", "control": "none", "prefixes": P, "n_samples": 2, "first_hit_rank": 1, "n_boosts": 2, "rank_s2_clean": 1.0}
sL = x7._score_row(row, None, "last"); sS = x7._score_row(row, None, "sum")
chk("_score_row rebuilds the N-sample set from the stored prefixes under the rule (n_samples 2: C in under last, out under sum)", float(sL["exact_top10"] != 1 or sS["exact_top10"] != 0 or sL["samples10"] != ["first", "C_lastbest"] or sS["exact_top1"] != 0 or sL["exact_top1"] != 1), 0.5)
rr = {"layers": {"2": {"rows": [dict(row, **sS)]}}, "track": "A"}
x7.resummarise(rr, None, 0, "last")
chk("resummarise (--rescore) honours --rank: the stored row's flags flip to the last rule; rank and tau recorded", float(rr["rank"] != "last" or rr["tau"] is not None or rr["layers"]["2"]["rows"][0]["exact_top10"] != 1 or rr["layers"]["2"]["summary"]["p3"]["none"]["rank"] != "last" or rr["any_layer"]["p3"]["none"]["exact_top10"] != 1.0), 0.5)
ft = x7.fit_tau({"layers": {"2": {"rows": [row]}}}, log=lambda s: None, rank="last")
chk("fit_tau under rank last picks a grid tau at which C is the N=1 sample (exact 1)", float(ft["exact_top1"] != 1.0 or ft["rank"] != "last"), 0.5)
chk("TAU_DEFAULT is None (tau optional; no implicit tau file)", float(x7.TAU_DEFAULT is not None), 0.5)

# ---------------- (4) x8 end to end on the tiny model, no tau file; X3 targets same,1; prefixes on disk; --rescore ----------------
runs = os.path.join(ROOT, "runs"); tmp = tempfile.mkdtemp()
lens_f = os.path.join(tmp, "lens.pt"); torch.save({"J": {l: torch.randn(d, d) * 0.05 for l in range(4)}, "source_layers": [0, 1, 2, 3], "d_model": d}, lens_f)
g = torch.Generator().manual_seed(0); recs = [torch.randint(6, V, (T,), generator=g) for _ in range(40)]
ctx_f = os.path.join(tmp, "ctx.pt"); torch.save({"model": "tiny", "T": T, "ctx": recs[:24], "bg": recs[24:32]}, ctx_f)
for mod in (x7, x8): mod.hf_hub_download = lambda repo, f: lens_f
x7.load_backend = lambda kind, model, dtype, device: (tok, m, B); x8.x7 = x7
import sjlens.eval.phase_a as pa
pa_orig = pa.contexts
def fake_contexts(tok_, n_ctx, n_bg, T_, device, path=None):
    o = torch.load(path, weights_only=False); return [c for c in o["ctx"][:n_ctx]], [c for c in o["bg"][:n_bg]]
pa.contexts = fake_contexts; x7.contexts = fake_contexts
bank_items = []
for i in range(4):
    prompt = f"The thing number {i} is called the"; ids = tok(prompt, return_tensors="pt").input_ids; tgt = tok.decode(B.greedy(ids, 3)).strip()
    forms = [tgt + " zz", tgt + " qq"]
    bank_items.append({"name": f"tiny-{i}*x", "subfamily": "entity", "prompt": prompt, "target": tgt, "target_alts": [], "eval_render": "plain", "readout": {"kind": "final_prompt_token", "offsets": [-1]}, "intermediates": [tgt],
                       "units": [{"role": "readout", "required": True, "multi_token": True, "forms": {"en": forms}, "match": forms}], "probe_token_lens": {"units": {"readout": {"en": [len(f) for f in forms]}}, "target": len(tgt)}, "bridges": []})
bank_root = os.path.join(tmp, "bank"); os.makedirs(bank_root); json.dump({"family": "tiny-mt", "contract": {"multi_token": True, "include_target": False, "conjunctive_units": True}, "items": bank_items}, open(os.path.join(bank_root, "tiny_mt.json"), "w"))
tag = "smk_n0_p6"; shutil.rmtree(os.path.join(runs, tag), ignore_errors=True)
sys.argv = ["x8", "--model", "tiny", "--bank-root", bank_root, "--families", "tiny_mt", "--layers", "1,3", "--p3-layers", "2", "--first", "logitlens", "--first-layer", "3", "--readers", "jlens,logitlens,jlens_chain,logitlens_chain,p3,x3",
            "--x3-target-layer", "same,1", "--x3-carriers", "2", "--x3-gen", "3", "--controls", "shuffled", "--n-ctx", "4", "--chunk", "4", "--T", str(T), "--contexts", ctx_f, "--k1", "2", "--beam", "2", "--max-pieces", "3", "--gate-tokens", "3", "--device", "cpu", "--tag", tag]
x8.main(); r8 = json.load(open(os.path.join(runs, tag, "results.json"))); f8 = r8["families"]["tiny_mt"]
chk("x8 e2e: runs without a tau file (tau None), rank last recorded, identity3 carriers with targets same / 1 recorded", float(r8["tau"] is not None or r8["rank"] != "last" or r8["x3"]["kind"] != "identity3" or r8["x3"]["targets"] != ["same", 1] or r8["x3"]["carriers"] != x3.IDENTITY3[:2]), 0.5)
chk("x8 e2e: the token grid is extended by the read layer (1,3 -> 1,2,3) so the J-lens / logit lens are read on the X3 cells", float(r8["layers"] != [1, 2, 3] or r8["layers_requested"] != [1, 3]), 0.5)
want = {"jlens", "logitlens", "jlens_chain", "logitlens_chain", "p3", "p3_top1", "x3_identity3", "x3_identity3_t1"}
chk("x8 e2e: every arm incl. x3_identity3 and x3_identity3_t1, each with its shuffled control, is written and scored", float(not want <= set(f8["arms"]) or not {a + "__shuffled" for a in want} <= set(f8["arms"])), 0.5)
for arm in ("x3_identity3", "x3_identity3_t1"):
    rows8 = [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"][arm]["file"]), encoding="utf-8")]
    chk(f"x8 e2e: {arm} readouts = one prose row per (gated item, read layer) with one sample per carrier (2), parsed by the bench contract", float(len(rows8) != 4 or any(len(r["samples"]) != 2 or r["layer"] != 2 or R.parse_readout_row(r) is None for r in rows8)), 0.5)
    chk(f"x8 e2e: {arm} scored over the read layers only", float(f8["arms"][arm]["layers"] != [2]), 0.5)
# the X3 samples equal x3_patchscope's generation for the same (h, carrier, target), cut at the carrier stops
it0 = bank_items[0]; ids0 = tok(it0["prompt"], return_tensors="pt").input_ids; h0 = q.resid(ids0, 2)[0, -1]
rows_t1 = [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["x3_identity3_t1"]["file"]), encoding="utf-8")]; rows_same = [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["x3_identity3"]["file"]), encoding="utf-8")]
r_t1 = next(r for r in rows_t1 if r["id"] == R.label_of(it0["name"])); r_same = next(r for r in rows_same if r["id"] == R.label_of(it0["name"]))
exp_t1 = [x3.cut(tok.decode(x3.patched_generate(q, tok(c, return_tensors="pt").input_ids, 2, len(tok.encode(c)) - 1, h0, 3, "replace", 1.0, None, 1)), "identity3") for c in x3.IDENTITY3[:2]]
exp_same = [x3.cut(tok.decode(x3.patched_generate(q, tok(c, return_tensors="pt").input_ids, 2, len(tok.encode(c)) - 1, h0, 3, "replace", 1.0, None, 2)), "identity3") for c in x3.IDENTITY3[:2]]
chk("x8 e2e: the x3_identity3_t1 samples == x3.patched_generate(target 1) cut at the stops; x3_identity3 == target = read layer", float(r_t1["samples"] != exp_t1 or r_same["samples"] != exp_same), 0.5)
chk("x8 e2e: token / bag arms report their pass rate on the read layers (same cells as X3) beside the full grid", float(any("on_read_layers" not in f8["arms"][a] or f8["arms"][a]["on_read_layers"]["layers"] != [2] for a in ("jlens", "logitlens", "jlens_chain", "logitlens_chain")) or "pass_on_read_layers" not in r8["table"]["jlens"]["tiny_mt"]), 0.5)
pdir = os.path.join(runs, tag, "prefixes")
chk("x8 e2e: every P3 / chain prefix is written to runs/<tag>/prefixes/<arm>__<ctrl>/<family>.jsonl", float(any(not os.path.exists(os.path.join(pdir, s, "tiny_mt.jsonl")) for s in ("p3", "p3__shuffled", "jlens_chain", "logitlens_chain", "logitlens_chain__shuffled"))), 0.5)
pr = [json.loads(l) for l in open(os.path.join(pdir, "p3", "tiny_mt.jsonl"), encoding="utf-8")]
chk("x8 e2e: stored p3 prefixes carry ids / text / S / step boosts / first rank and the candidates, one row per (item, read layer)", float(len(pr) != 4 or any(k not in pr[0]["prefixes"][0] for k in ("ids", "t", "S", "d", "fr")) or "cands" not in pr[0] or pr[0]["layer"] != 2), 0.5)
p3rows = [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["p3"]["file"]), encoding="utf-8")]
chk("x8 e2e: the p3 readouts samples == samples_topn(stored prefixes, 10, last); p3_top1 == sample_top1(prefixes, None, last)", float(any(r["samples"] != x7.samples_topn(p["prefixes"], 10, "last") for r, p in zip(p3rows, pr))
    or any(r["samples"] != [x7.sample_top1(p["prefixes"], None, "last")] for r, p in zip([json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["p3_top1"]["file"]), encoding="utf-8")], pr))), 0.5)
# --rescore under rank sum, from the stored prefixes (no model): the p3 samples are rebuilt, every arm re-scored, the table rewritten
sys.argv = ["x8", "--rescore", "--rank", "sum", "--tag", tag]; x8.main(); r8b = json.load(open(os.path.join(runs, tag, "results.json"))); f8b = r8b["families"]["tiny_mt"]
p3rows_b = [json.loads(l) for l in open(os.path.join(ROOT, f8b["arms"]["p3"]["file"]), encoding="utf-8")]
chk("x8 --rescore honours --rank: rank sum recorded, p3 samples rebuilt under sum from the prefixes on disk, x3 / token arms kept", float(r8b["rank"] != "sum" or not r8b.get("rescored") or any(r["samples"] != x7.samples_topn(p["prefixes"], 10, "sum") for r, p in zip(p3rows_b, pr))
    or set(f8b["arms"]) != set(f8["arms"]) or f8b["arms"]["x3_identity3_t1"]["pass_rate"] != f8["arms"]["x3_identity3_t1"]["pass_rate"] or f8b["arms"]["jlens"]["pass_rate"] != f8["arms"]["jlens"]["pass_rate"]), 0.5)
chk("x8 e2e: judge_cmds.sh lists the prose x3 arms", float("x3_identity3_t1" not in open(os.path.join(runs, tag, "judge_cmds.sh")).read()), 0.5)
# x7 main end to end without a tau file: results carry tau None and rank last; --rescore --rank sum re-ranks the stored rows
items = []
for i in range(6):
    prompt = f"Fact: item {i} is the"; ids = tok(prompt, return_tensors="pt").input_ids; pieces = B.greedy(ids, 2 + (i % 2))
    items.append({"id": f"a{i}", "source": "curated", "category": "cap", "kind": "answer", "name": "", "prompt": prompt, "readout": "last_prompt_token", "string": tok.decode(pieces), "pieces": pieces, "pieces_text": [tok.decode([p]) for p in pieces], "target": tok.decode(pieces).strip(), "p1_chars": 2})
items_f = os.path.join(tmp, "items.json"); json.dump(items, open(items_f, "w")); split_f = os.path.join(tmp, "split.json"); json.dump({"seed": 0, "tuning": ["a0", "a1", "a2"], "heldout": ["a3", "a4", "a5"]}, open(split_f, "w"))
tag7 = "smk_n0_p3"; shutil.rmtree(os.path.join(runs, tag7), ignore_errors=True)
sys.argv = ["x7", "--items", items_f, "--split-file", split_f, "--split", "heldout", "--layers", "2", "--first", "jlens", "--first-layer", "same", "--k1", "3", "--beam", "2", "--max-pieces", "3", "--n-ctx", "4", "--chunk", "4", "--T", str(T), "--contexts", ctx_f, "--device", "cpu", "--tag", tag7]; x7.main()
r7 = json.load(open(os.path.join(runs, tag7, "results.json"))); row7 = next(r for r in r7["layers"]["2"]["rows"] if r["reader"] == "p3")
chk("x7 e2e: runs with no tau (None) under rank last; rows store prefixes, n_samples and the last-rule samples", float(r7["tau"] is not None or r7["rank"] != "last" or row7["samples10"] != x7.samples_topn(row7["prefixes"], 10, "last") or row7["sample_top1"] != x7.sample_top1(row7["prefixes"], None, "last")), 0.5)
sys.argv = ["x7", "--rescore", "--rank", "sum", "--tag", tag7]; x7.main(); r7b = json.load(open(os.path.join(runs, tag7, "results.json"))); row7b = next(r for r in r7b["layers"]["2"]["rows"] if r["reader"] == "p3")
chk("x7 --rescore --rank sum re-ranks every stored row from its prefixes (rank recorded, samples under sum)", float(r7b["rank"] != "sum" or row7b["samples10"] != x7.samples_topn(row7b["prefixes"], 10, "sum") or r7b["layers"]["2"]["summary"]["p3"]["none"]["rank"] != "sum"), 0.5)

pa.contexts = pa_orig
for t in (tag, tag7): shutil.rmtree(os.path.join(runs, t), ignore_errors=True)
shutil.rmtree(tmp, ignore_errors=True)
print("\nSUMMARY:", sum(ok_ for *_, ok_ in res), "/", len(res), "checks passed")
