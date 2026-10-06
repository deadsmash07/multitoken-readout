"""Wave-4 code paths (prereg A15-A18; ANALYSIS_WAVE3 section 6 E1 / E2 / E3 / E5a) on the tiny random Qwen3 (CPU, float64)
and on the real item files: (1) the gate's `immediate` flag and its offline re-derivation; (2) the new carrier kinds are
item-free and the extended check catches both directions; (3) x3s seeded generation == the patched carrier + seed through
the plain generator (Min and HF backends), lens seeds, typo spans; (4) x8 end to end with --no-gate, x3c / x3s / x3long /
spellfix / span positions / _uncut arms and the three denominators, then --rescore; (5) the E5a bench-rule rescoring;
(6) the Gemini judge wrapper's protocol (bundle, fingerprint, scoring with the summary as one more sample, client
downgrade / retry) with a stub HTTP function; (7) x3_patchscope.py with identity3_long + continuation. Content is
meaningless on a random model; identities, formats and rules are what is checked. Run: python tests/test_wave4_tiny.py"""
import importlib.util, json, os, shutil, sys, tempfile, torch
sys.path.insert(0, "."); sys.path.insert(0, "tests")
from tiny import tiny
from sjlens.model.qwen3_min import Qwen3Min
from sjlens.model.backend import MinBackend, HFHooksBackend
from sjlens import wsbench_regex as R

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(os.path.dirname(HERE), "scripts"); ROOT = os.path.dirname(HERE)
load = lambda n, f: (lambda s: (lambda mod: (s.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s)))(importlib.util.spec_from_file_location(n, os.path.join(SC, f)))
x7 = load("x7", "x7_chain_reader.py"); x8 = load("x8", "x8_wsbench_items.py"); x3 = x8.x3; e5a = load("e5a", "e5a_bench_rule_rescore.py"); jg = load("jg", "wsbench_judge_gemini.py")
res = []
def chk(name, err, tol=0.5): res.append((name, err, err < tol)); print(f"[{'PASS' if err < tol else 'FAIL'}] {name}: {err:.2e}")

# ---------------- (1) immediate flag ----------------
forms = ["Aghlabid dynasty", "Aghlabids", "Banu al-Aghlab"]
chk("immediate: continuation starting with an alias (word boundary) -> True", float(not x8.immediate_hit(" Aghlabids. The Aghlabids were a dynasty", forms)))
chk("immediate: after an article / quotes -> True", float(not x8.immediate_hit(' the "Aghlabid dynasty", which ruled', forms)))
chk("immediate: target later in the text (eventual) -> False", float(x8.immediate_hit(" a dynasty called the Aghlabids", forms)))
chk("immediate: a longer word sharing the prefix -> False", float(x8.immediate_hit(" Aghlabidian rulers", forms)))
chk("immediate: fold (accents, case) applies", float(not x8.immediate_hit(" MÉXICO city is", ["Mexico City"])))
chk("immediate: CJK form as a plain prefix", float(not x8.immediate_hit("北京是首都", ["北京"])))
chk("immediate: empty forms -> False", float(x8.immediate_hit(" anything", ["", None])))
it_ = {"target": "Aghlabid dynasty", "target_alts": ["Aghlabids"]}
old_row = {"id": "a", "ok": True, "surface": {"greedy": " Aghlabids. The", "ok": True}}
chk("immediate_of_row: a pre-A15 row (no flag) is re-derived from the stored greedy text", float(not x8.immediate_of_row(old_row, it_) or x8.immediate_of_row({**old_row, "surface": {"greedy": " the rulers, the Aghlabids", "ok": True}}, it_)))
chk("immediate_of_row: a stored flag is respected; a failed gate is never immediate", float(x8.immediate_of_row({**old_row, "immediate": False}, it_) or x8.immediate_of_row({**old_row, "ok": False}, it_)))

# ---------------- (2) carriers (A15 / A17): item-free in both directions ----------------
chk("A15/A17: every wave-4 kind passes the extended overlap check on the real data", float(any(x3.carrier_overlap(k) for k in x3.WAVE4_KINDS)))
chk("A15/A17: assert_no_overlap returns [] for every wave-4 kind", float(any(x3.assert_no_overlap(k) != [] for k in x3.WAVE4_KINDS)))
chk("A12 unchanged: identity3 still clean; identity2 still detected", float(x3.carrier_overlap("identity3") != [] or not x3.carrier_overlap("identity2")))
chk("identity3_long: 8 variants x 3 exemplars of 3-4 words (A15)", float(len(x3.IDENTITY3_LONG) != 8 or any(len(x3.exemplars_of(c)) != 3 or any(not 3 <= len(e.split()) <= 4 for e in x3.exemplars_of(c)) for c in x3.IDENTITY3_LONG)))
chk("spellfix: 8 variants x 3 exemplars, both sides parsed, misspelling != correction", float(len(x3.SPELLFIX) != 8 or any(len(x3.exemplar_sides_of(c)) != 6 for c in x3.SPELLFIX) or any(x3.exemplar_sides_of(c)[2 * i] == x3.exemplar_sides_of(c)[2 * i + 1] for c in x3.SPELLFIX for i in range(3))))
chk("continuation: 8 prefixes, 2 seed frames, no stop cut, patch at the last token", float(len(x3.CONTINUATION) != 8 or len(x3.CONTINUATION_SEED) != 2 or x3.STOPS["continuation"] != () or x3.cut("a; b -> c\n", "continuation") != "a; b -> c\n"))
chk("identity kinds: identity3_long / spellfix patch at '?' with the identity stops", float(any(k not in x3.IDENTITY_KINDS or x3.STOPS[k] != x3.STOPS["identity3"] for k in ("identity3_long", "spellfix"))))
tmp = tempfile.mkdtemp(); os.makedirs(os.path.join(tmp, "wsbench"))
json.dump([{"string": " Foo Bar", "target": "Foo Bar", "pieces_text": [" Foo", " Bar"]}, {"string": " committee", "target": "committee", "pieces_text": [" committee"]}, {"string": " x", "target": "x", "pieces_text": [" x"]}], open(os.path.join(tmp, "h2a_items.json"), "w"))
json.dump({"items": [{"name": "t", "target": "receive", "typo_word": "recieve", "units": [{"forms": {"en": ["receive"]}}]}]}, open(os.path.join(tmp, "wsbench", "typo.json"), "w"))
x3.CARRIERS["_w4"] = ["the Foo Bar Inn -> the Foo Bar Inn; nothing here -> nothing here; 1234 -> 1234; ?"]; x3.CARRIERS["_w4c"] = ["The committee met on Tuesday and the"]; x3.CARRIERS["_w4s"] = ["recieve -> receive; wierd -> weird; ?"]
x3.WAVE4_KINDS = x3.WAVE4_KINDS + ("_w4", "_w4c", "_w4s")
o1 = {(e, s) for e, s, _ in x3.carrier_overlap("_w4", root=tmp)}; o2 = {(e, s) for e, s, _ in x3.carrier_overlap("_w4c", root=tmp)}; o3 = {(e, s) for e, s, _ in x3.carrier_overlap("_w4s", root=tmp)}
chk("extended check: an item string INSIDE a long exemplar is caught (reverse containment), 1-letter strings ignored", float(o1 != {("the Foo Bar Inn", "foo bar")}))
chk("extended check: an item string as a whole word in a continuation prompt is caught", float(o2 != {("The committee met on Tuesday and the", "committee")}))
chk("extended check: BOTH sides of a spellfix exemplar are checked (typo_word and target from the bank)", float(o3 != {("recieve", "recieve"), ("receive", "receive")}))
chk("typo_word strings are part of item_strings (A17)", float(("recieve", "wsbench/typo.json") not in x3.item_strings(tmp)))
x3.WAVE4_KINDS = tuple(k for k in x3.WAVE4_KINDS if not k.startswith("_w4"))
for k in ("_w4", "_w4c", "_w4s"): del x3.CARRIERS[k]
shutil.rmtree(tmp, ignore_errors=True)


# ---------------- (3) seeded generation, lens seeds, typo span ----------------
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
cid = tok(x3.CONTINUATION_SEED[0], return_tensors="pt").input_ids; cp = cid.shape[1] - 1; h = torch.randn(d, dtype=torch.float64) * 0.5; L = 2
chk("x3s: seeded_generate with an empty seed == x3_generate", float(x8.seeded_generate(B, cid, cp, h, 1, [], 4) != x8.x3_generate(B, cid, cp, h, 1, 4)))
for tl in (1, L):
    seed = [40]; g = x8.seeded_generate(B, cid, cp, h, tl, seed, 4)
    ids2 = torch.cat([cid, torch.tensor([seed])], 1); mask = torch.zeros(1, ids2.shape[1], dtype=torch.bool); mask[0, cp] = True
    xc = B.resid(cid, tl)[0, cp]; ref = seed + B.greedy(ids2, 4, (tl, mask, (h - xc).view(1, 1, -1).expand(1, ids2.shape[1], -1)))
    ref2 = seed + x3.patched_generate(q, ids2, L, cp, h, 4, "replace", 1.0, None, tl)
    gh = x8.seeded_generate(HB, cid, cp, h, tl, seed, 4)
    chk(f"x3s: seeded_generate(target {tl}) == carrier + seed with h patched at the frame's last token (Min == manual == x3.patched_generate == HF hooks); starts with the seed", float(g != ref or g != ref2 or gh != g or g[:1] != seed))
xc1 = B.resid(cid, 1)[0, cp].float().clone()
chk("x3s: the cached clean residual (carrier alone) gives the same seeded generation (causal: the seed does not change cp)", float(x8.seeded_generate(B, cid, cp, h, 1, [41], 3, xc1) != x8.seeded_generate(B, cid, cp, h, 1, [41], 3)))
z = torch.randn(V, dtype=torch.float64); z[0] = 100.0; z[1] = 99.0
top = x8.lens_seeds(z, 5, [0, 1])
chk("x3s: lens_seeds = top-k ids with the special tokens excluded", float(len(top) != 5 or 0 in top or 1 in top or top != torch.topk(z[2:], 5).indices.add(2).tolist()))
chk("arm names: x3c / x3c_t4 / x3s_ll_t8 / x3_identity3_long", float(x8.arm_of("x3c", "same", 32) != "x3c" or x8.arm_of("x3c", 4, 32) != "x3c_t4" or x8.arm_of("x3s_ll", 8, 32) != "x3s_ll_t8" or x8.x3_arm_name("identity3_long", "same", 32) != "x3_identity3_long"))
chk("is_read_layer_arm covers x3c / x3s / x3_<kind> / p3", float(not all(x8.is_read_layer_arm(a) for a in ("x3c", "x3s_ll_t4", "x3_identity3", "x3_spellfix_t4_off2", "p3")) or x8.is_read_layer_arm("jlens")))
pid = tok("this word is misspelt hte", return_tensors="pt").input_ids
sp = x8.typo_span(tok, pid, "hte")
chk("typo_span: the last k tokens that spell typo_word (char-level: 3 positions), [] when absent", float(sp != [pid.shape[1] - 3, pid.shape[1] - 2, pid.shape[1] - 1] or x8.typo_span(tok, pid, "zzz") != [] or x8.typo_span(tok, pid, "") != []))

# ---------------- (4) x8 end to end: --no-gate, new readers, span positions, denominators, --rescore ----------------
runs = os.path.join(ROOT, "runs"); tmp = tempfile.mkdtemp()
lens_f = os.path.join(tmp, "lens.pt"); torch.save({"J": {l: torch.randn(d, d) * 0.05 for l in range(4)}, "source_layers": [0, 1, 2, 3], "d_model": d}, lens_f)
for mod in (x7, x8): mod.hf_hub_download = lambda repo, f: lens_f
x7.load_backend = lambda kind, model, dtype, device: (tok, m, B); x8.x7 = x7
bank_items = []
for i in range(5):
    prompt = f"The thing number {i} is called the wrd"; ids = tok(prompt, return_tensors="pt").input_ids; tgt = tok.decode(B.greedy(ids, 3)).strip() or "zz"; tgt = tgt if i < 2 else f"{tgt}{i}"  # distinct targets so the within-subfamily control has partners
    forms = [tgt + " zz", tgt + " qq"]
    bank_items.append({"name": f"tiny-{i}*x", "subfamily": "entity" if i % 2 else "novel", "prompt": prompt, "target": tgt, "target_alts": [], "typo_word": "wrd", "eval_render": "plain", "readout": {"kind": "final_prompt_token", "offsets": [-1]}, "intermediates": [tgt],
                       "units": [{"role": "correction", "required": True, "multi_token": True, "forms": {"en": forms}, "match": forms}], "probe_token_lens": {"units": {"correction": {"en": [len(f) for f in forms]}}, "target": len(tgt)}, "bridges": []})
bank_root = os.path.join(tmp, "bank"); os.makedirs(bank_root); json.dump({"family": "tiny-typo", "contract": {"multi_token": True, "include_target": False, "conjunctive_units": True}, "items": bank_items}, open(os.path.join(bank_root, "tiny_typo.json"), "w"))
tag = "smk_w4_p6"; shutil.rmtree(os.path.join(runs, tag), ignore_errors=True)
sys.argv = ["x8", "--model", "tiny", "--bank-root", bank_root, "--families", "tiny_typo", "--layers", "1,3", "--p3-layers", "2", "--readers", "jlens,logitlens,x3,x3c,x3s,x3long", "--x3-carriers-kind", "identity3,spellfix", "--x3-target-layer", "same,1", "--x3-carriers", "2", "--x3-gen", "3", "--x3c-gen", "4", "--x3s-gen", "2", "--x3s-seeds", "ll,jl",
            "--x3-positions", "span", "--controls", "shuffled", "--gate-tokens", "3", "--no-gate", "--device", "cpu", "--tag", tag]
x8.main(); r8 = json.load(open(os.path.join(runs, tag, "results.json"))); f8 = r8["families"]["tiny_typo"]
chk("x8 --no-gate: every item is scored (n_scored == n_items), the gate is recorded with the immediate flag, no_gate stamped", float(f8["n_scored"] != 5 or not f8["no_gate"] or any("immediate" not in r or "immediate" not in r["surface"] for r in f8["gate_rows"]) or not r8["no_gate"]))
chk("x8: results carry the kinds list, the x3c / x3s blocks and the prereg A15-A17 tag; single-kind fields kept (carriers = first kind)", float(r8["x3"]["kinds"] != ["identity3", "spellfix", "identity3_long"] or r8["x3"]["carriers"] != x3.IDENTITY3[:2] or r8["x3c"]["gen"] != 4 or r8["x3s"]["seed_lenses"] != ["ll", "jl"] or "A15" not in r8["prereg"]))
want = {"jlens", "logitlens", "x3_identity3", "x3_identity3_t1", "x3_identity3_uncut", "x3_spellfix", "x3_spellfix_t1", "x3_identity3_long", "x3_identity3_long_t1", "x3c", "x3c_t1", "x3s_ll", "x3s_ll_t1", "x3s_jl", "x3s_jl_t1", "x3_identity3_off2", "x3_spellfix_t1_off3", "x3c_off2"}
chk("x8: every new arm (x3c, x3s_ll / _jl, identity3_long, spellfix, _uncut, _off<k> span positions) with its shuffled control is written and scored", float(not want <= set(f8["arms"]) or not {a + "__shuffled" for a in want} <= set(f8["arms"])))
rows_c = [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["x3c_t1"]["file"]), encoding="utf-8")]
rows_s = [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["x3s_ll_t1"]["file"]), encoding="utf-8")]
rows_u = [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["x3_identity3_t1_uncut"]["file"]), encoding="utf-8")]; rows_i = [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["x3_identity3_t1"]["file"]), encoding="utf-8")]
chk("x8 x3c: one prose row per (item, read layer) with one UNCUT sample per carrier (2), 4 tokens, contract-valid", float(len(rows_c) != 5 or any(len(r["samples"]) != 2 or r["layer"] != 2 or R.parse_readout_row(r) is None or any(len(s) != 4 for s in r["samples"]) for r in rows_c)))
chk("x8 x3s: 5 seeds x 2 frames = 10 samples per cell, each = seed + 2 tokens", float(len(rows_s) != 5 or any(len(r["samples"]) != 10 or any(len(s) != 3 for s in r["samples"]) for r in rows_s)))
chk("x8 _uncut: the raw generations; the base arm = their cut", float(any(len(u["samples"]) != 2 or [x3.cut(s, "identity3") for s in u["samples"]] != i["samples"] for u, i in zip(rows_u, rows_i))))
it0 = bank_items[0]; ids0 = tok(it0["prompt"], return_tensors="pt").input_ids; h0 = q.resid(ids0, 2)[0, -1]
r_c = next(r for r in rows_c if r["id"] == R.label_of(it0["name"])); exp_c = [tok.decode(x3.patched_generate(q, tok(c, return_tensors="pt").input_ids, 2, len(tok.encode(c)) - 1, h0, 4, "replace", 1.0, None, 1)) for c in x3.CONTINUATION[:2]]
chk("x8 x3c: samples == x3.patched_generate at the prefix's last token (target 1), uncut", float(r_c["samples"] != exp_c))
r_s = next(r for r in rows_s if r["id"] == R.label_of(it0["name"])); seeds0 = x8.lens_seeds(B.logits(h0), 5, tok.all_special_ids)
exp_s = [tok.decode(x8.seeded_generate(B, tok(c, return_tensors="pt").input_ids, len(tok.encode(c)) - 1, h0, 1, [sd], 2)) for c in x3.CONTINUATION_SEED for sd in seeds0]
chk("x8 x3s_ll: samples == seeded_generate over (frame, logit-lens top-5 seed) at target 1", float(r_s["samples"] != exp_s))
rows_o = [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["x3_identity3_off2"]["file"]), encoding="utf-8")]
chk("x8 span: the _off2 arm reads the second-to-last token (pos = final - 1, the typo span piece) with its own token string", float(any(r["pos"] != len(tok.encode(bank_items[k]["prompt"])) - 2 or r["token"] != "r" for k, r in enumerate(rows_o))))
e_ = f8["arms"]["x3c"]; dn = e_["denominators"]
chk("x8 denominators: all (5) / gated / gated-immediate blocks with n, pass, CI; gated == gate rows ok; immediate <= gated", float(dn["all"]["n"] != 5 or dn["gated"]["n"] != sum(r["ok"] for r in f8["gate_rows"]) or dn["gated_immediate"]["n"] != sum(r["immediate"] for r in f8["gate_rows"]) or dn["gated_immediate"]["n"] > dn["gated"]["n"] or "pass_ci" not in dn["all"]))
tb = r8["table"]["x3c"]["tiny_typo"]
chk("x8 table: pass_all / n_all / pass_gated / n_gated / pass_immediate / n_immediate beside pass", float(any(k not in tb for k in ("pass", "pass_all", "n_all", "pass_gated", "n_gated", "pass_immediate", "n_immediate")) or tb["n_all"] != 5 or tb["pass_all"] != e_["pass_rate"]))
chk("x8 shuffled control: partners within subfamily, never the item itself (rows are per item; samples differ from the real arm for at least one item)", float(not any(a["samples"] != b["samples"] for a, b in zip([json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["x3c__shuffled"]["file"]))], [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["x3c"]["file"]))]))))
jc = open(os.path.join(runs, tag, "judge_cmds.sh")).read()
chk("x8 judge_cmds.sh: bench judge lines for every arm plus the direct-Gemini line for the token arms (no key inside)", float("x3s_jl_t1" not in jc or "wsbench_judge_gemini.py --run runs/smk_w4_p6 --family tiny_typo --arms jlens,logitlens" not in jc or "sk-" in jc or "AIza" in jc))
sys.argv = ["x8", "--rescore", "--tag", tag]; x8.main(); r8b = json.load(open(os.path.join(runs, tag, "results.json"))); f8b = r8b["families"]["tiny_typo"]
chk("x8 --rescore on a --no-gate run: every item still scored, denominators and table fields identical, arms kept", float(f8b["n_scored"] != 5 or set(f8b["arms"]) != set(f8["arms"]) or r8b["table"]["x3c"]["tiny_typo"] != tb or f8b["arms"]["x3s_jl"]["denominators"] != f8["arms"]["x3s_jl"]["denominators"]))
# a GATED run of the same bank with the old (pre-A15) results format: gate rows without `immediate` -> rescore re-derives it
tag2 = "smk_w4_p6_gated"; shutil.rmtree(os.path.join(runs, tag2), ignore_errors=True)
sys.argv = ["x8", "--model", "tiny", "--bank-root", bank_root, "--families", "tiny_typo", "--layers", "1,3", "--p3-layers", "2", "--readers", "logitlens,x3", "--x3-carriers-kind", "identity3", "--x3-target-layer", "same", "--x3-carriers", "2", "--x3-gen", "3", "--controls", "shuffled", "--gate-tokens", "3", "--device", "cpu", "--tag", tag2, "--no-uncut-arms"]
x8.main(); r2 = json.load(open(os.path.join(runs, tag2, "results.json"))); f2 = r2["families"]["tiny_typo"]
chk("x8 gated run: n_scored == n_gated, pass_all is None (not bench-comparable), no _uncut arms with --no-uncut-arms", float(f2["n_scored"] != f2["n_gated"] or (f2["arms"] and (r2["table"]["x3_identity3"]["tiny_typo"]["pass_all"] is not None or any(a.endswith("_uncut") for a in f2["arms"])))))
if f2["arms"]:
    for r in f2["gate_rows"]: r.pop("immediate", None); r["surface"].pop("immediate", None)
    f2.pop("n_immediate", None); f2.pop("n_scored", None); f2.pop("no_gate", None); json.dump(r2, open(os.path.join(runs, tag2, "results.json"), "w"))
    sys.argv = ["x8", "--rescore", "--tag", tag2]; x8.main(); r2b = json.load(open(os.path.join(runs, tag2, "results.json"))); f2b = r2b["families"]["tiny_typo"]
    chk("x8 --rescore on a pre-A15 results file: immediate re-derived per gate row, n_immediate / denominators added, scored set = gated", float(any("immediate" not in r for r in f2b["gate_rows"]) or "n_immediate" not in f2b or f2b["n_scored"] != f2["n_gated"] or f2b["arms"]["x3_identity3"]["denominators"]["gated"]["n"] != f2["n_gated"]))
else:
    chk("x8 gated run on the tiny model gated 0 items (no rescore check possible; not a failure)", 0.0)

# ---------------- (5) E5a bench-rule rescoring on a synthetic w3-style results file ----------------
gens_a = [" Kuala Lumpur is the", " Kuala Lumpur -> Kuala", " Kuala -> Kuala; 1", " Kuala Lumpur; 19", " Kuala Lumpurian ->", " KUALA lumpur -> K", " Lumpur -> Lumpur", " Kuala Lumpur\n"]  # bench hits: 1, 2, 4, 6, 8 = 5 (not 'Kuala' alone, not 'Lumpurian', not 'Lumpur' alone)
gens_n = [" 1945 -> 1945; 3", " 1945 ->", " 19451 -> 1", " the year 1945", " 1945.5 ->", " 1945", " 2024 -> 2024", " 1945 -> 1"]  # bench: at start 1945 (1,2,6,8) + '1945.5' no + 'the year 1945' no (no answer marker) = 4
row_a = {"id": "a", "string": " Kuala Lumpur", "arm": "identity3:replace:t4", "control": "none", "vote_exact": 0, "any_exact": 1, "kind": "answer", "category": "capital", "position_rule": "last", "gens": gens_a}
row_n = {"id": "n", "string": " 1945", "arm": "identity3:replace:t4", "control": "none", "vote_exact": 1, "any_exact": 1, "kind": "answer", "category": "year", "position_rule": "last", "gens": gens_n}
row_p = {"id": "p", "string": " x", "arm": "identity3:replace:t4", "control": "none", "vote_exact": 0, "any_exact": 0, "kind": "bridge", "position_rule": "all", "profile": []}
b_a, b_n = e5a.rescore_row(row_a), e5a.rescore_row(row_n)
chk("E5a: bench rule = word-boundary match anywhere in the uncut generation (5 of 8 -> majority 1, any 1); registered rule stays as stored", float(b_a["bench_hits"] != 5 or b_a["bench_vote"] != 1 or b_a["bench_any"] != 1))
chk("E5a: numeric strings use the bench's answer-position rule (4 of 8 -> majority 0, any 1)", float(b_n["bench_hits"] != 4 or b_n["bench_vote"] != 0 or b_n["bench_any"] != 1))
chk("E5a: rows without stored generations (position rule all) are skipped and counted", float(e5a.rescore_row(row_p) is not None))
w3 = os.path.join(tmp, "w3_fake"); os.makedirs(w3); json.dump({"model": "tiny", "track": "A", "kinds": ["identity3"], "layers": {"22": {"rows": [row_a, row_n, row_p, {**row_a, "id": "c", "control": "shuffled", "gens": [" nothing"] * 8, "vote_exact": 0, "any_exact": 0}]}}}, open(os.path.join(w3, "results.json"), "w"))
o = e5a.rescore_run(os.path.join(w3, "results.json")); c = o["layers"]["22"]["identity3:replace:t4 [none]"]
chk("E5a summary: n 2 (+1 skipped), registered majority .5 / any 1, bench majority .5 / any 1, bench-only rows 0 here, control row separate", float(c["n"] != 2 or c["skipped_no_gens"] != 1 or c["vote_exact"] != 0.5 or c["bench_vote"] != 0.5 or c["bench_any"] != 1.0 or "identity3:replace:t4 [shuffled]" not in o["layers"]["22"]))
chk("E5a: the local run wrote runs/e5a_rescore/results.json + table.md with every w3_* run", float(not os.path.exists(os.path.join(runs, "e5a_rescore", "table.md")) or "w3_B_x3i3_14b_ans_L32" not in open(os.path.join(runs, "e5a_rescore", "table.md")).read()))

# ---------------- (6) the Gemini judge wrapper: protocol pieces with a stub HTTP function ----------------
chk("judge: render_bag = 'tok (score)' best first, 2 dp, ' | '-joined; tokens only without scores", float(jg.render_bag(["a", "b"], [1.0, 2.0]) != "b (2.00) | a (1.00)" or jg.render_bag(["a", "b"], None) != "a | b"))
chk("judge: cell key and fingerprint follow the bench (readouts.Cell.key, cache.fingerprint)", float(jg.cell_key("it", 16, 34) != "it__L016__p34" or len(jg.fingerprint("interp-v1", "m", "t")) != 16 or jg.fingerprint("a") == jg.fingerprint("b")))
chk("judge: the prompt / schema / model constants are the bench's", float("top-10 token readouts" not in jg.INTERP_SYSTEM or jg.INTERP_USER != "TOKEN READOUTS:\n{txt}" or jg.INTERP_SCHEMA["schema"]["required"] != ["interpretation"] or jg.DEFAULT_MODEL != "gemini-3.8-flash" or jg.MAX_TOKENS != 8000))
body = jg.request_body("x | y", "minimal")
chk("judge: request body = system instruction + user turn + JSON schema output, thinking level, no temperature", float(body["systemInstruction"]["parts"][0]["text"] != jg.INTERP_SYSTEM or body["contents"][0]["parts"][0]["text"] != "TOKEN READOUTS:\nx | y" or "temperature" in body["generationConfig"] or body["generationConfig"]["thinkingConfig"] != {"thinkingLevel": "minimal"} or "thinkingConfig" in jg.request_body("x", "none")["generationConfig"]))
chk("judge: parse_interpretation accepts a JSON object (also fenced), rejects empty / garbled", float(jg.parse_interpretation({"candidates": [{"content": {"parts": [{"text": '```json\n{"interpretation": "a dynasty"}\n```'}]}}]}) != {"interpretation": "a dynasty"} or jg.parse_interpretation({"candidates": [{"content": {"parts": [{"text": '{"interpretation": ""}'}]}}]}) is not None or jg.parse_interpretation({"candidates": []}) is not None))
calls = []
def stub_post(body):
    calls.append(body)
    if len(calls) == 1: return 400, {"error": "Invalid JSON payload: thinkingConfig.thinkingLevel unknown"}
    if len(calls) == 2: return 429, {"error": "rate"}
    return 200, {"candidates": [{"content": {"parts": [{"text": json.dumps({"interpretation": "the Aghlabid dynasty of Ifriqiya"})}]}}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}
jg.time.sleep = lambda s: None
cl = jg.Gemini("KEY", "gemini-3.8-flash", "minimal", rpm=1e9, post=stub_post); out = cl.summarize("dynasty (9.50) | kings (9.01)")
chk("judge client: a 400 on thinkingConfig drops it (recorded), a 429 is retried, the 200 is parsed; spend counted", float(out != {"interpretation": "the Aghlabid dynasty of Ifriqiya"} or cl.used["thinking_level"] != "none" or cl.spend["retries"] != 1 or cl.spend["calls"] != 1 or cl.spend["prompt_tokens"] != 10 or "thinkingConfig" in calls[-1]["generationConfig"]))
# scoring: summary as ONE more sample beside the raw tokens; a failed summary leaves the item undecided unless another layer passes
header, items = R.load_bank(os.path.join(ROOT, "data", "wsbench", "basic_readout_mt.json")); contract = R.contract_for(header, header["family"]); it0 = items[0]  # Aghlabid dynasty
toks = [" dynasty", " kings", " Arabs"]; cells = [{"id": it0["id"], "layer": l, "pos": 34, "tokens": toks, "scores": [3.0, 2.0, 1.0], "samples": None} for l in (16, 24)]  # no form of the item among the raw tokens (" Aghlabids" would be an alias hit)
key16, key24 = jg.cell_key(it0["id"], 16, 34), jg.cell_key(it0["id"], 24, 34)
s1 = jg.score(cells, [it0], contract, [16, 24], {key16: "These tokens point to the Aghlabid dynasty.", key24: "noise"})
s2 = jg.score(cells, [it0], contract, [16, 24], {key16: None, key24: "noise"}); s3 = jg.score(cells, [it0], contract, [16, 24], {key16: None, key24: "It is the Aghlabid dynasty."})
chk("judge scoring: the summary is one more sample (pass at L16 via summary_only; raw tokens alone score 0)", float(s1["pass_rate"] != 1.0 or s1["passing_layers_by_source"]["summary_only"] != 1 or s1["raw"]["pass_rate"] != 0.0 or s1["rows"][0]["earliest_layer"] != 16))
chk("judge scoring: a failed summary = undecided item (None) when no layer passes; decided when another layer passes", float(s2["rows"][0]["pass"] is not None or s2["n_items_decided"] != 0 or s3["rows"][0]["pass"] is not True or s3["n_unsummarized_items"] != 1))
chk("judge scoring: scorer version mt-regex-summarized, denominators present", float(s1["scorer_version"] != R.SUMMARIZED_SCORER_VERSION or "gated_immediate" not in s1["denominators"]))
# the CLI's --dry-run on the tiny run: prompts printed, no key needed, no network
sys.argv = ["jg", "--run", f"runs/{tag}", "--family", "tiny_typo", "--arms", "logitlens", "--dry-run", "--limit", "2", "--out", os.path.join(tmp, "jd")]
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf): jg.main()
chk("judge --dry-run on an x8 run: prints the first cell's summarizer prompt and the cell count, needs no key", float("TOKEN READOUTS:" not in buf.getvalue() or "dry run; 2 cells" not in buf.getvalue()))

# ---------------- (7) x3_patchscope.py with identity3_long + continuation (the track-B continuity line) ----------------
items = []
for i in range(6):
    prompt = f"Fact: item {i} is the"; ids = tok(prompt, return_tensors="pt").input_ids; pieces = B.greedy(ids, 2)
    items.append({"id": f"a{i}", "source": "curated", "category": "cap", "kind": "answer", "name": "", "prompt": prompt, "readout": "last_prompt_token", "string": tok.decode(pieces), "pieces": pieces, "pieces_text": [tok.decode([p]) for p in pieces], "target": tok.decode(pieces).strip(), "p1_chars": 2})
items_f = os.path.join(tmp, "items.json"); json.dump(items, open(items_f, "w")); split_f = os.path.join(tmp, "split.json"); json.dump({"seed": 0, "tuning": ["a0", "a1", "a2"], "heldout": ["a3", "a4", "a5"]}, open(split_f, "w"))
x3.load_q = lambda model, dtype, device: (tok, m, q)
tag3 = "smk_w4_x3"; shutil.rmtree(os.path.join(runs, tag3), ignore_errors=True)
sys.argv = ["x3", "--items", items_f, "--split-file", split_f, "--split", "heldout", "--layers", "2", "--target-layer", "1", "--carriers", "identity3_long,continuation", "--n-carriers", "2", "--gen", "3", "--controls", "shuffled", "--device", "cpu", "--tag", tag3]; x3.main()
r3 = json.load(open(os.path.join(runs, tag3, "results.json"))); rows3 = r3["layers"]["2"]["rows"]
arms3 = {r["arm"] for r in rows3}
chk("x3_patchscope: identity3_long and continuation run (arms :t1), overlap recorded empty, patch at the last token", float(arms3 != {"identity3_long:replace:t1", "continuation:replace:t1"} or r3["carrier_overlap"] != {"identity3_long": []} or r3["stops"]["continuation"] != []))
rc = next(r for r in rows3 if r["arm"] == "continuation:replace:t1" and r["control"] == "none")
chk("x3_patchscope continuation: gens stored uncut (3 tokens), prefix / first-piece flags present", float(any(len(g) != 3 for g in rc["gens"]) or "vote_prefix" not in rc))

for t in (tag, tag2, tag3): shutil.rmtree(os.path.join(runs, t), ignore_errors=True)
shutil.rmtree(tmp, ignore_errors=True)
print("\nSUMMARY:", sum(ok_ for *_, ok_ in res), "/", len(res), "checks passed")
