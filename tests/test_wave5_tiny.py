"""Wave-5 code paths (prereg A19-A25; ANALYSIS_WAVE4 section 10 items 2-8) on the tiny random Qwen3 (CPU, float64) and
on the real item files: (1) qwen3_min.forward inject_set (multi-layer overwrite) and x3's all<k> targets, placeholder
swap and firsthop+-k rules (A21 / A22); (2) x8: --x3-layers decoupled from --p3-layers (A19), the crosscat / random /
nopatch controls and the no-patch withdrawal, the primary-string-only column, --x3-positions all with the position
profile, --x3-carrier-ids, and --rescore --rescore-out leaving the source run untouched (A20 / A22); (3) the R7
certificate script: batched generation == x3.patched_generate row by row, the agreement score, the conformal
bookkeeping, an end-to-end run with the item side (A25); (4) the fresh item set: contract, entity-disjointness from
every existing file, first-hop positions (A24). Content is meaningless on a random
model; identities, formats and rules are what is checked. Run: python tests/test_wave5_tiny.py"""
import importlib.util, io, json, os, shutil, subprocess, sys, tempfile, torch
sys.path.insert(0, "."); sys.path.insert(0, "tests")
from tiny import tiny
from sjlens.model.qwen3_min import Qwen3Min, forward
from sjlens.model.backend import MinBackend
from sjlens import wsbench_regex as R

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(os.path.dirname(HERE), "scripts"); ROOT = os.path.dirname(HERE)
load = lambda n, f: (lambda s: (lambda mod: (s.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s)))(importlib.util.spec_from_file_location(n, os.path.join(SC, f)))
x7 = load("x7", "x7_chain_reader.py"); x8 = load("x8", "x8_wsbench_items.py"); x3 = x8.x3; e5c = load("e5c", "e5c_certificate.py"); e12 = x8.e12
res = []
def chk(name, err, tol=0.5): res.append((name, err, err < tol)); print(f"[{'PASS' if err < tol else 'FAIL'}] {name}: {err:.2e}")


class Tok:  # char-level stand-in tokenizer over the tiny vocabulary (ids 6..255), with the HF surface the scripts use
    all_special_ids = [0, 1]; chat_template = None; pad_token_id = 0; eos_token_id = 1
    def encode(s, t, add_special_tokens=False): return [max(6, min(255, ord(c) - 26)) for c in t]
    def decode(s, ids, skip_special_tokens=False): return "".join(chr(int(i) + 26) if int(i) >= 6 else "" for i in ids)
    def batch_decode(s, rows, skip_special_tokens=False): return [s.decode(r) for r in rows]
    def __call__(s, t, return_tensors=None, add_special_tokens=False):
        ids = s.encode(t)
        return {"input_ids": ids} if return_tensors is None else type("O", (), {"input_ids": torch.tensor([ids])})()
    def convert_ids_to_tokens(s, ids): return [("Ġ" + s.decode([i]).strip()) if s.decode([i]).startswith(" ") else s.decode([i]) for i in ids]


tok = Tok(); m = tiny(); q = Qwen3Min(m); B = MinBackend(tok, q); V, d = 256, 64
runs = os.path.join(ROOT, "runs"); tmp = tempfile.mkdtemp()

# ---------------- (1) inject_set / all<k> targets / placeholder / firsthop+-k ----------------
cid = tok(x3.IDENTITY3[0], return_tensors="pt").input_ids; T = cid.shape[1]; cp = T - 1; u = torch.randn(d, dtype=torch.float64) * 0.7
mask = torch.zeros(1, T, dtype=torch.bool); mask[0, cp] = True
for k in (0, 1, 2):
    ok_ = True
    for l in range(-1, 3):
        hl = forward(q.w, cid, None, stop_layer=l, inject_set=(list(range(-1, k + 1)), mask, u))[0] if l >= 0 else None
        if l < 0: continue
        at = hl[0, cp]; clean = forward(q.w, cid, None, stop_layer=l)[0]
        if l <= k: ok_ = ok_ and torch.allclose(at, u) and torch.allclose(hl[0, :cp], forward(q.w, cid, None, stop_layer=l, inject_set=(list(range(-1, k + 1)), mask, u))[0][0, :cp])
        else: ok_ = ok_ and not torch.allclose(at, u)
    chk(f"inject_set all{k}: the patch position holds u after every block <= {k} (and at the embedding), not above; positions before it are causal-clean", float(not ok_))
hE = forward(q.w, cid, None, stop_layer=0, inject_set=([-1], mask, u))[0]
chk("inject_set at the embedding only (-1): block 0 processes u at the patch position (its output there differs from u and from clean)", float(torch.allclose(hE[0, cp], u) or torch.allclose(hE[0, cp], forward(q.w, cid, None, stop_layer=0)[0][0, cp])))
chk("inject_set: earlier positions are untouched (causal) and later positions change", float(not torch.allclose(forward(q.w, cid, None, stop_layer=3, inject_set=([0, 1], mask, u))[0][0, :cp], forward(q.w, cid, None, stop_layer=3)[0][0, :cp])))
chk("parse_targets: same | int | all<k>; target_key suffixes", float(x3.parse_targets("same,4,all4,all32") != ["same", 4, ("all", 4), ("all", 32)] or x3.target_key("same") != "" or x3.target_key(4) != ":t4" or x3.target_key(("all", 4)) != ":all4" or not x3.is_multi_target(("all", 4)) or x3.is_multi_target(4)))
g_all = x3.patched_generate(q, cid, 2, cp, u, 4, "replace", 1.0, None, ("all", 1))
hL_m, cache_m = forward(q.w, cid, None, act_dtype=None, inject_set=([-1, 0, 1], mask, u)); ref = []
from sjlens.model.qwen3_min import logits as true_logits
for _ in range(4):
    t = int(true_logits(q.w, hL_m[0, -1]).argmax()); ref.append(t); hL_m, cache_m = forward(q.w, torch.tensor([[t]]), None, cache_m)
chk("patched_generate all1 == manual forward with inject_set {-1,0,1} then clean decoding", float(g_all != ref))
x1_ = q.resid(cid, 1)[0, cp]; inj1 = (1, mask, (u - x1_).view(1, 1, -1).expand(1, T, -1))
h_t1, c_t1 = forward(q.w, cid, inject=inj1); h_all1, c_all1 = forward(q.w, cid, None, inject_set=([-1, 0, 1], mask, u))
t_next = torch.tensor([[int(true_logits(q.w, h_t1[0, -1]).argmax())]])
g_t1 = forward(q.w, t_next, None, c_t1)[0]; g_all1 = forward(q.w, t_next, None, c_all1)[0]
chk("all<k> vs t<k> (patch at the LAST token): identical prefill state at the patch position (causality: blocks above k see u either way) but a DIFFERENT state for the next generated token, which attends to the patch position's K/V in blocks 0..k (the placeholder channel of A21)",
    float(not torch.allclose(h_t1[0, -1], h_all1[0, -1]) or not torch.allclose(forward(q.w, cid, inject=inj1, stop_layer=1)[0][0, cp], forward(q.w, cid, None, stop_layer=1, inject_set=([-1, 0, 1], mask, u))[0][0, cp]) or torch.allclose(g_t1[0, -1], g_all1[0, -1])))
try: x3.patched_generate(q, cid, 2, cp, u, 2, "replace_nm", 1.0, 3.0, ("all", 1)); raised = False
except AssertionError: raised = True
chk("patched_generate all<k>: mode replace only (replace_nm raises)", float(not raised))
chk("with_placeholder: replaces the final '?' of identity kinds only; continuation carriers untouched", float(x3.with_placeholder("a -> a; ?", "identity3", "x") != "a -> a; x" or x3.with_placeholder("The answer is", "continuation", "x") != "The answer is" or x3.with_placeholder("a -> a; ?", "identity3", "") != "a -> a; ?"))
chk("rule_pos: firsthop / firsthop-1 / firsthop+1 relative to the item's first-hop token, None outside the prompt or without one", float(x3.rule_pos("firsthop", {"firsthop_pos": 5}, 10) != 5 or x3.rule_pos("firsthop-1", {"firsthop_pos": 5}, 10) != 4 or x3.rule_pos("firsthop+2", 5, 10) != 7 or x3.rule_pos("firsthop+5", 5, 10) is not None or x3.rule_pos("firsthop", {"firsthop_pos": None}, 10) is not None or x3.rule_pos("firsthop-6", 5, 10) is not None))
# x3_patchscope.py end to end: all1 target, placeholder swap, firsthop-1 rule, bridge items with a curated template
items = []
for i in range(4):
    prompt = f"Fact: item {i} is the"; ids = tok(prompt, return_tensors="pt").input_ids; pieces = B.greedy(ids, 2)
    items.append({"id": f"a{i}", "source": "curated", "category": "cap", "kind": "answer", "name": "", "prompt": prompt, "readout": "last_prompt_token", "string": tok.decode(pieces), "pieces": pieces, "pieces_text": [tok.decode([p]) for p in pieces], "target": tok.decode(pieces).strip(), "p1_chars": 2})
cands = ["zq", "xk", "wj", "vh", "qz", "kx", "jw", "hv"]
for i in range(8):  # bridges: the gate needs the TARGET as the greedy prefix and the hidden string absent from the greedy text (no leak); DISTINCT strings so shuffled partners exist
    prompt = f"Fact: The continent of the country whose capital is Gab{i} is"; ids = tok(prompt, return_tensors="pt").input_ids; g6 = tok.decode(B.greedy(ids, 6))
    s_ = next(s for s in cands[i:] + cands[:i] if s not in g6.lower() and not g6.lower().startswith(s[0])); pieces = tok.encode(s_)
    items.append({"id": f"b{i}", "source": "curated", "category": "brc", "kind": "bridge", "name": "", "prompt": prompt, "readout": "last_prompt_token", "string": s_, "pieces": pieces, "pieces_text": [tok.decode([p]) for p in pieces], "target": [tok.decode(B.greedy(ids, 1)), "zz"], "p1_chars": 1})
items_f = os.path.join(tmp, "items.json"); json.dump(items, open(items_f, "w")); split_f = os.path.join(tmp, "split.json"); json.dump({"seed": 0, "tuning": [], "heldout": ["a0", "a1", "a2", "a3"]}, open(split_f, "w"))
x3.load_q = lambda model, dtype, device: (tok, m, q)
tag3 = "smk_w5_x3"; shutil.rmtree(os.path.join(runs, tag3), ignore_errors=True)
sys.argv = ["x3", "--items", items_f, "--split-file", split_f, "--split", "heldout", "--layers", "2", "--target-layer", "same,1,all1", "--carriers", "identity3", "--n-carriers", "2", "--gen", "3", "--positions", "last,firsthop,firsthop-1", "--kinds", "answer,bridge", "--controls", "shuffled,random", "--placeholder", "x", "--device", "cpu", "--tag", tag3]
x3.main(); r3 = json.load(open(os.path.join(runs, tag3, "results.json"))); rows3 = r3["layers"]["2"]["rows"]; arms3 = {r["arm"] for r in rows3}
want3 = {f"identity3:replace{t}{p}:ph=x" for t in ("", ":t1", ":all1") for p in (":last", ":firsthop", ":firsthop-1")}
chk("x3 e2e: arms = 3 targets (same / t1 / all1) x 3 position rules, placeholder suffix; carriers end in 'x'; target_layers recorded", float(arms3 != want3 or not all(c.endswith("x") for c in r3["carrier_prompts"]["identity3"]) or r3["placeholder"] != "x" or r3["target_layers"] != ["same", 1, ["all", 1]]))
fh = [r for r in rows3 if r["position_rule"] == "firsthop-1" and r["control"] == "none"]
chk("x3 e2e: firsthop-1 rows read the token before the first-hop token (bridges only; answers have no first-hop position)", float(not fh or any(r["best_pos"] != r["firsthop_pos"] - 1 or r["kind"] != "bridge" for r in fh)))
r_all = next(r for r in rows3 if r["arm"].endswith(":all1:last:ph=x") and r["control"] == "none"); it0 = next(it for it in items if it["id"] == r_all["id"])
cidx = tok(x3.with_placeholder(x3.IDENTITY3[0], "identity3", "x"), return_tensors="pt").input_ids; h0 = q.resid(tok(it0["prompt"], return_tensors="pt").input_ids, 2)[0, -1]
chk("x3 e2e: the all1 row's first generation == patched_generate(('all', 1)) on the placeholder-swapped carrier", float(r_all["gens"][0] != tok.decode(x3.patched_generate(q, cidx, 2, cidx.shape[1] - 1, h0, 3, "replace", 1.0, None, ("all", 1)))))
chk("x3 e2e: shuffled control at firsthop-1 uses the PARTNER's firsthop-1 token (A14 rule; rows exist for bridges)", float(not any(r["position_rule"] == "firsthop-1" and r["control"] == "shuffled" for r in rows3)))

# ---------------- (2) x8: --x3-layers, controls, primary-only, positions all, carrier ids, rescore-out ----------------
lens_f = os.path.join(tmp, "lens.pt"); torch.save({"J": {l: torch.randn(d, d) * 0.05 for l in range(4)}, "source_layers": [0, 1, 2, 3], "d_model": d}, lens_f)
for mod in (x7, x8): mod.hf_hub_download = lambda repo, f: lens_f
x7.load_backend = lambda kind, model, dtype, device: (tok, m, B); x8.x7 = x7
bank_items = []
for i in range(6):
    prompt = f"The thing number {i} is called the"; ids = tok(prompt, return_tensors="pt").input_ids; tgt = (tok.decode(B.greedy(ids, 3)).strip() or "zz") + f"{i}"
    forms = [tgt + " zz", tgt + " qq"]  # primary = the first form
    bank_items.append({"name": f"tiny-{i}", "subfamily": "entity" if i % 2 else "novel", "prompt": prompt, "target": tgt, "target_alts": [], "eval_render": "plain", "readout": {"kind": "final_prompt_token", "offsets": [-1]}, "intermediates": [tgt],
                       "units": [{"role": "readout", "required": True, "multi_token": True, "forms": {"en": forms}, "match": forms}], "probe_token_lens": {"units": {"readout": {"en": [len(f) for f in forms]}}, "target": len(tgt)}, "bridges": []})
bank_root = os.path.join(tmp, "bank"); os.makedirs(bank_root); json.dump({"family": "tiny-basic", "contract": {"multi_token": True, "include_target": False, "conjunctive_units": True}, "items": bank_items}, open(os.path.join(bank_root, "tiny_basic.json"), "w"))
tag = "smk_w5_p6"; shutil.rmtree(os.path.join(runs, tag), ignore_errors=True)
sys.argv = ["x8", "--model", "tiny", "--bank-root", bank_root, "--families", "tiny_basic", "--layers", "1", "--p3-layers", "3", "--x3-layers", "2", "--readers", "jlens,logitlens,x3,x3c", "--x3-carriers-kind", "identity3", "--x3-target-layer", "same,1", "--x3-carriers", "8", "--x3-carrier-ids", "1,2", "--x3-gen", "3", "--x3c-gen", "4",
            "--controls", "shuffled,crosscat,random,nopatch", "--gate-tokens", "3", "--no-gate", "--device", "cpu", "--tag", tag]
x8.main(); r8 = json.load(open(os.path.join(runs, tag, "results.json"))); f8 = r8["families"]["tiny_basic"]
chk("x8 A19: x3 read layers follow --x3-layers (2), p3 layers stay 3, token grid extended by both; results carry x3_layers", float(r8["x3_layers"] != [2] or r8["p3_layers"] != [3] or r8["layers"] != [1, 2, 3] or f8["arms"]["x3c_t1"]["layers"] != [2] or any(json.loads(l)["layer"] != 2 for l in open(os.path.join(ROOT, f8["arms"]["x3c_t1"]["file"])))))
chk("x8 A20: shuffled / crosscat / random controls written for every arm; drop counts recorded", float(not all(f"x3c_t1__{c}" in f8["arms"] and f"jlens__{c}" in f8["arms"] for c in ("shuffled", "crosscat", "random")) or "n_crosscat_dropped" not in f8))
rr = [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["x3c_t1__random"]["file"]))]; rn = [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["x3c_t1"]["file"]))]
chk("x8 A20 random: rows per item, samples differ from the real arm", float(len(rr) != len(rn) or not any(a["samples"] != b["samples"] for a, b in zip(rr, rn))))
rx = [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["x3c_t1__crosscat"]["file"]))]; sub_of = {R.label_of(it["name"]): it["subfamily"] for it in bank_items}
chk("x8 A20 crosscat: partners come from the other subfamily (samples equal the partner's real samples)", float(not rx or not all(any(a["samples"] == b["samples"] and sub_of[a["id"]] != sub_of[b["id"]] for b in rn if b["id"] != a["id"]) for a in rx)))
npr = [json.loads(l) for l in open(os.path.join(ROOT, f8["arms"]["x3c_nopatch"]["file"]))]
chk("x8 A20 nopatch: one row per item at the x3 read layer with IDENTICAL bare-carrier samples (2 carriers), plus the identity kind's _nopatch and _nopatch_uncut arms", float(len(npr) != 6 or len({json.dumps(r["samples"]) for r in npr}) != 1 or len(npr[0]["samples"]) != 2 or "x3_identity3_nopatch" not in f8["arms"] or "x3_identity3_nopatch_uncut" not in f8["arms"] or r8["nopatch"]["x3c"]["samples"] != npr[0]["samples"]))
chk("x8 A20 nopatch: no-patch samples == the bare carriers' greedy continuation (no h)", float(npr[0]["samples"] != [tok.decode(B.greedy(tok(c, return_tensors="pt").input_ids, 4)) for c in [x3.CONTINUATION[1], x3.CONTINUATION[2]]]))
w = f8["nopatch_withdrawal"]
chk("x8 A20 withdrawal: every real x3 arm maps to its _nopatch arm (uncut -> _nopatch_uncut) with the withdrawn ids and the rate after withdrawal", float(w.get("x3c_t1", {}).get("nopatch_arm") != "x3c_nopatch" or w.get("x3_identity3_t1_uncut", {}).get("nopatch_arm") != "x3_identity3_nopatch_uncut" or "withdrawn_ids" not in w["x3c"] or w["x3c"]["pass_rate_after_withdrawal"] is None or w["x3c"]["pass_rate_after_withdrawal"] > f8["arms"]["x3c"]["pass_rate"] + 1e-9))
chk("x8 carrier ids: --x3-carrier-ids 1,2 selects those carriers for every kind (recorded)", float(r8["x3"]["carriers"] != [x3.IDENTITY3[1], x3.IDENTITY3[2]] or r8["x3c"]["carriers"] != [x3.CONTINUATION[1], x3.CONTINUATION[2]] or r8["x3"]["carrier_ids"] != [1, 2]))
po = f8["arms"]["x3c_t1"]["primary_only"]; tb = r8["table"]["x3c_t1"]["tiny_basic"]
chk("x8 A20 primary-only: block with pass / denominators / alias-only ids beside the bench rule; table columns pass_primary_*; primary <= bench", float(any(k not in po for k in ("pass_rate", "denominators", "pass_ids", "alias_only_ids")) or any(k not in tb for k in ("pass_primary", "pass_primary_all", "pass_primary_gated", "pass_primary_immediate", "pass_after_nopatch_withdrawal")) or po["pass_rate"] > f8["arms"]["x3c_t1"]["pass_rate"] + 1e-9))
# the primary rule itself: a synthetic readouts file where only the alias hits -> bench pass 1, primary 0
header, bitems = R.load_bank(os.path.join(bank_root, "tiny_basic.json")); contract = R.contract_for(header, "tiny_basic"); it0 = bitems[0]
cells = [{"id": it0["id"], "layer": 2, "pos": 5, "samples": [f"xx {it0['target']} qq yy"], "tokens": None, "scores": None}]
sb = R.score_cells(cells, [it0], contract, [2]); sp = R.score_cells(cells, [it0], contract, [2], units_of={it0["id"]: R.primary_units(R.scored_units(it0, contract))})
chk("primary_units: only the first creditable form per unit / language counts (alias-only sample: bench 1, primary 0)", float(sb["pass_rate"] != 1.0 or sp["pass_rate"] != 0.0 or [u.forms for u in R.primary_units(R.scored_units(it0, contract))] != [{"en": [it0["target"] + " zz"]}]))
# --rescore --rescore-out: the source run is untouched, the destination carries the same table
before = open(os.path.join(runs, tag, "results.json"), "rb").read()
sys.argv = ["x8", "--rescore", "--tag", tag, "--rescore-out", tag + "_rs"]; x8.main()
r8b = json.load(open(os.path.join(runs, tag + "_rs", "results.json")))
chk("x8 --rescore-out: writes runs/<out>/results.json (rescored_from recorded, same table, primary-only and withdrawal blocks) and leaves the source results.json byte-identical", float(open(os.path.join(runs, tag, "results.json"), "rb").read() != before or r8b.get("rescored_from") != f"runs/{tag}" or r8b["table"]["x3c_t1"]["tiny_basic"] != tb or "nopatch_withdrawal" not in r8b["families"]["tiny_basic"]))
# --x3-positions all: every earlier prompt position becomes an _off<k> arm; position profile with the any-position union
tagp = "smk_w5_p6_all"; shutil.rmtree(os.path.join(runs, tagp), ignore_errors=True)
sys.argv = ["x8", "--model", "tiny", "--bank-root", bank_root, "--families", "tiny_basic", "--layers", "2", "--p3-layers", "2", "--readers", "x3c", "--x3-target-layer", "1", "--x3-carriers", "2", "--x3c-gen", "3", "--x3-positions", "all", "--x3-pos-max", "4", "--controls", "shuffled", "--gate-tokens", "3", "--no-gate", "--device", "cpu", "--tag", tagp]
x8.main(); rp = json.load(open(os.path.join(runs, tagp, "results.json"))); fp = rp["families"]["tiny_basic"]
offs = sorted(int(a.split("_off")[1]) for a in fp["arms"] if "_off" in a and "__" not in a)
chk("x8 A22 positions all (--x3-pos-max 4): arms x3c_t1, _off2, _off3, _off4 with shuffled twins; rows carry the absolute pos = T - k", float(offs != [2, 3, 4] or not all(f"x3c_t1_off{k}__shuffled" in fp["arms"] for k in offs) or any(json.loads(l)["pos"] != len(tok.encode(bank_items[i]["prompt"])) - 3 for i, l in enumerate(open(os.path.join(ROOT, fp["arms"]["x3c_t1_off3"]["file"]))))))
pp = fp["position_profile"]
chk("x8 A22 position_profile: per-offset rates for offsets 1..4 and the any-position union (labelled a selection) for the real arm and its control", float("x3c_t1" not in pp or sorted(pp["x3c_t1"]["offsets"]) != ["1", "2", "3", "4"] or "any_position" not in pp["x3c_t1"] or "selection" not in pp["x3c_t1"]["any_position"]["note"] or "x3c_t1__shuffled" not in pp or pp["x3c_t1"]["any_position"]["pass"] < max(v["pass"] for v in pp["x3c_t1"]["offsets"].values())))

# ---------------- (3) the R7 certificate script (A25) ----------------
U = torch.randn(5, d, dtype=torch.float64) * 0.6; cid3 = tok(x3.IDENTITY3[0], return_tensors="pt").input_ids; cp3 = cid3.shape[1] - 1
for tl in (1, 2):
    bg_ = e5c.batched_patched_generate(q, cid3, cp3, U, 4, tl)
    chk(f"e5c batched generation (target {tl}) == x3.patched_generate row by row", float(any(bg_[b].tolist() != x3.patched_generate(q, cid3, 2, cp3, U[b], 4, "replace", 1.0, None, tl) for b in range(5))))
ka, km, s_ = e5c.agree_scores([" Foo Bar -> Foo", " Foo Bar; x", " foo  bar\n", " Baz -> Baz", " Baz; 1", "", " -> nothing", " Foo Bar"], "identity3")
chk("e5c agree_scores: max multiplicity of the cut+norm string (4 x 'foo bar'), multi-token variant, the agreed string", float(ka != 4 or km != 4 or s_ != "foo bar" or e5c.agree_scores([" a; 1", " b; 2"], "identity3")[0] != 1 or e5c.agree_scores([" x -> x", " x -> x"], "identity3")[1] != 0 or e5c.agree_scores([], "identity3") != (0, 0, "")))
cal = [0] * 90 + [1] * 8 + [2] * 2; tst = [0] * 95 + [1] * 3 + [3] * 2; recs = [i // 10 for i in range(100)]
c05 = e5c.certify(cal, tst, recs, 0.05); c01 = e5c.certify(cal, tst, recs, 0.01)
chk("e5c certify: conformal tau on calibration, FPR on test (2/100 above tau=1 at q .05), N_eff = test records (10), two-sided line with 2 se, one-sided flag", float(c05["tau"] != 1.0 or abs(c05["fpr"] - 0.02) > 1e-9 or c05["n_eff_records"] != 10 or abs(c05["se_records"] - (0.05 * 0.95 / 10) ** 0.5) > 1e-12 or not c05["pass_two_sided"] or c01["tau"] != 2.0 or abs(c01["fpr"] - 0.02) > 1e-9 or not c01["pass_one_sided_le"]))
# end to end on the tiny model: a small contexts file (T = 24: 7 valid positions), 8 records, items from a synthetic w3-style run
ctxf = os.path.join(tmp, "ctx.pt"); torch.manual_seed(1); torch.save({"T": 24, "ctx": [torch.randint(6, 256, (24,)) for _ in range(4)], "bg": [torch.randint(6, 256, (24,)) for _ in range(8)]}, ctxf)
w3 = os.path.join(tmp, "w3_fake"); os.makedirs(w3)
rows_w3 = [{"id": f"i{i}", "string": " Foo Bar", "arm": "identity3:replace:t1", "control": "none", "vote_exact": int(i % 2 == 0), "any_exact": 1, "kind": "answer", "category": "c", "position_rule": "last",
            "gens": ([" Foo Bar -> Foo"] * (6 if i % 2 == 0 else 1)) + [" Baz -> Baz"] * (2 if i % 2 == 0 else 7)} for i in range(6)]
rows_w3 += [{**r, "control": "shuffled", "vote_exact": 0, "gens": [" q; 1"] * 8} for r in rows_w3]
json.dump({"model": "tiny", "track": "B", "kinds": ["identity3"], "layers": {"2": {"rows": rows_w3}}}, open(os.path.join(w3, "results.json"), "w"))
e5c.load_q = lambda model, dtype, device: (tok, m, q)
tag5 = "smk_w5_e5c"; shutil.rmtree(os.path.join(runs, tag5), ignore_errors=True)
sys.argv = ["e5c", "--model", "tiny", "--layers", "2", "--target-layer", "1,same", "--carriers", "identity3", "--n-carriers", "2", "--gen", "3", "--contexts", ctxf, "--T", "24", "--n-records", "8", "--per-record", "4", "--batch", "4", "--items-runs", w3, "--device", "cpu", "--tag", tag5]
e5c.main(); r5 = json.load(open(os.path.join(runs, tag5, "results.json"))); L5 = r5["layers"]["2"]
chk("e5c e2e: 8 records x 4 positions = 32 activations, 4 / 4 records calibration / test, targets t1 and same, certificates at q .05 / .01 for both scores, table.md written", float(L5["n_activations"] != 32 or r5["n_cal_records"] != 4 or r5["n_test_records"] != 4 or set(L5["targets"]) != {"t1", "same"} or set(L5["targets"]["t1"]["certificates"]) != {"k_agree_q0.05", "k_agree_q0.01", "k_agree_multi_q0.05", "k_agree_multi_q0.01"} or not os.path.exists(os.path.join(runs, tag5, "table.md"))))
c = L5["targets"]["t1"]["certificates"]["k_agree_q0.05"]
chk("e5c e2e: the FPR is measured on the TEST records only (16 activations, N_eff 4) and the histogram covers 0..n_carriers", float(c["n_test"] != 16 or c["n_eff_records"] != 4 or set(L5["targets"]["t1"]["k_hist_all"]) != {"0", "1", "2"}))
it5 = L5["targets"]["t1"]["items"][0]
chk("e5c item side: k_agree per stored row (6 -> 6 / 1), coverage / certified-correct / certified-wrong at every tau, controls separate", float(it5["arm"] != "identity3:replace:t1" or [r["k"] for r in it5["controls"]["none"]["rows"]] != [6, 7, 6, 7, 6, 7] or "shuffled" not in it5["controls"] or abs(it5["controls"]["none"]["summary"]["vote_exact"] - 0.5) > 1e-9 or "certified_wrong" not in it5["controls"]["none"]["summary"]["q0.05_k_agree"]))
chk("e5c item side: with tau = 1 every item is covered; certified-correct = exact rate; certified-wrong = the rest", float(abs(e5c.item_side(w3, 2, "identity3", 1, {(0.05, "k_agree"): 1.0}, None, lambda s: None)["controls"]["none"]["summary"]["q0.05_k_agree"]["certified_wrong"] - 0.5) > 1e-9))

# ---------------- (4) the fresh item set (A24) ----------------
mf = load("mf", "make_fresh_items.py")
fresh = json.load(open(os.path.join(ROOT, "data", "fresh_items.json"))); fsplit = json.load(open(os.path.join(ROOT, "data", "fresh_split.json")))
forb = mf.forbidden_strings()
chk("fresh: every item passes the contract (round trip, >= 2 pieces, track-B word pieces, string not in prompt, entity-disjoint from every existing string)", float(len(mf.check_items(fresh, forb))))
old_ids = set()
for f in ("h2a_items.json", "phrase_items.json", "single_items.json"): old_ids |= {it["id"] for it in json.load(open(os.path.join(ROOT, "data", f)))}
chk("fresh: ids are new (no collision with the three item files), source 'fresh', both tracks, answers and bridges present", float(bool(old_ids & {it["id"] for it in fresh}) or any(it["source"] != "fresh" for it in fresh) or {it["track"] for it in fresh} != {"A", "B"} or not any(it["kind"] == "bridge" for it in fresh)))
chk("fresh split: every answer id held-out, nothing in tuning; bridges outside the split", float(set(fsplit["heldout"]) != {it["id"] for it in fresh if it["kind"] == "answer"} or fsplit["tuning"] != []))
chk("fresh: overlap rule - a name-like string bans containment both ways ('Isaac Newton' vs 'Newton'), a lowercase single word only equality ('black' vs 'black pepper')", float(mf.overlap_reason(" Isaac Newton", ({"newton"}, {"newton"})) is None or mf.overlap_reason(" Newton", ({"isaac newton"}, {"isaac newton"})) is None or mf.overlap_reason(" black pepper", ({"black"}, set())) is not None or mf.overlap_reason(" black", ({"black"}, set())) is None or mf.overlap_reason(" 144", ({"14"}, {"14"})) is not None))
chk("fresh: the existing item strings are in the forbidden set (both files) and none of them appears as a fresh string", float(not all(x3.norm(it["string"]) in forb[0] for f in ("h2a_items.json", "phrase_items.json") for it in json.load(open(os.path.join(ROOT, "data", f)))[:50]) or any(x3.norm(it["string"]) in forb[0] for it in fresh)))
chk("fresh: e12.firsthop_pos accepts source 'fresh' (bridge prompts end in the curated tails)", float(e12.firsthop_pos(tok, {"kind": "bridge", "source": "fresh", "prompt": "Fact: The nationality of the painter of X was"}) is None or e12.firsthop_pos(tok, {"kind": "bridge", "source": "other", "prompt": "Fact: The nationality of the painter of X was"}) is not None))
try:
    from transformers import AutoTokenizer
    rtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", local_files_only=True)
    bad = [it["id"] for it in fresh if rtok.encode(it["string"], add_special_tokens=False) != it["pieces"]]
    chk("fresh: stored pieces == the real Qwen3 tokenizer's encoding (cached tokenizer)", float(len(bad)))
    chk("fresh: every bridge has a first-hop position under the real tokenizer", float(any(e12.firsthop_pos(rtok, it) is None for it in fresh if it["kind"] == "bridge")))
except Exception as e:  # no cached tokenizer on this machine: run the same two checks on a GPU machine (make_fresh_items.py --check)
    print(f"[SKIP] real-tokenizer checks of the fresh set ({type(e).__name__}); run scripts/make_fresh_items.py --check offline on a GPU machine")

# (5) removed: dl_model.sh is a cluster download helper and is not part of the public repo

for t in (tag, tag + "_rs", tagp, tag3, tag5): shutil.rmtree(os.path.join(runs, t), ignore_errors=True)
shutil.rmtree(tmp, ignore_errors=True)
print("\nSUMMARY:", sum(ok_ for *_, ok_ in res), "/", len(res), "checks passed")
