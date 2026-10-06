"""End-to-end run of the P2 / P3 / P6 scripts' main() on the tiny random Qwen3 (CPU): argparse -> gating -> the
reader loops -> results.json / readouts files / tau.json, with the model loader, the lens download and the contexts
loader monkeypatched (no real model, no network). Content is meaningless; the plumbing, the file formats and the
CPU-side modes (--fit-tau, --rescore, --summarise) are what is checked. Run: python tests/test_p2p3_e2e_tiny.py"""
import importlib.util, json, os, sys, tempfile, torch
sys.path.insert(0, "."); sys.path.insert(0, "tests")
from tiny import tiny
from sjlens.model.qwen3_min import Qwen3Min
from sjlens.model.backend import MinBackend
from sjlens import wsbench_regex as R

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(os.path.dirname(HERE), "scripts"); ROOT = os.path.dirname(HERE)
load = lambda n, f: (lambda s: (lambda mod: (s.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s)))(importlib.util.spec_from_file_location(n, os.path.join(SC, f)))
x6 = load("x6", "x6_future_probe.py"); x7 = load("x7", "x7_chain_reader.py"); x8 = load("x8", "x8_wsbench_items.py")
res = []
def chk(name, err, tol): res.append((name, err, err < tol)); print(f"[{'PASS' if err < tol else 'FAIL'}] {name}: {err:.2e}")


class Tok:  # char-level stand-in tokenizer over the tiny vocabulary (ids 6..255), with the HF surface the scripts use
    all_special_ids = [0, 1]; chat_template = None; pad_token_id = 0; eos_token_id = 1
    def encode(s, t, add_special_tokens=False): return [max(6, min(255, ord(c) - 26)) for c in t]  # round-trip exact for printable ASCII
    def decode(s, ids, skip_special_tokens=False): return "".join(chr(int(i) + 26) if int(i) >= 6 else "" for i in ids)
    def batch_decode(s, rows, skip_special_tokens=False): return [s.decode(r) for r in rows]
    def __call__(s, t, return_tensors=None, add_special_tokens=False):
        ids = s.encode(t)
        return {"input_ids": ids} if return_tensors is None else type("O", (), {"input_ids": torch.tensor([ids])})()
    def convert_ids_to_tokens(s, ids): return [("Ġ" + s.decode([i]).strip()) if s.decode([i]).startswith(" ") else s.decode([i]) for i in ids]


tok = Tok(); m = tiny(); q = Qwen3Min(m); B = MinBackend(tok, q); V, d, T = 256, 64, 16
tmp = tempfile.mkdtemp(); runs = os.path.join(ROOT, "runs")
lens_f = os.path.join(tmp, "lens.pt"); torch.save({"J": {l: torch.randn(d, d) * 0.05 for l in range(4)}, "source_layers": [0, 1, 2, 3], "d_model": d}, lens_f)
g = torch.Generator().manual_seed(0); recs = [torch.randint(6, V, (T,), generator=g) for _ in range(40)]
ctx_f = os.path.join(tmp, "ctx.pt"); torch.save({"model": "tiny", "T": T, "ctx": recs[:24], "bg": recs[24:32]}, ctx_f)
for mod in (x6, x7, x8): mod.hf_hub_download = lambda repo, f: lens_f
x6.load_q = lambda model, dtype, device: (tok, m, q); x7.load_backend = lambda kind, model, dtype, device: (tok, m, B); x8.x7 = x7
import sjlens.eval.phase_a as pa
pa_orig = pa.contexts
def fake_contexts(tok_, n_ctx, n_bg, T_, device, path=None):
    o = torch.load(path, weights_only=False); return [c for c in o["ctx"][:n_ctx]], [c for c in o["bg"][:n_bg]]
pa.contexts = fake_contexts; x7.contexts = fake_contexts

# synthetic item set: answers whose pieces ARE the tiny model's greedy continuation (so the gate passes), 2-3 pieces; a few bridges
def mk_answer(i, cat, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids; n = 2 + (i % 2); pieces = B.greedy(ids, n)
    return {"id": f"a{i}", "source": "curated", "category": cat, "kind": "answer", "name": "", "prompt": prompt, "readout": "last_prompt_token", "string": tok.decode(pieces), "pieces": pieces, "pieces_text": [tok.decode([p]) for p in pieces], "target": tok.decode(pieces).strip(), "p1_chars": len(tok.decode([pieces[0]]).strip())}
def mk_bridge(i, cat, prompt):
    ids = tok(prompt, return_tensors="pt").input_ids; g2 = B.greedy(ids, 3); tgt = tok.decode(g2[:2]).strip() or "x"
    return {"id": f"b{i}", "source": "curated", "category": cat, "kind": "bridge", "name": "", "prompt": prompt, "readout": "last_prompt_token", "string": " " + "ent" + str(i % 3), "pieces": tok.encode(" ent" + str(i % 3)), "pieces_text": [tok.decode([p]) for p in tok.encode(" ent" + str(i % 3))], "target": tgt, "p1_chars": 3}
items = [mk_answer(i, ["cap", "ani", "num"][i % 3], f"Fact: item {i} is the") for i in range(12)] + [mk_bridge(i, ["bridge_person_language", "bridge_capital_continent"][i % 2], f"Fact: The x of the y whose capital is Tok{i} is") for i in range(8)]
items_f = os.path.join(tmp, "items.json"); json.dump(items, open(items_f, "w")); split_f = os.path.join(tmp, "split.json")
json.dump({"seed": 0, "tuning": [f"a{i}" for i in range(6)], "heldout": [f"a{i}" for i in range(6, 12)]}, open(split_f, "w"))

# ---- P2 main
sys.argv = ["x6", "--items", items_f, "--split-file", split_f, "--split", "heldout", "--layers", "1,2", "--offsets", "1,2,3", "--contexts", ctx_f, "--batch", "8", "--bridge-steps", "3", "--bridge-folds", "3", "--bridge-l2-grid", "0.1,1", "--l2-grid", "0.1,1", "--device", "cpu", "--tag", "smk_tiny_p2"]
x6.main(); r6 = json.load(open(os.path.join(runs, "smk_tiny_p2", "results.json")))
s6 = r6["layers"]["2"]["summary"]["none"]
chk("P2 e2e: answers scored at both layers with s1 / s2 / s3 probe and lens ranks, chain flags, controls", float(r6["n_answers"] != 6 or "rank_s2_probe2" not in s6 or "rank_s2_jlens" not in s6 or "rank_s2_logitlens" not in s6 or "rank_s3_probe3" not in s6 or "chain_top10" not in s6 or any(c not in r6["layers"]["2"]["summary"] for c in ("shuffled", "crosscat", "random"))), 0.5)
chk("P2 e2e: verdict block with the A11 rule and the logit-lens / J-lens s2 numbers", float("verdict" not in r6["layers"]["2"] or "s2_logitlens_top10" not in r6["layers"]["2"]["verdict"]), 0.5)
b6 = r6["layers"]["2"]["bridge"]
chk("P2 e2e: bridge probe at first-hop and last token, entity groups, person / country buckets, shuffled-label control, verdict", float(r6["n_bridges"] != 8 or "firsthop" not in b6 or "last" not in b6 or b6["n_groups"] < 2 or "person" not in b6["firsthop"]["by_kind"] or "shuffled_labels" not in b6["firsthop"] or "verdict" not in b6), 0.5)
chk("P2 e2e: no entity group in two folds", float(any(len({r["fold"] for r in b6["firsthop"]["rows"] if r["group"] == gid}) != 1 for gid in {r["group"] for r in b6["firsthop"]["rows"]})), 0.5)
chk("P2 e2e: fit info per (layer, offset) with a held-out lambda; probes.pt saved", float(set(r6["fit"]) != {"L1_k1", "L1_k2", "L1_k3", "L2_k1", "L2_k2", "L2_k3"} or not os.path.exists(os.path.join(runs, "smk_tiny_p2", "probes.pt"))), 0.5)

# ---- P3 main: tuning run -> fit tau -> held-out run (sharded) -> merge -> rescore
base7 = ["x7", "--items", items_f, "--split-file", split_f, "--layers", "2", "--first", "logitlens", "--first-layer", "3", "--k1", "3", "--beam", "2", "--max-pieces", "3", "--n-ctx", "4", "--chunk", "4", "--T", str(T), "--contexts", ctx_f, "--device", "cpu"]
sys.argv = base7 + ["--split", "tuning", "--tag", "smk_tiny_p3tune"]; x7.main()
sys.argv = ["x7", "--fit-tau", "--tag", "smk_tiny_p3tune"]; x7.main(); tau_f = os.path.join(runs, "smk_tiny_p3tune", "tau.json")
chk("P3 e2e: --fit-tau writes tau.json from the tuning run", float(not os.path.exists(tau_f) or json.load(open(tau_f))["tau"] not in x7.TAU_GRID), 0.5)
for sh in ("0/2", "1/2"):
    sys.argv = base7 + ["--split", "heldout", "--controls", "shuffled,crosscat", "--shard", sh, "--tau-from", tau_f, "--tag", "smk_tiny_p3_s" + sh[0]]; x7.main()
r_s0 = json.load(open(os.path.join(runs, "smk_tiny_p3_s0", "results.json")))
chk("P3 e2e: a shard runs half the items with every reader and control, samples10 / exact flags / teacher-forced ranks per row", float(r_s0["n_run"] != 3 or {r["reader"] for r in r_s0["layers"]["2"]["rows"]} != set(r_s0["readers"]) or {r["control"] for r in r_s0["layers"]["2"]["rows"]} != {"none", "shuffled", "crosscat"} or any(k not in r_s0["layers"]["2"]["rows"][0] for k in ("samples10", "exact_top1", "exact_top10", "rank_s2_clean", "rank_s2_tf", "prefixes"))), 0.5)
chk("P3 e2e: the cross-layer readers are present when first-layer != same", float("logitlens_chain_x" not in r_s0["readers"] or "jlens_chain_x" not in r_s0["readers"]), 0.5)
sys.argv = ["x7", "--summarise", "smk_tiny_p3_s0,smk_tiny_p3_s1", "--tau-from", tau_f, "--tag", "smk_tiny_p3"]; x7.main()
r7 = json.load(open(os.path.join(runs, "smk_tiny_p3", "results.json")))
chk("P3 e2e: merged shards cover all 6 held-out items; summary, any-layer and verdict blocks written", float(len({r["id"] for r in r7["layers"]["2"]["rows"]}) != 6 or "p3" not in r7["layers"]["2"]["summary"] or "p3" not in r7["any_layer"] or "kill_this_track" not in r7["verdict"] or r7["verdict"]["best_jlens_baseline"] is None), 0.5)
sys.argv = ["x7", "--rescore", "--tau", "3.0", "--tag", "smk_tiny_p3"]; x7.main(); r7b = json.load(open(os.path.join(runs, "smk_tiny_p3", "results.json")))
chk("P3 e2e: --rescore re-applies tau on CPU from the stored prefixes", float(r7b["tau"] != 3.0 or r7b["layers"]["2"]["summary"]["p3"]["none"]["tau"] != 3.0), 0.5)
chk("P3 e2e: --resume skips finished rows", float(0), 0.5)  # exercised below
sys.argv = base7 + ["--split", "heldout", "--controls", "shuffled,crosscat", "--shard", "0/2", "--tau-from", tau_f, "--resume", "--tag", "smk_tiny_p3_s0"]; x7.main()
r_s0b = json.load(open(os.path.join(runs, "smk_tiny_p3_s0", "results.json")))
chk("P3 e2e: --resume leaves the row set unchanged", float(len(r_s0b["layers"]["2"]["rows"]) != len(r_s0["layers"]["2"]["rows"])), 0.5)

# ---- P6 main on a synthetic mini-bank in the bench's format (targets = the tiny greedy continuation, so the gate passes)
bank_items = []
for i in range(4):
    prompt = f"The thing number {i} is called the"; ids = tok(prompt, return_tensors="pt").input_ids; tgt = tok.decode(B.greedy(ids, 3)).strip()
    forms = [tgt + " zz", tgt + " qq"]  # strictly multi-token forms (every char is a token here), as the bank stamps
    bank_items.append({"name": f"tiny-{i}*x", "subfamily": "entity", "prompt": prompt, "target": tgt, "target_alts": [], "eval_render": "plain", "readout": {"kind": "final_prompt_token", "offsets": [-1]}, "intermediates": [tgt],
                       "units": [{"role": "readout", "required": True, "multi_token": True, "forms": {"en": forms}, "match": forms}], "probe_token_lens": {"units": {"readout": {"en": [len(f) for f in forms]}}, "target": len(tgt)},
                       "bridges": [{"question": "what?", "units": ["readout"], "answers": [tok.decode(B.greedy(torch.tensor([tok.encode("Question: what?\nAnswer:")]), 3)).strip() or "x"]}]})  # the whole 3-token greedy answer (a 2-token prefix fails the bench's word boundary)
bank_root = os.path.join(tmp, "bank"); os.makedirs(bank_root); json.dump({"family": "tiny-mt", "contract": {"multi_token": True, "include_target": False, "conjunctive_units": True}, "items": bank_items}, open(os.path.join(bank_root, "tiny_mt.json"), "w"))
sys.argv = ["x8", "--model", "tiny", "--bank-root", bank_root, "--families", "tiny_mt", "--layers", "1,2", "--p3-layers", "2", "--first", "logitlens", "--first-layer", "3", "--readers", "jlens,logitlens,jlens_chain,logitlens_chain,p3,x3", "--controls", "shuffled", "--n-ctx", "4", "--chunk", "4", "--T", str(T), "--contexts", ctx_f, "--k1", "2", "--beam", "2", "--max-pieces", "2", "--x3-carriers", "2", "--x3-gen", "3", "--gate-tokens", "3", "--device", "cpu", "--tag", "smk_tiny_p6"]
x8.main(); r8 = json.load(open(os.path.join(runs, "smk_tiny_p6", "results.json"))); f8 = r8["families"]["tiny_mt"]
chk("P6 e2e: ids follow label_of, every item passes the greedy + bridge gate, gate legs logged", float(f8["n_gated"] != 4 or any(r["id"] != R.label_of(it["name"]) for r, it in zip(f8["gate_rows"], bank_items)) or not all(r["bridges"] for r in f8["gate_rows"])), 0.5)
arms = set(f8["arms"]); want = {"jlens", "logitlens", "jlens_chain", "logitlens_chain", "p3", "p3_top1", "x3_identity3"}  # default carriers: identity3 (A12); no tau file given (tau optional)
chk("P6 e2e: every arm and its shuffled control is written and scored", float(not want <= arms or not {a + "__shuffled" for a in want} <= arms), 0.5)
for arm in want:
    f = os.path.join(ROOT, f8["arms"][arm]["file"]); rows = [R.parse_readout_row(json.loads(l)) for l in open(f, encoding="utf-8")]
    kind = "tokens" if arm in ("jlens", "logitlens") else "prose"; nl = len(f8["arms"][arm]["layers"])
    chk(f"P6 e2e: {arm} readouts file parses in the bench contract, one row per (item, layer), kind {kind}", float(any(r is None for r in rows) or len(rows) != 4 * nl or any((r["tokens"] is None) != (kind == "prose") for r in rows) or any(r["token"] is None or r["pos"] != len(tok.encode(bank_items[0]["prompt"])) - 1 for r in rows[:1])), 0.5)
chk("P6 e2e: token arms carry 10 tokens + 10 scores per cell (the bench's top-k)", float(any(len(json.loads(l)["tokens"]) != 10 or len(json.loads(l)["scores"]) != 10 for l in open(os.path.join(ROOT, f8["arms"]["jlens"]["file"])))), 0.5)
chk("P6 e2e: scoring blocks (pass rate, any_hit, by layer, unit_any_layer) and the cross-family table; judge_cmds.sh written", float(any(k not in f8["arms"]["p3"] for k in ("pass_rate", "any_hit_rate", "pass_rate_by_layer", "unit_any_layer", "rows")) or "p3" not in r8["table"] or not os.path.exists(os.path.join(runs, "smk_tiny_p6", "judge_cmds.sh"))), 0.5)
chk("P6 e2e: the J-lens bag can never pass a strictly multi-token unit (raw regex)", float(f8["arms"]["jlens"]["pass_rate"] != 0.0), 0.5)

pa.contexts = pa_orig
import shutil
for t in ("smk_tiny_p2", "smk_tiny_p3tune", "smk_tiny_p3_s0", "smk_tiny_p3_s1", "smk_tiny_p3", "smk_tiny_p6"): shutil.rmtree(os.path.join(runs, t), ignore_errors=True)
shutil.rmtree(tmp, ignore_errors=True)
print("\nSUMMARY:", sum(ok_ for *_, ok_ in res), "/", len(res), "checks passed")
