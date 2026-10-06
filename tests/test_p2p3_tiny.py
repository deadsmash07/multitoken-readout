"""Agent-H code (P2 probe, P3 chain reader, P6 WorkspaceBench adapter, the HF-hooks backend) on the tiny random Qwen3
(CPU, float64) and on the bench's own goldens. Content is meaningless on a random model; identities, determinism,
split discipline, formats and the scorer port are what is checked. Run: python tests/test_p2p3_tiny.py"""
import importlib.util, json, os, sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "tests")
from tiny import tiny
from sjlens.model.qwen3_min import Qwen3Min, forward, logits as true_logits
from sjlens.model.backend import MinBackend, HFHooksBackend, make_backend
from sjlens import wsbench_regex as R

HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(os.path.dirname(HERE), "scripts"); ROOT = os.path.dirname(HERE)
load = lambda n, f: (lambda s: (lambda mod: (s.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s)))(importlib.util.spec_from_file_location(n, os.path.join(SC, f)))
x6 = load("x6", "x6_future_probe.py"); x7 = load("x7", "x7_chain_reader.py"); x8 = load("x8", "x8_wsbench_items.py"); x1 = x7.x1; x3 = x8.x3

torch.manual_seed(0)
res = []
def chk(name, err, tol): res.append((name, err, err < tol)); print(f"[{'PASS' if err < tol else 'FAIL'}] {name}: {err:.2e}")


class Tok:  # byte-ish stand-in tokenizer with a few whitespace / punctuation ids
    all_special_ids = [0, 1]; chat_template = None
    _sp = {2: " ", 3: "\n", 4: "-", 5: ".", 6: " ,"}
    def decode(s, ids): return "".join(s._sp.get(int(i), (" " if int(i) % 7 == 0 else "") + chr(97 + int(i) % 26)) for i in ids)
    def encode(s, t, add_special_tokens=False): return [ord(c) % 250 + 6 for c in t]
    def __call__(s, t, return_tensors=None, add_special_tokens=False):
        ids = s.encode(t)
        return {"input_ids": ids} if return_tensors is None else type("O", (), {"input_ids": torch.tensor([ids])})()
    def convert_ids_to_tokens(s, ids): return ["Ġ" + chr(97 + int(i) % 26) if int(i) % 7 == 0 else chr(97 + int(i) % 26) for i in ids]


tok = Tok(); m = tiny(); q = Qwen3Min(m); V = 256; d = 64; L = 2; T = 20
A_ = MinBackend(tok, q); B_ = HFHooksBackend(m, tok)
ids = torch.stack([torch.randint(6, V, (T,)) for _ in range(4)]); h = torch.randn(d, dtype=torch.float64) * 0.5

# --- backend: HF hooks == Qwen3Min (residual read, patch, patched log-probs, patched greedy) ---
chk("backend: resid after block 2 equal", float((A_.resid(ids, L) - B_.resid(ids, L)).abs().max()), 1e-6)
chk("backend: final residual equal", float((A_.final(ids) - B_.final(ids)).abs().max()), 1e-6)
mask = torch.zeros(4, T, dtype=torch.bool); mask[:, 5] = True; inj = (1, mask, h)
chk("backend: patched last log-probs equal (batched carriers, masked position)", float((A_.last_logp(ids, inj) - B_.last_logp(ids, inj)).abs().max()), 1e-5)
delta3 = torch.randn(4, T, d, dtype=torch.float64) * 0.1
chk("backend: per-carrier [B,T,d] delta equal", float((A_.last_logp(ids, (L, None, delta3)) - B_.last_logp(ids, (L, None, delta3))).abs().max()), 1e-5)
m1 = torch.zeros(1, T, dtype=torch.bool); m1[0, T - 1] = True
chk("backend: patched greedy decode equal (patch baked into the cache)", float(A_.greedy(ids[:1], 5, (L, m1, 3 * h)) != B_.greedy(ids[:1], 5, (L, m1, 3 * h))), 0.5)
chk("backend: plain greedy equal", float(A_.greedy(ids[:1], 4) != B_.greedy(ids[:1], 4)), 0.5)
chk("backend: MinBackend.greedy == e12-style greedy (prefill + cache)", float(A_.greedy(ids[:1], 4) != x7.e12.greedy(q, ids[:1], 4)), 0.5)
chk("backend: make_backend kinds", float(make_backend("hf", tok, m).name != "hf" or make_backend("min", tok, m, q).name != "min"), 0.5)
chk("backend: logits through norm + head equal", float((A_.logits(A_.final(ids)) - B_.logits(B_.final(ids))).abs().max()), 1e-5)

# --- P3: boost == x1.delta_logp bit-for-bit; the reader never touches the prompt; determinism; sample rules ---
xlast = A_.resid(ids, L)[:, T - 1]; hn = float(A_.resid(ids, L)[:, 4:].norm(dim=-1).mean())
clean = x1.clean_logp(q, ids, [7]); chk("P3: clean_mean == x1.clean_logp", float((clean - x7.clean_mean(A_, ids, [7])).abs().max()), 0.0 + 1e-300)
d1 = x1.delta_logp(q, ids, L, h, [7], mode="replace", hn=hn, clean=clean, xlast=xlast); d2, pat = x7.boost(A_, ids, L, h, [7], clean, xlast)
chk("P3: boost == x1.delta_logp (finite replace, same layer) bit-for-bit", float((d1 - d2).abs().max()), 0.0 + 1e-300)
chk("P3: patched = boost + clean", float((pat - (d2 + clean)).abs().max()), 1e-9)
ok = x7.allowed_tokens(tok, V)
chk("P3: token filter drops specials, whitespace-only, newline, ', ' but keeps '-' and '.'", float(ok[0] or ok[1] or ok[2] or ok[3] or ok[6] or not ok[4] or not ok[5]), 0.5)
cache = x7.CleanCache(A_, ids, 8); zf = A_.logits(h).double()
prompt_sentinel = {"prompt": "THE PROMPT", "prompt_ids": torch.tensor([[9, 9, 9]])}
calls = []
orig_resid, orig_last = A_.resid, A_.last_logp
def spy_resid(*a, **k): raise AssertionError("chain_read must not read any residual (the prompt never enters the reader)")
def spy_last(ids_, inject=None):
    calls.append(ids_.shape); assert ids_.shape[1] >= T and torch.equal(ids_[:, :T], ids[: ids_.shape[0]]), "the reader ran a forward on something other than carriers + emitted prefix"
    return orig_last(ids_, inject)
A_.resid, A_.last_logp = spy_resid, spy_last
r1 = x7.chain_read(A_, tok, ids, L, h, zf, ok, xlast, cache, k1=3, beam=2, max_pieces=3)
A_.resid, A_.last_logp = orig_resid, orig_last
chk("P3: the reader runs forwards only on carriers + its own emitted prefix (never the prompt)", float(len(calls) == 0), 0.5)
r2 = x7.chain_read(A_, tok, ids, L, h, zf, ok, xlast, x7.CleanCache(A_, ids, 8), k1=3, beam=2, max_pieces=3)
chk("P3: chain_read is deterministic (fresh cache)", float(r1 != r2), 0.5)
chk("P3: HFHooksBackend gives the same chain", float(x7.chain_read(B_, tok, ids, L, h, zf, ok, xlast, x7.CleanCache(B_, ids, 8), 3, 2, 3)["prefixes"][:3] != r1["prefixes"][:3]), 0.5)
chk("P3: k1 first candidates, beam-limited expansions, boosts counted", float(len(r1["cands"]) != 3 or r1["n_boosts"] > 3 + 2 or any(len(p["ids"]) > 3 for p in r1["prefixes"])), 0.5)
chk("P3: every prefix's S is the sum of its step boosts", max(abs(p["S"] - sum(p["d"])) for p in r1["prefixes"]), 1e-9)
chk("P3: candidates obey the token filter", float(any(not ok[c["id"]] for c in r1["cands"]) or any(not ok[t] for p in r1["prefixes"] for t in p["ids"])), 0.5)
s10 = x7.samples_topn(r1["prefixes"], 10)
chk("P3: sample 1 of the N=10 set is the first reader's top-1 token; samples are distinct", float(s10[0] != r1["cands"][0]["t"] or len(set(map(x7.normexact, s10))) != len(s10)), 0.5)
big = x7.sample_top1(r1["prefixes"], 1e9); chk("P3: N=1 under an impossible tau is the top-1 first candidate alone", float(big != r1["cands"][0]["t"]), 0.5)
best2 = max([p for p in r1["prefixes"] if p["d"]], key=lambda p: p["S"]); t0 = x7.sample_top1(r1["prefixes"], min(min(p["d"]) for p in r1["prefixes"] if p["d"]), "sum")
chk("P3: N=1 under the loosest tau is the highest-S multi-piece prefix (rank sum)", float(t0 != best2["t"]), 0.5)
bestL = max([p for p in r1["prefixes"] if p["d"]], key=lambda p: p["d"][-1]); tL = x7.sample_top1(r1["prefixes"], None, "last")
chk("P3: N=1 with no tau under rank last (A11c default) is the prefix with the best LAST step boost", float(tL != bestL["t"]), 0.5)
# no prefix is extended past a non-positive boost
ext = [(p, c) for p in r1["prefixes"] for c in r1["prefixes"] if len(c["ids"]) == len(p["ids"]) + 1 and c["ids"][: len(p["ids"])] == p["ids"]]
chk("P3: a prefix whose last boost < 0 is never extended (tau_min = 0)", float(any(p["d"] and p["d"][-1] < 0 for p, c in ext)), 0.5)
cnd, pre = x7.ordered_bag_chain(tok, zf, zf, ok, 3, 3)
chk("P3: ordered-bag chain: 3 x 3 prefixes, N=1 = ordered top-1 + top-2, continuation skips the first candidate", float(len(pre) != 9 or x7.bag_pair_top1(pre) != tok.decode([cnd[0]["id"], [x for x in torch.topk(zf.masked_fill(~ok, -float("inf")), 10).indices.tolist() if x != cnd[0]["id"]][0]])), 0.5)
chk("P3: normexact folds accents / case / whitespace like the bench", float(x7.normexact("  MÉxico  Lindo ") != "mexico lindo"), 0.5)
# rescoring / summary / any-layer / verdict on synthetic rows
key = x7.normexact(best2["t"])
def row(i, reader, ctrl, prefixes, clean_rank):
    r = {"id": f"i{i}", "category": "k" if i % 2 else "j", "key": key, "reader": reader, "control": ctrl, "prefixes": prefixes, "first_hit_rank": 1, "n_boosts": 3, "rank_s2_clean": clean_rank, "rank_s2_tf": 2.0}
    r.update(x7._score_row(r, 0.0, "sum")); return r  # the synthetic key is the highest-S prefix: the registered sum rule (A11); rank last is checked in test_n0_tiny.py
rows = [row(i, "p3", "none", r1["prefixes"], 1.0 if i < 5 else 50.0) for i in range(10)] + [row(i, "p3", "shuffled", pre, 1.0) for i in range(10)] + [row(i, "jlens_bag10", "none", pre[:1], 1.0) for i in range(10)]
sm = x7.summarise(rows, 0.0, rank="sum")
chk("P3 summary: p3 real exact N=10 = 1, shuffled 0, paired diff +1 with McNemar 10/0", float(sm["p3"]["none"]["exact_top10"] != 1.0 or sm["p3"]["shuffled"]["exact_top10"] != 0.0 or sm["p3"]["shuffled"]["paired"]["exact_top10_diff"] != 1.0 or sm["p3"]["shuffled"]["paired"]["exact_top10_mcnemar"]["b"] != 10), 0.5)
chk("P3 summary: s1-fixes-s2 split counts 5 / 5", float(sm["p3"]["none"]["split_s1_fixes_s2"]["n"] != 5 or sm["p3"]["none"]["split_s1_not_fix"]["n"] != 5), 0.5)
chk("P3 summary: N=1 under tau = 1e9 falls to the first token (exact 0)", float(x7.summarise(rows, 1e9, rank="sum")["p3"]["none"]["exact_top1"] != 0.0), 0.5)
rr = {"layers": {"2": {"rows": rows}, "3": {"rows": [row(i, "p3", "none", pre, 1.0) for i in range(10)]}}, "track": "B"}
al = x7.any_layer(rr, 0.0, rank="sum"); vd = x7.verdict(al, "B")
chk("P3 any-layer: an item passes if it passes at any layer; verdict pass vs the J-lens baseline", float(al["p3"]["none"]["exact_top10"] != 1.0 or not vd["pass"] or vd["kill_this_track"] or vd["best_jlens_baseline"] != 0.0), 0.5)
ft = x7.fit_tau({"layers": {"2": {"rows": rows}}}, log=lambda s: None, rank="sum"); chk("P3 fit_tau returns a grid value that reaches the best exact N=1", float(ft["tau"] not in x7.TAU_GRID or ft["exact_top1"] < 1.0 - 1e-9), 0.5)

# --- P2: streaming ridge == batch; closed form recovers a linear map; lambda selection; entity-grouped folds; bridge CV ---
recs = [torch.randint(0, V, (16,)) for _ in range(24)]
RR = x6.residual_layers(q, torch.stack(recs[:2]), [1, 2])
chk("P2: residual_layers == q.resid and == the final residual", float((RR[2] - q.resid(torch.stack(recs[:2]), 2)).abs().max()) + float((RR[-1] - forward(q.w, torch.stack(recs[:2]))[0]).abs().max()), 1e-6)
acc = x6.accumulate(q, recs[:20], [2], [2], batch=8, log=lambda s: None)
X = torch.cat([x6.residual_layers(q, r.view(1, -1), [2])[2][0, 1:16 - 2] for r in recs[:20]]); Y = torch.cat([x6.residual_layers(q, r.view(1, -1), [2])[-1][0, 2:16 - 1] for r in recs[:20]])
X1 = torch.cat([X.double(), torch.ones(X.shape[0], 1, dtype=torch.float64)], 1)
chk("P2: streaming S_xx / S_xy equal the batch products (x at t, y at t+1 for k = 2)", float((acc[(2, 2)].Sxx - X1.T @ X1).abs().max()) + float((acc[(2, 2)].Sxy - X1.T @ Y.double()).abs().max()), 1e-9)
chk("P2: pair count = records x (T - k - skip)", float(acc[(2, 2)].n != 20 * (16 - 2 - 1)), 0.5)
Atrue = torch.randn(d, d, dtype=torch.float64); Xs = torch.randn(400, d, dtype=torch.float64); Ys = Xs @ Atrue + 0.3
ac = x6.RidgeAccum(d, "cpu"); ac.add(Xs[:200], Ys[:200]); ac.add(Xs[200:], Ys[200:]); Ah, bh = ac.solve(1e-9)
chk("P2: ridge closed form recovers A and the bias", float((Ah - Atrue.float()).abs().max()) + float((bh - 0.3).abs().max()), 1e-4)
lam, info = x6.choose_lambda(q, acc[(2, 2)], recs[20:], 2, 2, [0.01, 1.0], log=lambda s: None)
chk("P2: lambda is a grid value times the S_xx scale, chosen on held-out records", float(info["grid_value"] not in (0.01, 1.0) or abs(lam - info["grid_value"] * info["scale"]) > 1e-9), 0.5)
A2, b2 = acc[(2, 2)].solve(lam); z = x6.probe_logits(q, A2, b2, X[:3]); chk("P2: probe logits go through the model's head ([n, V])", float(tuple(z.shape) != (3, V)), 0.5)
chk("P2: the identity probe is the logit lens", float((x6.probe_logits(q, torch.eye(d), torch.zeros(d), X[:2]) - true_logits(q.w, X[:2]).float()).abs().max()), 1e-6)
groups = x6.entity_groups([(("b", "uzbekistan"), ("f", "tashkent")), (("b", "uzbekistan"), ("f", "samarkand")), (("b", "peru"), ("f", "lima")), (("b", "chile"), ("f", "lima")), (("b", "japan"), ("f", "tokyo"))])
chk("P2: entity groups join items sharing the bridge OR the first-hop token (union-find)", float(groups != [0, 0, 1, 1, 2]), 0.5)
folds = x6.grouped_folds(groups, 3, 0)
chk("P2: grouped folds never split a group", float(any(len({folds[i] for i in range(5) if groups[i] == g}) != 1 for g in set(groups))), 0.5)
chk("P2: grouped folds accept non-contiguous group ids", float(len(set(x6.grouped_folds([5, 5, 9, 9, 2], 2, 0))) != 2), 0.5)
Xb = torch.randn(12, d); yb = torch.randint(0, V, (12,)); gb = [i // 2 for i in range(12)]
rk, rkc, fl, lams = x6.bridge_cv(q, A2, b2, Xb, yb, gb, 3, [0.1, 1.0], 3, 0, log=lambda s: None, name="t")
chk("P2: bridge CV scores every item once by a fold that never held its group; lambdas from the grid", float(rk.shape[0] != 12 or any(len({fl[i] for i in range(12) if gb[i] == g}) != 1 for g in set(gb)) or any(l_ not in (0.1, 1.0) for l_ in lams) or (rk < 1).any()), 0.5)
chk("P2: person / country bucket rule", float(x6.person_country("bridge_person_language") != "person" or x6.person_country("bridge_capital_currency") != "country"), 0.5)
sep = torch.zeros(12, d); sep[torch.arange(12), torch.arange(12)] = 20.0  # a separable toy: fine-tuning must move the map
A3, b3 = x6.finetune_probe(q, A2, b2, sep, yb, 0.0, steps=30, lr=0.05)
chk("P2: fine-tuning through the frozen head lowers the training CE", float(torch.nn.functional.cross_entropy(x6.probe_logits(q, A3, b3, sep), yb) >= torch.nn.functional.cross_entropy(x6.probe_logits(q, A2, b2, sep), yb)), 0.5)

# --- P6: the vendored regex scorer on the bench's own goldens and banks ---
G = json.load(open(os.path.join(HERE, "wsbench_golden", "mt_regex_matching.json"))); U = json.load(open(os.path.join(HERE, "wsbench_golden", "mt_regex_units.json")))
chk("P6 regex: every golden (form, sample) verdict of the bench's matcher", float(sum(R.unicode_word_matcher(p["form"])(p["sample"]) is not p["hit"] for p in G["pairs"])), 0.5)
forms = {"en": ["Battle of Ain Jalut", "Ayn Jalut"], "zh": ["阿因贾鲁特战役", "艾因贾卢特"]}
chk("P6 regex: hit_forms golden (a form never spans two samples)", float(any(R.hit_forms([s], forms) != l for s, l in G["hit_forms"].items()) or R.hit_forms(["New", "Delhi"], {"en": ["New Delhi"]}) != []), 0.5)
lang = {"en": ["Arabic"], "zh": ["阿拉伯语"], "ar": ["عين جالوت", "موقعة عين جالوت"]}
units = [R.ScoredUnit("readout", True, forms), R.ScoredUnit("language", True, lang)]
grid = {20: ["nothing here"], 24: ["Ayn Jalut was in 1260"], 28: ["battle of ain jalut, in Arabic"], 32: ["阿因贾鲁特战役 عين جالوت"]}
chk("P6 regex: the grid verdict golden (conjunctive per layer, any layer)", float(R.item_result({k: R.layer_unit_hits(v, units) for k, v in grid.items()}, units) != G["grid"]), 0.5)
mism = 0; n_items = 0
for fam in R.MT_FAMILIES:
    hdr, items = R.load_bank(os.path.join(ROOT, "data", "wsbench", fam + ".json")); c = R.contract_for(hdr, hdr["family"])
    for it in items:
        n_items += 1; mism += int([u.to_json() for u in R.scored_units(it, c)] != U[fam][it["name"]])
chk(f"P6 regex: scored_units == the bench golden for every item of the six banks ({n_items} items)", float(mism), 0.5)
try:
    R.scored_units({"name": "x", "units": [{"role": "r", "required": True, "forms": {"en": ["one"]}}], "probe_token_lens": {"units": {"r": {"en": [1]}}, "target": 1}}, R.BankContract(True, False, True)); refused = False
except ValueError: refused = True
kept = R.scored_units({"name": "x", "units": [{"role": "r", "required": True, "multi_token": False, "forms": {"en": ["one"]}}], "probe_token_lens": {"units": {"r": {"en": [1]}}, "target": 1}}, R.BankContract(True, False, True))
chk("P6 regex: a required unit with no multi-token form is a bank error; multi_token: false keeps single-token forms", float(not refused or kept[0].forms != {"en": ["one"]}), 0.5)
chk("P6 regex: extract_phrase strips chat scaffolding", float(R.extract_phrase("<|im_start|>assistant\nx</think>") != "x"), 0.5)
chk("P6 regex: label_of / read_offset follow the bench", float(R.label_of("br-proc-arith-2*3") != "br-proc-arith-2_3" or R.read_offset({"readout": {"kind": "final_prompt_token", "offsets": [-1]}}) != 1 or R.read_offset({"readout": {"kind": "final_prompt_token", "offsets": [-3]}}) != 3), 0.5)
chk("P6: display_tokens shows the byte-level space", float(R.display_tokens(tok, [7, 8]) != [" h", "i"]), 0.5)
# the bench's cosine J-lens readout = the transcription of produce/methods.py::JLens.read
Dm = torch.randn(V, d); hh = torch.randn(d); Jm = torch.randn(d, d); w_u = torch.randn(V, d)
bench = ((hh @ Jm.T) @ w_u.T) / (w_u @ Jm).norm(dim=1).clamp_min(1e-9)
chk("P6: jlens_cosine_scores == the bench's JLens.read formula", float((R.jlens_cosine_scores(w_u @ Jm, hh) - bench).abs().max()), 1e-4)
# readouts rows: format, parse, raw-regex scoring end to end on a toy family (the bench's example file also parses)
ex = [json.loads(l) for l in open(os.path.join(HERE, "wsbench_golden", "example_basic_readout_mt.jsonl"))]
chk("P6: the bench's example readouts parse as prose cells", float(any(R.parse_readout_row(r) is None for r in ex) or R.parse_readout_row({"id": "a", "layer": 1, "pos": 2, "tokens": ["x"], "samples": ["y"]}) is not None), 0.5)
hdr, items = R.load_bank(os.path.join(ROOT, "data", "wsbench", "basic_readout_mt.json")); c = R.contract_for(hdr, hdr["family"]); it0 = items[0]
form0 = it0["units"][0]["forms"]["en"][0]
cells = [R.parse_readout_row(x8.prose_row(it0["id"], 20, 40, "x", ["nothing"])), R.parse_readout_row(x8.prose_row(it0["id"], 36, 40, "x", ["The passage names " + form0 + "."])),
         R.parse_readout_row(x8.token_row(tok, items[1]["id"], 20, 40, "x", [7, 8], [1.0, 0.5])), R.parse_readout_row(x8.token_row(tok, items[1]["id"], 36, 40, "x", [7, 8], [1.0, 0.5]))]
sc = R.score_cells(cells, items[:3], c, [20, 36])
by = {r["id"]: r for r in sc["rows"]}
chk("P6 scorer: prose hit at L36 passes (earliest 36), a token bag of single tokens fails, a missing item is undecided", float(by[it0["id"]]["pass"] is not True or by[it0["id"]]["earliest_layer"] != 36 or by[items[1]["id"]]["pass"] is not False or by[items[2]["id"]]["pass"] is not None or sc["n_items_decided"] != 2 or abs(sc["pass_rate"] - 0.5) > 1e-9), 0.5)
chk("P6 gate matcher: target found at a word boundary in a greedy continuation; a numeric target needs answer position", float(not x8.matches_any(" the Aghlabid dynasty ruled", ["Aghlabid dynasty"]) or x8.matches_any(" Aghlabids", ["Aghlabid dynasty"]) or not x8.matches_any(" 63.", ["63"]) or x8.matches_any(" the 63 men", ["63"])), 0.5)
chk("P6 gate: chat_question falls back to a plain Q / A render without a template", float(x8.chat_question(tok, "Who?") != tok.encode("Question: Who?\nAnswer:")), 0.5)
# the P6 gate through the backend on the tiny model: an item whose target is the greedy continuation passes the surface leg
g = A_.greedy(ids[:1], 2); target = tok.decode(g).strip()
it_ok = {"target": target, "target_alts": [], "bridges": []}; gr = x8.gate_item(A_, tok, m, it_ok, ids[:1], 2)
gr2 = x8.gate_item(A_, tok, m, {"target": "zzzzq", "target_alts": [], "bridges": [{"question": "q", "answers": ["zzzzq"]}]}, ids[:1], 2)
chk("P6 gate: greedy-matched target passes, an unmatched target / bridge fails, legs are logged", float(not gr["ok"] or gr2["ok"] or len(gr2["bridges"]) != 1), 0.5) if target and target.isalnum() else chk("P6 gate: (tiny greedy gave a non-word; skipped)", 0.0, 0.5)
# X3 identity2 through the backend equals x3.patched_generate
cid = tok("cat -> cat; ?", return_tensors="pt").input_ids; cp = cid.shape[1] - 1; xx = A_.resid(cid, L)[0, cp]
mk = torch.zeros(1, cid.shape[1], dtype=torch.bool); mk[0, cp] = True
gA = A_.greedy(cid, 4, (L, mk, (h - xx).view(1, 1, -1).expand(1, cid.shape[1], -1))); gX = x3.patched_generate(q, cid, L, cp, h, 4, "replace", 1.0, None, L)
chk("P6: X3 identity2 patch through the backend == x3.patched_generate", float(gA != gX), 0.5)

# --- X5: the added cosine reader (import only; the script's main needs a model) ---
x5 = load("x5", "x5_reader_topk.py"); chk("X5: jlens_cos reader present in summarise loop", float("jlens_cos" not in open(os.path.join(SC, "x5_reader_topk.py")).read()), 0.5)

res = [r for r in res if r is not None]
print("\nSUMMARY:", sum(ok_ for *_, ok_ in res), "/", len(res), "checks passed")
