"""Carrier-wise counts (eight carriers), cut versus uncut scoring, and primary-only versus alias scoring for the
headline tables. Our items: per-carrier exact-cut hits, majority / any recount (asserted against the stored votes),
the benchmark's uncut matcher, and the prefix rule for the continuation prompt. Benchmark: per-carrier pass counts
under the implemented unit-wise contract (any sample, any tested layer), unions of the four prose prefixes and the
four answer frames, the strict single-sample recount, and the primary-only column. CPU only.
Usage: python paper/analysis/carriers.py"""
import json
from common import ROOT, R, load, rows, exact_hits, prefix_hits, bench_hits, majority, write_json, write_csv

# ---------------------------------------------------------------------------------------------------------------
# 1. our items
# ---------------------------------------------------------------------------------------------------------------
OURS = [
    ("14B two-word (original), same layer", "w3_B_x3i3_14b_ans_L32", "identity3:replace", "exact"),
    ("14B two-word (original), block 4", "w3_B_x3i3_14b_ans_L32", "identity3:replace:t4", "exact"),
    ("14B two-word (fresh), same layer", "w5_fresh_14b_B_x3i3_L32", "identity3:replace", "exact"),
    ("14B two-word (fresh), block 4", "w5_fresh_14b_B_x3i3_L32", "identity3:replace:t4", "exact"),
    ("14B sub-word (original), same layer", "w3_A_x3i3_14b_ans_L32", "identity3:replace", "exact"),
    ("14B sub-word (original), block 4", "w3_A_x3i3_14b_ans_L32", "identity3:replace:t4", "exact"),
    ("14B sub-word (fresh), same layer", "w5_fresh_14b_A_x3i3_L32", "identity3:replace", "exact"),
    ("14B sub-word (fresh), block 4", "w5_fresh_14b_A_x3i3_L32", "identity3:replace:t4", "exact"),
    ("8B two-word, same layer", "w5_8b_B_x3i3_L29", "identity3:replace", "exact"),
    ("8B two-word, block 8", "w5_8b_B_x3i3_L29", "identity3:replace:t8", "exact"),
    ("8B sub-word, same layer", "w5_8b_A_x3i3_L29", "identity3:replace", "exact"),
    ("8B sub-word, block 8", "w5_8b_A_x3i3_L29", "identity3:replace:t8", "exact"),
    ("1.7B sub-word, same layer", "w3_A_x3i3_1p7b_ans_L22", "identity3:replace", "exact"),
    ("1.7B sub-word, block 8", "w3_A_x3i3_1p7b_ans_L22", "identity3:replace:t8", "exact"),
    ("14B two-word (original), continuation, same layer", "w5_ph_14b_B_cont", "continuation:replace", "prefix"),
    ("14B two-word (original), continuation, block 4", "w4_B_x3cont_14b_ans_L32", "continuation:replace:t4", "prefix"),
    ("14B two-word (fresh), continuation, block 4", "w5_fresh_14b_B_x3i3_L32", "continuation:replace:t4", "prefix"),
]
ours = []
for label, tag, arm, rule in OURS:
    rs = rows(tag, arm, kind="answer")
    n = len(rs)
    per = [0] * 8
    per_bench = [0] * 8
    maj = anyc = maj_bench = any_bench = maj_prefix = 0
    for r in rs:
        h = exact_hits(r) if rule == "exact" else prefix_hits(r)
        b = bench_hits(r)
        p = prefix_hits(r)
        assert len(h) == 8, (tag, r["id"])
        if rule == "exact":
            assert majority(h) == int(r["vote_exact"]) and int(any(h)) == int(r["any_exact"]), (tag, r["id"])
        else:
            assert majority(h) == int(r["vote_prefix"]), (tag, r["id"])
        for j in range(8):
            per[j] += h[j]
            per_bench[j] += b[j]
        maj += majority(h); anyc += int(any(h)); maj_bench += majority(b); any_bench += int(any(b)); maj_prefix += majority(p)
    e = {"cell": label, "run": tag, "arm": arm, "rule": rule, "n": n, "per_carrier": per, "majority": maj, "any": anyc,
         "uncut_bench_majority": maj_bench, "uncut_bench_any": any_bench, "per_carrier_uncut_bench": per_bench, "prefix_majority": maj_prefix,
         "carrier_min": min(per), "carrier_max": max(per)}
    ours.append(e)
    print(f"{label:52s} n={n:3d} per-carrier={per} maj={maj} any={anyc} | uncut bench maj={maj_bench} any={any_bench} | prefix maj={maj_prefix}")

# ---------------------------------------------------------------------------------------------------------------
# 2. benchmark readouts: per carrier, unions, strict single-sample, primary-only
# ---------------------------------------------------------------------------------------------------------------
BENCH = [
    ("27B basic, continuation block 4", "w5_27b_full_basic", "basic_readout_mt", "x3c_t4"),
    ("27B basic, continuation block 8", "w5_27b_full_basic", "basic_readout_mt", "x3c_t8"),
    ("27B typo, continuation block 4", "w5_27b_full_typo", "typo_mt", "x3c_t4"),
    ("27B typo, continuation block 8", "w5_27b_full_typo", "typo_mt", "x3c_t8"),
    ("27B multihop, continuation block 4", "w5_27b_full_multihop", "multihop_mt", "x3c_t4"),
    ("14B basic (63 gated), continuation block 4", "w5_ctrl_14b_basic_typo", "basic_readout_mt", "x3c_t4"),
    ("14B typo (25 gated), continuation block 8", "w5_ctrl_14b_basic_typo", "typo_mt", "x3c_t8"),
]
PROSE, FRAMES = [0, 1, 2, 3], [4, 5, 6, 7]


def bench_cell(label, tag, fam, arm):
    res = load(tag)
    f = res["families"][fam]
    a = f["arms"][arm]
    header, items = R.load_bank(ROOT / "data" / "wsbench" / f"{fam}.json")
    contract = R.contract_for(header, header.get("family", fam))
    if not f["no_gate"]:
        ok = {g["id"] for g in f["gate_rows"] if g["ok"]}
        items = [it for it in items if it["id"] in ok]
    layers = a["layers"]
    cells = [R.parse_readout_row(json.loads(line)) for line in open(ROOT / a["file"], encoding="utf-8")]
    units_of = {it["id"]: R.scored_units(it, contract) for it in items}

    def score(sample_idx):
        sub = [dict(c, samples=[c["samples"][j] for j in sample_idx]) for c in cells]
        sc = R.score_cells(sub, items, contract, layers, units_of=units_of)
        return sum(1 for r in sc["rows"] if r["pass"]), sc["n_items_decided"]

    full, n_dec = score(list(range(8)))
    assert n_dec == a["n_items_decided"] and abs(full / n_dec - a["pass_rate"]) < 1e-9, (label, full, n_dec, a["pass_rate"])
    per = [score([j])[0] for j in range(8)]
    prose = score(PROSE)[0]
    frames = score(FRAMES)[0]
    # strict: every required unit inside ONE sample at one layer
    by = {}
    for c in cells:
        by.setdefault((c["id"], c["layer"]), c)
    strict = 0
    lost = []
    for it in items:
        u = units_of[it["id"]]
        ok = False
        for l in layers:
            c = by.get((it["id"], l))
            if not c:
                continue
            for s in c["samples"]:
                if R.layer_passes(R.layer_unit_hits([s], u), u):
                    ok = True
                    break
            if ok:
                break
        strict += ok
    sc_full = R.score_cells(cells, items, contract, layers, units_of=units_of)
    strict_ids = set()
    for it in items:
        u = units_of[it["id"]]
        for l in layers:
            c = by.get((it["id"], l))
            if c and any(R.layer_passes(R.layer_unit_hits([s], u), u) for s in c["samples"]):
                strict_ids.add(it["id"])
    lost = sorted({r["id"] for r in sc_full["rows"] if r["pass"]} - strict_ids)
    per_layer = {str(l): round(a["pass_rate_by_layer"][str(l)] * n_dec) for l in layers}
    e = {"cell": label, "run": tag, "family": fam, "arm": arm, "layers": layers, "n": n_dec, "pass": full, "per_layer": per_layer,
         "per_carrier": per, "union_prose": prose, "union_frames": frames, "strict_single_sample": strict, "strict_lost_ids": lost,
         "primary_only": round(a["primary_only"]["pass_rate"] * a["primary_only"]["n_items_decided"]), "alias_only_ids": a["primary_only"]["alias_only_ids"],
         "n_required_units_gt1": sum(1 for it in items if sum(u.required for u in units_of[it["id"]]) > 1)}
    print(f"{label:46s} n={n_dec:3d} pass={full:3d} per-layer={per_layer} per-carrier={per} prose={prose} frames={frames} strict={strict} lost={lost} primary={e['primary_only']} alias-only={len(e['alias_only_ids'])}")
    return e


bench = [bench_cell(*b) for b in BENCH]
carrier_names = ["prose 1 (committee)", "prose 2 (report)", "prose 3 (letter)", "prose 4 (chapter)", "Question/Answer: the", "The answer is", "Q/A:", "The correct term for it is"]
write_json("carriers.json", {"what": "paper/analysis/carriers.py: per-carrier counts, cut vs uncut, prose vs answer-frame unions, strict single-sample recount, primary-only", "continuation_carriers_in_order": carrier_names, "ours": ours, "bench": bench})
write_csv("carriers_ours.csv", [dict(e, per_carrier=" ".join(map(str, e["per_carrier"])), per_carrier_uncut_bench=" ".join(map(str, e["per_carrier_uncut_bench"]))) for e in ours],
          ["cell", "run", "arm", "rule", "n", "per_carrier", "majority", "any", "uncut_bench_majority", "uncut_bench_any", "per_carrier_uncut_bench", "prefix_majority", "carrier_min", "carrier_max"])
write_csv("carriers_bench.csv", [dict(e, per_carrier=" ".join(map(str, e["per_carrier"])), per_layer=json.dumps(e["per_layer"]), strict_lost_ids=" ".join(e["strict_lost_ids"]), alias_only_ids=" ".join(e["alias_only_ids"])) for e in bench],
          ["cell", "run", "family", "arm", "n", "pass", "per_layer", "per_carrier", "union_prose", "union_frames", "strict_single_sample", "strict_lost_ids", "primary_only", "alias_only_ids", "n_required_units_gt1"])
print("-> carriers.json, carriers_ours.csv, carriers_bench.csv")
