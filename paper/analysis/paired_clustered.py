"""Paired early-target gaps with target-clustered intervals; the A21 mechanism gaps with paired intervals; the A26
neutral-context control clustered by first-hop cue; exact binomial tests with Holm adjustment for the registered
families. CPU only; reads stored rows. Writes paired_clustered.json / .csv next to this file.
Usage: python paper/analysis/paired_clustered.py"""
import json, re
from common import ROOT, rows, exact_hits, bench_hits, norm, cluster_boot, item_boot, binom_p_ge, holm, write_json, write_csv, f3

N_BOOT = 10000

# ---------------------------------------------------------------------------------------------------------------
# 1. paired early-target benefit, clustered by normalised target string
# ---------------------------------------------------------------------------------------------------------------
DATASETS = [
    ("14B two-word (original)", "w3_B_x3i3_14b_ans_L32", "identity3:replace", "identity3:replace:t4", "L32", "block 4"),
    ("14B two-word (fresh)", "w5_fresh_14b_B_x3i3_L32", "identity3:replace", "identity3:replace:t4", "L32", "block 4"),
    ("14B sub-word (original)", "w3_A_x3i3_14b_ans_L32", "identity3:replace", "identity3:replace:t4", "L32", "block 4"),
    ("14B sub-word (fresh)", "w5_fresh_14b_A_x3i3_L32", "identity3:replace", "identity3:replace:t4", "L32", "block 4"),
    ("8B two-word", "w5_8b_B_x3i3_L29", "identity3:replace", "identity3:replace:t8", "L29", "block 8"),
    ("8B sub-word", "w5_8b_A_x3i3_L29", "identity3:replace", "identity3:replace:t8", "L29", "block 8"),
    ("1.7B sub-word", "w3_A_x3i3_1p7b_ans_L22", "identity3:replace", "identity3:replace:t4", "L22", "block 4"),
    ("1.7B sub-word", "w3_A_x3i3_1p7b_ans_L22", "identity3:replace", "identity3:replace:t8", "L22", "block 8"),
]


def paired(tag, arm_a, arm_b, metric="vote_exact"):
    a = {r["id"]: r for r in rows(tag, arm_a, kind="answer")}
    b = {r["id"]: r for r in rows(tag, arm_b, kind="answer")}
    ids = sorted(set(a) & set(b))
    assert len(ids) == len(a) == len(b), (tag, len(a), len(b), len(ids))
    per = {}
    for i in ids:
        d = int(b[i][metric]) - int(a[i][metric])
        per.setdefault(norm(a[i]["string"]), []).append(d)
    diffs = [d for c in per.values() for d in c]
    same = sum(int(a[i][metric]) for i in ids)
    early = sum(int(b[i][metric]) for i in ids)
    fav = sum(d > 0 for d in diffs)
    unf = sum(d < 0 for d in diffs)
    m, lo, hi = cluster_boot(list(per.values()), n=N_BOOT)
    mi, ilo, ihi = item_boot(diffs, n=N_BOOT)
    return {"n": len(ids), "n_targets": len(per), "same": same, "early": early, "same_rate": same / len(ids), "early_rate": early / len(ids),
            "diff": m, "cluster_lo": lo, "cluster_hi": hi, "item_lo": ilo, "item_hi": ihi, "favorable": fav, "unfavorable": unf}


results = {"what": "paper/analysis/paired_clustered.py: paired same-layer vs early-target differences (exact majority of 8) with target-string-clustered and item-level percentile bootstrap intervals (10,000 resamples, seed 0); A21 paired gaps; A26 cue-clustered; exact binomial tests with Holm.", "n_boot": N_BOOT, "seed": 0}
res_pairs = []
for label, tag, a, b, src, tgt in DATASETS:
    for metric in ("vote_exact", "any_exact"):
        p = paired(tag, a, b, metric)
        p.update({"dataset": label, "run": tag, "source": src, "target": tgt, "metric": metric})
        res_pairs.append(p)
        if metric == "vote_exact":
            print(f"{label:26s} {tag:28s} {tgt:8s} n={p['n']:3d} targets={p['n_targets']:3d} same={p['same']:3d} early={p['early']:3d} diff={p['diff']:.4f} cluster[{p['cluster_lo']:.4f},{p['cluster_hi']:.4f}] item[{p['item_lo']:.4f},{p['item_hi']:.4f}] fav/unf={p['favorable']}/{p['unfavorable']}")
results["paired_early_target"] = res_pairs

# ---------------------------------------------------------------------------------------------------------------
# 2. A21 mechanism arms, paired on the same 175 two-word items
# ---------------------------------------------------------------------------------------------------------------
def arm_map(tag, arm, metric):
    return {r["id"]: int(r[metric]) for r in rows(tag, arm, kind="answer")}


copy = {"same": arm_map("w3_B_x3i3_14b_ans_L32", "identity3:replace", "vote_exact"),
        "all32": arm_map("w5_ph_14b_B_i3", "identity3:replace:all32", "vote_exact"),
        "all4": arm_map("w5_ph_14b_B_i3", "identity3:replace:all4", "vote_exact"),
        "t4": arm_map("w3_B_x3i3_14b_ans_L32", "identity3:replace:t4", "vote_exact")}
cont = {"same": arm_map("w5_ph_14b_B_cont", "continuation:replace", "vote_prefix"),
        "all32": arm_map("w5_ph_14b_B_cont", "continuation:replace:all32", "vote_prefix"),
        "all4": arm_map("w5_ph_14b_B_cont", "continuation:replace:all4", "vote_prefix"),
        "t4": arm_map("w4_B_x3cont_14b_ans_L32", "continuation:replace:t4", "vote_prefix")}
strings = {r["id"]: norm(r["string"]) for r in rows("w3_B_x3i3_14b_ans_L32", "identity3:replace", kind="answer")}


def mech(arms, label):
    ids = sorted(set.intersection(*[set(v) for v in arms.values()]))
    assert len(ids) == 175, len(ids)
    out = {"carrier": label, "n": len(ids), "counts": {k: sum(v[i] for i in ids) for k, v in arms.items()}}
    out["rates"] = {k: c / len(ids) for k, c in out["counts"].items()}
    pairs = [("all32", "same"), ("t4", "all32"), ("all4", "t4"), ("t4", "same"), ("all4", "same")]
    out["diffs"] = {}
    for b, a in pairs:
        per = {}
        for i in ids:
            per.setdefault(strings[i], []).append(arms[b][i] - arms[a][i])
        diffs = [d for c in per.values() for d in c]
        m, lo, hi = cluster_boot(list(per.values()), n=N_BOOT)
        mi, ilo, ihi = item_boot(diffs, n=N_BOOT)
        out["diffs"][f"{b}-{a}"] = {"diff": m, "cluster_lo": lo, "cluster_hi": hi, "item_lo": ilo, "item_hi": ihi,
                                    "favorable": sum(d > 0 for d in diffs), "unfavorable": sum(d < 0 for d in diffs)}
    # descriptive ratio (all32 - same) / (t4 - same) with an item bootstrap of the ratio
    import random
    rng = random.Random(0)
    ratios = []
    for _ in range(N_BOOT):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        num = sum(arms["all32"][i] - arms["same"][i] for i in s)
        den = sum(arms["t4"][i] - arms["same"][i] for i in s)
        ratios.append(num / den if den else float("nan"))
    ratios = sorted(x for x in ratios if x == x)
    out["ratio_all32_gap"] = {"ratio": (out["counts"]["all32"] - out["counts"]["same"]) / (out["counts"]["t4"] - out["counts"]["same"]),
                              "item_lo": ratios[int(0.025 * len(ratios))], "item_hi": ratios[int(0.975 * len(ratios)) - 1]}
    # the registered reading (A21) on the copy prompt's exact majority
    r = out["rates"]
    out["registered_reading"] = {"contamination": abs(r["all32"] - r["t4"]) <= .05 and r["all32"] - r["same"] >= .15,
                                 "computation": abs(r["all32"] - r["same"]) <= .05 and r["t4"] - r["all32"] >= .15,
                                 "all4_vs_t4_within_05": abs(r["all4"] - r["t4"]) <= .05}
    print(label, out["counts"], {k: f"{v['diff']:.3f} [{v['cluster_lo']:.3f},{v['cluster_hi']:.3f}]" for k, v in out["diffs"].items()}, out["ratio_all32_gap"], out["registered_reading"])
    return out


results["a21_mechanism"] = [mech(copy, "copy prompt (exact majority)"), mech(cont, "continuation prompt (prefix majority)")]

# ---------------------------------------------------------------------------------------------------------------
# 3. A26: neutral-context control, paired by entity and clustered by first-hop cue
# ---------------------------------------------------------------------------------------------------------------
two = {r["id"]: r for r in rows("w5_br_14b_B_fh", "continuation:replace:t4:firsthop", position_rule="firsthop")}
neu = rows("w5_a26_14b_neutral_v2", "continuation:replace:t4:firsthop", position_rule="firsthop")
assert len(two) == 30 and len(neu) == 60, (len(two), len(neu))
a26_items = {it["id"]: it for it in json.load(open(ROOT / "data" / "a26_neutral_items.json", encoding="utf-8"))}
phrase = {it["id"]: it for it in json.load(open(ROOT / "data" / "phrase_items.json", encoding="utf-8"))}


def cue_of(item_id):
    p = re.sub(r" is( the)?$", "", phrase[item_id]["prompt"])
    return re.split(r"(?:whose capital is|whose currency is the) ", p)[-1]


# the cue is the entity the A26 builder placed in the neutral sentence (data/a26_neutral_items.json, `a26_entity`);
# the prompt-derived cue is printed beside it when the two spellings differ (e.g. an article or accent in the prompt)
cues = {}
for r in neu:
    it = a26_items[r["id"]]
    cues[it["a26_orig"]] = it["a26_entity"]
    if it["a26_entity"] != cue_of(it["a26_orig"]):
        print("cue spelling differs:", it["a26_orig"], repr(it["a26_entity"]), "prompt:", repr(cue_of(it["a26_orig"])))
assert set(cues) == set(two)
two_hit = {i: int(any(bench_hits(r))) for i, r in two.items()}
neu_hit = {"n1": {}, "n2": {}}
for r in neu:
    it = a26_items[r["id"]]
    neu_hit[it["a26_template"]][it["a26_orig"]] = int(any(bench_hits(r)))
a26 = {"n_items": 30, "n_cues": len(set(cues.values())), "cues": sorted(set(cues.values())),
       "two_hop": sum(two_hit.values()), "neutral_n1": sum(neu_hit["n1"].values()), "neutral_n2": sum(neu_hit["n2"].values())}
assert a26["two_hop"] == 26 and a26["neutral_n1"] == 20 and a26["neutral_n2"] == 23, a26
for name, getter in (("n1", lambda i: neu_hit["n1"][i]), ("n2", lambda i: neu_hit["n2"][i]), ("mean", lambda i: (neu_hit["n1"][i] + neu_hit["n2"][i]) / 2)):
    per = {}
    for i in two:
        per.setdefault(cues[i], []).append(two_hit[i] - getter(i))
    diffs = [d for c in per.values() for d in c]
    m, lo, hi = cluster_boot(list(per.values()), n=N_BOOT)
    mi, ilo, ihi = item_boot(diffs, n=N_BOOT)
    a26[f"diff_two_hop_minus_{name}"] = {"diff": m, "cue_lo": lo, "cue_hi": hi, "item_lo": ilo, "item_hi": ihi}
per = {}
for i in two:
    per.setdefault(cues[i], []).extend([neu_hit["n1"][i], neu_hit["n2"][i]])
m, lo, hi = cluster_boot(list(per.values()), n=N_BOOT)
a26["neutral_rate"] = {"rate": m, "cue_lo": lo, "cue_hi": hi}
per = {}
for i in two:
    per.setdefault(cues[i], []).append(two_hit[i])
m, lo, hi = cluster_boot(list(per.values()), n=N_BOOT)
a26["two_hop_rate"] = {"rate": m, "cue_lo": lo, "cue_hi": hi}
print("A26", json.dumps(a26, indent=None)[:1200])
results["a26_neutral"] = a26

# ---------------------------------------------------------------------------------------------------------------
# 4. exact binomial tests against the registered .30 line, Holm within each declared family
# ---------------------------------------------------------------------------------------------------------------
def count(tag, arm, metric="vote_exact", kind="answer", position_rule=None):
    rs = rows(tag, arm, kind=kind, position_rule=position_rule)
    return sum(int(r[metric]) for r in rs), len(rs)


def bench_any_count(tag, arm, position_rule):
    rs = rows(tag, arm, position_rule=position_rule)
    return sum(int(any(bench_hits(r))) for r in rs), len(rs)


families = {
    "R3, 14B two-word (L32/L36 x same/block 4)": [("L32 same", count("w3_B_x3i3_14b_ans_L32", "identity3:replace")), ("L32 block 4", count("w3_B_x3i3_14b_ans_L32", "identity3:replace:t4")),
                                                   ("L36 same", count("w3_B_x3i3_14b_ans_L36", "identity3:replace")), ("L36 block 4", count("w3_B_x3i3_14b_ans_L36", "identity3:replace:t4"))],
    "R3, 14B sub-word (L32/L36 x same/block 4)": [("L32 same", count("w3_A_x3i3_14b_ans_L32", "identity3:replace")), ("L32 block 4", count("w3_A_x3i3_14b_ans_L32", "identity3:replace:t4")),
                                                   ("L36 same", count("w3_A_x3i3_14b_ans_L36", "identity3:replace")), ("L36 block 4", count("w3_A_x3i3_14b_ans_L36", "identity3:replace:t4"))],
    "A24, fresh 14B two-word (one cell)": [("L32 block 4", count("w5_fresh_14b_B_x3i3_L32", "identity3:replace:t4"))],
    "A24, fresh 14B sub-word (one cell)": [("L32 block 4", count("w5_fresh_14b_A_x3i3_L32", "identity3:replace:t4"))],
    "A23, 8B two-word (pre-named L29 block 4; L32 block 4; post hoc block 8)": [("L29 block 4", count("w5_8b_B_x3i3_L29", "identity3:replace:t4")), ("L32 block 4", count("w5_8b_B_x3i3_L32", "identity3:replace:t4")),
                                                                              ("L29 block 8 (post hoc)", count("w5_8b_B_x3i3_L29", "identity3:replace:t8"))],
    "A22 (i), first-hop bridge arms (2 carriers x 2 rules)": [("continuation, bench any-of-8", bench_any_count("w5_br_14b_B_fh", "continuation:replace:t4:firsthop", "firsthop")),
                                                            ("continuation, exact majority", count("w5_br_14b_B_fh", "continuation:replace:t4:firsthop", kind="bridge", position_rule="firsthop")),
                                                            ("copy prompt, bench any-of-8", bench_any_count("w5_br_14b_B_fh", "identity3:replace:t4:firsthop", "firsthop")),
                                                            ("copy prompt, exact majority", count("w5_br_14b_B_fh", "identity3:replace:t4:firsthop", kind="bridge", position_rule="firsthop"))],
}
tests = []
for fam, cells in families.items():
    ps = [binom_p_ge(k, n, 0.30) for _, (k, n) in cells]
    adj = holm(ps)
    for (name, (k, n)), p, pa in zip(cells, ps, adj):
        tests.append({"family": fam, "cell": name, "k": k, "n": n, "rate": k / n, "p_one_sided_vs_0.30": p, "p_holm": pa})
        print(f"{fam:70s} {name:32s} {k:3d}/{n:3d} = {k/n:.3f}  p={p:.2e}  holm={pa:.2e}")
results["binomial_tests_vs_030"] = tests

write_json("paired_clustered.json", results)
write_csv("paired_clustered.csv", res_pairs, ["dataset", "run", "source", "target", "metric", "n", "n_targets", "same", "early", "same_rate", "early_rate", "diff", "cluster_lo", "cluster_hi", "item_lo", "item_hi", "favorable", "unfavorable"])
write_csv("binomial_tests.csv", tests, ["family", "cell", "k", "n", "rate", "p_one_sided_vs_0.30", "p_holm"])
print("-> paired_clustered.json, paired_clustered.csv, binomial_tests.csv")
