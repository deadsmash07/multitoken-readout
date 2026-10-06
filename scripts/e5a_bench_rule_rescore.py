"""E5a (prereg A18; ANALYSIS_WAVE3 section 6 E5a): rescore the X3 identity3 runs on OUR items (runs/w3_*/results.json)
under the BENCH's matcher, beside the registered rule. CPU only, no model.

Registered rule (R3 / A12): the generation is CUT at the carrier stops (";", "->", newline), normalised (strip,
collapse whitespace, casefold) and must EQUAL the item string; majority of the 8 carriers (vote_exact) and any of 8
(any_exact) are stored per row. Bench rule (wsbench.multitoken.regex, vendored in sjlens/wsbench_regex.py):
unicode_word_matcher(string) - a word-boundary match of the folded string (NFKD, combining marks dropped, casefold)
ANYWHERE in the UNCUT generation (numbers: in answer position, i.e. after "->" / "=" / an answer word or at the
start). Every w3_* row stores its 8 uncut generations (`gens`), so the bench rule is applied offline: majority of 8
and any of 8, per (layer, arm, control), with the paired difference to the registered rule, the item-level bootstrap
CI, and the rows the two rules disagree on (with example generations). Report-only: the registered R3 verdicts are
unchanged; this column is labelled "bench rule" (ANALYSIS_WAVE3 7.5).
Usage: python scripts/e5a_bench_rule_rescore.py [--runs 'runs/w3_*'] [--out runs/e5a_rescore]"""
import argparse, glob, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens import wsbench_regex as R

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


def bench_hit(gen, string):
    """the bench's matcher for the item string inside one uncut generation."""
    return bool(R.unicode_word_matcher(str(string).strip())(gen))


def boot_ci(xs, n=1000, seed=0):
    """(mean, lo, hi): item-level percentile bootstrap, pure python (no torch on the CPU path)."""
    if not xs: return None, None, None
    rng = random.Random(seed); m = len(xs); means = []
    for _ in range(n): means.append(sum(xs[rng.randrange(m)] for _ in range(m)) / m)
    means.sort(); return sum(xs) / m, means[int(0.025 * n)], means[int(0.975 * n) - 1]


def rescore_row(row):
    """(bench majority, bench any, n hits) of one stored row from its uncut generations; None without gens."""
    gens = row.get("gens")
    if not gens: return None
    k = sum(bench_hit(g, row["string"]) for g in gens)
    return {"bench_vote": int(k > len(gens) / 2), "bench_any": int(k > 0), "bench_hits": k, "n_gens": len(gens)}


def rescore_run(path, seed=0, n_examples=8):
    res = json.load(open(path, encoding="utf-8")); out = {"file": os.path.relpath(path, ROOT), "model": res.get("model"), "track": res.get("track"), "kinds": res.get("kinds"), "layers": {}}
    for l, blk in res.get("layers", {}).items():
        cells = {}
        for r in blk.get("rows", []):
            b = rescore_row(r)
            if b is None: cells.setdefault((r["arm"], r["control"]), {"skipped_no_gens": 0, "rows": []})["skipped_no_gens"] = cells.get((r["arm"], r["control"]), {}).get("skipped_no_gens", 0) + 1; continue
            cells.setdefault((r["arm"], r["control"]), {"skipped_no_gens": 0, "rows": []})["rows"].append({**b, "vote_exact": int(r["vote_exact"]), "any_exact": int(r["any_exact"]), "id": r["id"], "string": r["string"], "kind": r.get("kind"), "category": r.get("category"), "position_rule": r.get("position_rule"), "gens": r["gens"]})
        summ = {}
        for (arm, ctrl), c in sorted(cells.items()):
            rs = c["rows"]; e = {"n": len(rs), "skipped_no_gens": c["skipped_no_gens"]}
            if rs:
                for k in ("vote_exact", "any_exact", "bench_vote", "bench_any"):
                    e[k], e[k + "_lo"], e[k + "_hi"] = boot_ci([float(r[k]) for r in rs], seed=seed)
                for a, b in (("bench_vote", "vote_exact"), ("bench_any", "any_exact")):
                    d = [float(r[a]) - float(r[b]) for r in rs]; e[f"{a}_minus_{b}"], e[f"{a}_minus_{b}_lo"], e[f"{a}_minus_{b}_hi"] = boot_ci(d, seed=seed)
                under = [r for r in rs if r["bench_any"] and not r["any_exact"]]; over = [r for r in rs if r["any_exact"] and not r["bench_any"]]
                e["n_bench_any_not_registered_any"] = len(under); e["n_registered_any_not_bench_any"] = len(over)
                e["examples_bench_only"] = [{"id": r["id"], "string": r["string"], "gens": r["gens"][:3]} for r in under[:n_examples]]
                e["examples_registered_only"] = [{"id": r["id"], "string": r["string"], "gens": r["gens"][:3]} for r in over[:n_examples]]
                for key in ("kind", "position_rule"):
                    per = {}
                    for r in rs: per.setdefault(str(r.get(key)), []).append(r)
                    e["by_" + key] = {k: {"n": len(v), "vote_exact": sum(x["vote_exact"] for x in v) / len(v), "bench_vote": sum(x["bench_vote"] for x in v) / len(v), "any_exact": sum(x["any_exact"] for x in v) / len(v), "bench_any": sum(x["bench_any"] for x in v) / len(v)} for k, v in sorted(per.items())}
            summ[f"{arm} [{ctrl}]"] = e
        out["layers"][l] = summ
    return out


def table(all_out):
    lines = ["| run | layer | arm [control] | n | registered majority / any | bench-rule majority / any | delta majority [CI] | delta any [CI] | bench-only / registered-only rows |", "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else f"{v:.3f}"
    for run, o in all_out.items():
        for l, summ in o["layers"].items():
            for cell, e in summ.items():
                if not e.get("n"): continue
                lines.append(f"| {run} | {l} | {cell} | {e['n']} | {f(e['vote_exact'])} / {f(e['any_exact'])} | {f(e['bench_vote'])} / {f(e['bench_any'])} | {f(e['bench_vote_minus_vote_exact'])} [{f(e['bench_vote_minus_vote_exact_lo'])}, {f(e['bench_vote_minus_vote_exact_hi'])}] | {f(e['bench_any_minus_any_exact'])} [{f(e['bench_any_minus_any_exact_lo'])}, {f(e['bench_any_minus_any_exact_hi'])}] | {e['n_bench_any_not_registered_any']} / {e['n_registered_any_not_bench_any']} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--runs", default="runs/w3_*", help="glob of run dirs (relative to sjlens/) whose results.json rows carry `gens`")
    p.add_argument("--out", default="runs/e5a_rescore"); p.add_argument("--seed", type=int, default=0); a = p.parse_args()
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True); log(" ".join(sys.argv))
    out_dir = os.path.join(ROOT, a.out); os.makedirs(out_dir, exist_ok=True); all_out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, a.runs))):
        f = os.path.join(d, "results.json")
        if not os.path.exists(f): continue
        r = json.load(open(f, encoding="utf-8"))
        if not r.get("layers") or not any("gens" in row for blk in r["layers"].values() for row in blk.get("rows", [])[:5]): log(f"skip {os.path.basename(d)}: no stored generations"); continue
        all_out[os.path.basename(d)] = rescore_run(f, a.seed); log(f"{os.path.basename(d)}: " + "; ".join(f"L{l} " + ", ".join(f"{c} {e['vote_exact']:.3f}->{e['bench_vote']:.3f} / {e['any_exact']:.3f}->{e['bench_any']:.3f}" for c, e in s.items() if e.get("n") and c.endswith("[none]")) for l, s in all_out[os.path.basename(d)]["layers"].items()))
    res = {"what": "E5a: bench word-boundary regex inside the UNCUT generation (any / majority of 8) beside the registered cut-then-exact rule (R3 / A12); report only (A18)", "scorer": R.SCORER_VERSION, "seed": a.seed, "runs": all_out}
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1, ensure_ascii=False)
    t = table(all_out); open(os.path.join(out_dir, "table.md"), "w").write(t + "\n"); print(t); log(f"-> {out_dir}/results.json, table.md")


if __name__ == "__main__":
    main()
