"""Recompute the H1a attenuation correction from saved e8_splithalf / g5 rows (no model needed).

corrected = cos(d, v_N) / sqrt(r_N) with r_N = 2r / (1 + r) (Spearman-Brown; r = cos(v_A, v_B) is the reliability of
an N/2-context estimate, r_N that of the N-context mean). The old key divided by sqrt(r) (over-correcting). Prints
per-stratum and overall medians of raw / r / corrected / old; --write adds the recomputed keys to each results.json."""
import argparse, glob, json, os, statistics
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser(); p.add_argument("runs", nargs="*", default=sorted(glob.glob(os.path.join(root, "runs", "g5_L*", "results.json"))) + [os.path.join(root, "runs", "e8_splithalf_L11", "results.json")])
p.add_argument("--write", action="store_true"); a = p.parse_args()
med = lambda rs, k: statistics.median(r[k] for r in rs)
for f in a.runs:
    if not os.path.exists(f): print(f"missing {f}"); continue
    R = json.load(open(f)); rows = R["tokens"]
    for row in rows:
        r = max(row["split_half"], 1e-6); row["corrected"] = row["cos_d_v"] / (2 * r / (1 + r)) ** 0.5; row["corrected_halfrel_old"] = row["cos_d_v"] / r ** 0.5
    keys = ("cos_d_v", "split_half", "corrected", "corrected_halfrel_old")
    R["summary"] = {k: med(rows, k) for k in keys} | ({"cos_d_vA": med(rows, "cos_d_vA")} if "cos_d_vA" in rows[0] else {})
    print(f"{os.path.relpath(f, root)}: {R.get('model')} L{R['layer']} N={R['n_ctx']} n={len(rows)}  (medians: raw / r / corrected Spearman-Brown / old)")
    strata = sorted({row.get("stratum", "") for row in rows})
    if strata != [""]:
        R["by_stratum"] = {}
        for st in strata:
            rs = [row for row in rows if row.get("stratum", "") == st]; R["by_stratum"][st] = {k: med(rs, k) for k in keys} | {"n": len(rs)}
            b = R["by_stratum"][st]; print(f"  {st:9s} n={len(rs):3d}: {b['cos_d_v']:.3f} / {b['split_half']:.3f} / {b['corrected']:.3f} / {b['corrected_halfrel_old']:.3f}")
    else:
        for row in rows: print(f"  {row['token']!r:14s}: {row['cos_d_v']:.3f} / {row['split_half']:.3f} / {row['corrected']:.3f} / {row['corrected_halfrel_old']:.3f}")
    s = R["summary"]; print(f"  {'all':9s} n={len(rows):3d}: {s['cos_d_v']:.3f} / {s['split_half']:.3f} / {s['corrected']:.3f} / {s['corrected_halfrel_old']:.3f}")
    if a.write: json.dump(R, open(f, "w"), indent=1); print(f"  wrote {f}")
