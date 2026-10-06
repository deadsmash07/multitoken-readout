"""A20's primary-string-only column, tabulated per CELL (read layer x target) and per denominator beside the bench
rule, from the rescored runs under runs/w5_primary_only/<run>/results.json (x8 --rescore --rescore-out; CPU).
For every x3-type arm: per layer L, bench pass = items whose passing_layers contain L; primary pass = the same over
primary_only.passing_layers_by_id; denominators gated / gated-immediate / all (--no-gate runs) from the gate rows.
Writes runs/w5_primary_only/table.md and table.json. Usage: python scripts/w5_primary_table.py [--root runs/w5_primary_only]"""
import argparse, glob, importlib.util, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens import wsbench_regex as R

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
x8 = (lambda s: (lambda mod: (s.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s)))(importlib.util.spec_from_file_location("x8", os.path.join(HERE, "x8_wsbench_items.py")))


def cell_table(res):
    out = []
    for fam, fres in res["families"].items():
        bank = fres["bank"] if os.path.exists(fres["bank"]) else os.path.join(ROOT, "data", "wsbench", fam + ".json")
        header, items = R.load_bank(bank); scored, gated, imm = x8.gate_sets(fres, items); ids_all = {it["id"] for it in scored}
        dens = {"all": ids_all if fres.get("no_gate") else None, "gated": gated & ids_all, "immediate": imm & ids_all}
        for sub, e in fres["arms"].items():
            if not sub.startswith("x3") or "primary_only" not in e: continue
            bench = {r["id"]: set(r["passing_layers"]) for r in e["rows"]}; prim = {i: set(v) for i, v in e["primary_only"]["passing_layers_by_id"].items()}
            for L in e["layers"]:
                row = {"family": fam, "arm": sub, "layer": L}
                for dn, ids in dens.items():
                    if ids is None: row[dn] = None; continue
                    n = len(ids); b = sum(1 for i in ids if L in bench.get(i, ())); p = sum(1 for i in ids if L in prim.get(i, ()))
                    row[dn] = {"n": n, "bench": b, "primary": p, "bench_rate": b / n if n else None, "primary_rate": p / n if n else None}
                out.append(row)
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", default="runs/w5_primary_only"); a = p.parse_args()
    root = os.path.join(ROOT, a.root); allrows = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
        run = os.path.basename(os.path.dirname(f)); allrows[run] = cell_table(json.load(open(f, encoding="utf-8")))
    f_ = lambda c: "-" if c is None or c["n"] == 0 else f"{c['bench']}/{c['n']}={c['bench_rate']:.3f} | {c['primary']}/{c['n']}={c['primary_rate']:.3f}"
    lines = ["# A20 primary-string-only column (first creditable form per unit, no aliases) beside the bench rule, per cell", "",
             "| run | family | arm | layer | gated: bench / primary | immediate: bench / primary | all: bench / primary |", "|---|---|---|---|---|---|---|"]
    for run, rows in allrows.items():
        for r in rows:
            if "__" in r["arm"] and not any((r[d] or {}).get("bench") for d in ("gated", "immediate", "all") if r[d]): continue  # silent control rows
            lines.append(f"| {run} | {r['family']} | {r['arm']} | L{r['layer']} | {f_(r['gated'])} | {f_(r['immediate'])} | {f_(r['all'])} |")
    open(os.path.join(root, "table.md"), "w").write("\n".join(lines) + "\n"); json.dump(allrows, open(os.path.join(root, "table.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
