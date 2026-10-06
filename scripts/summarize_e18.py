"""Markdown tables from e18 results.json files: python scripts/summarize_e18.py runs/hp_screen_L18 runs/hp_screen_L22 [--e14 runs/e14b_L22]
Per layer: seed / token-lens baseline, second-piece table (variant x numerator), reliability + tau, beam configs
with their offline stopping rules, per-category s2 top-10 for a few variants; with --e14, the dictionary baseline on
the same item ids (mid-rank)."""
import argparse, importlib.util, json, os, sys, statistics
HERE = os.path.dirname(os.path.abspath(__file__))


def f(x, p=2): return "-" if x is None else f"{x:.{p}f}"


def main():
    p = argparse.ArgumentParser(); p.add_argument("dirs", nargs="+"); p.add_argument("--e14", default=""); p.add_argument("--nums", default="hc_r0.05,hJ_r0.05,hJ_r0.02,hJ_r0.1,hJ_jvp")
    p.add_argument("--variants", default=""); a = p.parse_args()
    e14 = json.load(open(os.path.join(a.e14, "results.json"))) if a.e14 else None
    for d in a.dirs:
        r = json.load(open(os.path.join(d, "results.json"))); print(f"\n## {d}: {r['half']} half, {r['n_kept']} gated items of {r['n_items']}; {r['version']}")
        ids = set(r["ids"])
        if e14:
            for l, L in e14["layers"].items():
                idx = [i for i, q in enumerate(L["queries"]) if q["id"] in ids]
                print(f"dictionary baseline (e14 L{l}, {e14['n_cands']} candidates, {len(idx)} of these items): " + "; ".join(f"{k} top-1 {sum(m['ranks'][i] <= 1 for i in idx) / len(idx):.3f} top-10 {sum(m['ranks'][i] <= 10 for i in idx) / len(idx):.3f}" for k, m in L["methods"].items()))
        for l, L in r["layers"].items():
            print(f"\n### layer {l} ({L.get('n_ctx')} ctx, m = {L.get('m_null')}, lens-reachable vocab {L.get('n_lens_reachable')})")
            if "seed_summary" in L:
                print("seed / token-lens: " + "; ".join(f"{k}: top-1 {v['top1']:.3f} top-10 {v['top10']:.3f} median {v['median']:.0f}" for k, v in L["seed_summary"].items()))
                print("FD error vs exact JVP: " + ", ".join(f"{k} {v:.4f}" for k, v in L.get("fd_err", {}).items()))
            ss = L.get("screen_summary", {}); nums = a.nums.split(","); vs = a.variants.split(",") if a.variants else sorted({k.split("|")[0] for k in ss}, key=lambda k: list(r["variants"]).index(k) if k in r["variants"] else 99)
            rel = L.get("reliability", {})
            if ss:
                print("\n| variant | " + " | ".join(f"s2 top-1/top-10/median [{n}]" for n in nums) + " | ov10 / top1-agree (hc; hJ) | tau05 (hc; hJ) |"); print("|---|" + "---|" * (len(nums) + 2))
                for k in vs:
                    cells = [f"{f(ss[f'{k}|{n}']['top1'], 3)} / {f(ss[f'{k}|{n}']['top10'], 3)} / {ss[f'{k}|{n}']['median']:.0f}" if f"{k}|{n}" in ss else "-" for n in nums]
                    rk = rel.get(k, {}); rc = "; ".join(f"{f(rk[i]['ov10'])} / {f(rk[i]['top1'])}" for i in ("hc", "hJ") if i in rk); tc = "; ".join(f"{f(rk[i]['tau05'], 1)}" for i in ("hc", "hJ") if i in rk)
                    print(f"| {k} | " + " | ".join(cells) + f" | {rc} | {tc} |")
            if "beam_summary" not in L and L.get("beam_rows"):  # partial run (stopped before its summary): recompute offline
                spec = importlib.util.spec_from_file_location("e18", os.path.join(HERE, "e18_jbeam_items.py")); e18 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e18)
                from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained(r["args"]["model"]); rel = L.get("reliability", {}); L["beam_summary"] = {}
                for name, kw in r["beam_configs"].items():
                    cfg = e18.Cfg(**{**dict(seeds=r["args"]["seeds"], beams=r["args"]["beams"], max_len=r["args"]["max_len"], expand=r["args"]["expand"], stop=r["args"]["stop"], coherence=bool(r["args"]["coherence"])), **kw})
                    rows = [x for x in L["beam_rows"] if name in x["beam"]]; summ = {"as_run": e18.beam_metrics(rows, name, tok.decode), "n_rows": len(rows)}
                    if cfg.expand == 1 and cfg.stop == "none" and not cfg.coherence and rel:
                        key = {"mc": f"mc{cfg.m_null}", "eb": f"eb{cfg.m_null}_{cfg.prior}", "proxy": f"proxy_{cfg.prior}", "raw": "raw", "floor": f"floor16_c{cfg.floor:g}"}.get(cfg.normalizer); rl = rel.get(key, {}).get(cfg.inject, {})
                        rules = {"zfw": L["zfw"], "tau05": rl.get("tau05"), "tau20": rl.get("tau20"), "none": -float("inf")}
                        summ["stop_rules"] = {f"{rn}{'_coh' if coh else ''}": e18.beam_metrics(rows, name, tok.decode, (th, coh)) for rn, th in rules.items() if th is not None for coh in (True, False)}
                        summ["stop_rules"]["_thresholds"] = {k: v for k, v in rules.items() if v is not None}
                    L["beam_summary"][name] = summ
            if "beam_summary" in L:
                print(f"\nbeam ({len(L.get('beam_rows', []))} items; z_fw {L.get('zfw', 0):.2f}):")
                print("| config | rule | recall@1 | recall@10 | prefix@10 | s2 in beam | junk script | junk lens | mean len | n ext |"); print("|---|---|---|---|---|---|---|---|---|---|")
                for name, b in L["beam_summary"].items():
                    rows = [("as_run", b["as_run"])] + [(k, v) for k, v in b.get("stop_rules", {}).items() if k != "_thresholds"]
                    for rn, v in rows: print(f"| {name} | {rn} | {f(v['recall1'], 3)} | {f(v['recall10'], 3)} | {f(v['prefix10'], 3)} | {f(v['s2_in_beam'], 3)} | {f(v['junk_script'])} | {f(v['junk_lens'])} | {f(v['mean_len'], 1)} | {v['n_ext']} |")
                    if "_thresholds" in b.get("stop_rules", {}): print(f"|  | thresholds {', '.join(f'{k} {v:.2f}' for k, v in b['stop_rules']['_thresholds'].items())} |")
            rows = L.get("screen", [])
            if rows:
                cats = sorted({x["category"] for x in rows}); keys = [k for k in (f"raw|{nums[0]}", f"mc16|{nums[0]}", f"eb16_pooled|{nums[0]}", f"eb16_lens|{nums[0]}", f"proxy_pooled|{nums[0]}") if k in rows[0]["rank_p2"]]
                print("\nper category (n; s1 lens top-10; s2 top-10 by " + ", ".join(keys) + "):")
                for c in cats:
                    sub = [x for x in rows if x["category"] == c]
                    print(f"  {c}: {len(sub)}; {sum(x['rank_p1_lens'] <= 10 for x in sub) / len(sub):.2f}; " + ", ".join(f"{sum(x['rank_p2'][k] <= 10 for x in sub) / len(sub):.2f}" for k in keys))


if __name__ == "__main__":
    main()
