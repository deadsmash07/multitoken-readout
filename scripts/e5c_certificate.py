"""E5c / R7 (prereg R7; amendment A25): the split-conformal false-alarm certificate on the identity Patchscope.

R7 as registered: "empirical FPR of the split-conformal threshold ... on the patchscope's score at q = 0.05 / 0.01, on
>= 2,000 background activations from >= 64 records; pass: within q +- 2 sqrt(q (1-q) / N_eff) with N_eff = number
of RECORDS". FULL_REPORT E5c fixes the score: the NUMBER OF AGREEING CARRIERS. This script runs it.

Score of one activation h (read at --layers, patched into the --carriers kind after block --target-layer, greedy
--gen tokens, cut at the carrier stops, norm()): k_agree = the largest number of carriers (of --n-carriers) that
emit the SAME non-empty string; k_agree_multi = the same over strings of >= 2 tokens (the multi-token claim).
Background: generic wikitext records (runs/contexts/qwen3_T128_256.pt: 256 ctx + 256 bg records of T = 128), every
record's residual at the layer at --per-record seeded positions from the valid window (skip 16, not the last token);
records are split in HALF by record (seeded): the calibration half sets tau_q = conformal_threshold(scores, q), the
test half measures FPR_q = P(score > tau_q). N_eff = the number of test RECORDS. The score is discrete (0..8), so the
threshold is conservative: an FPR BELOW q - 2 se is reported as "conservative", not as a miss (R7's pass clause is
two-sided; A25 says which side counts).
Items (offline, CPU, from stored generations): for every results.json in --items-runs, the rows of the arm
`<kind>:replace[:t<target>]` at the layer, per control: coverage = P(k_agree > tau_q), certified-correct =
P(k_agree > tau_q and vote_exact), certified-wrong = P(k_agree > tau_q and not vote_exact), and the same with
k_agree_multi. These are the numbers that turn "emits the string" into "emits a string the certificate allows".
Generation is BATCHED over activations (one carrier, B activations per forward; tests/test_wave5_tiny.py pins
batched == x3.patched_generate row by row). ~0.6 A100-h for 2048 activations x 8 carriers x 2 targets on the 14B.
Usage: python scripts/e5c_certificate.py --model Qwen/Qwen3-14B --dtype bf16-mixed --layers 32 --target-layer 4,same
  --carriers identity3 --n-records 96 --per-record 24 --batch 16 --items-runs runs/w3_B_x3i3_14b_ans_L32,runs/w3_A_x3i3_14b_ans_L32
  --tag w5_e5c_14b_L32   (--limit 4 for a smoke: 4 records)"""
import argparse, glob, importlib.util, json, math, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import forward, logits as true_logits
from sjlens.lens.jlens import valid_mask
from sjlens.lens.score import conformal_threshold
from sjlens.eval.common import load_q
from sjlens.eval.phase_a import contexts

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
_load = lambda n, f: (lambda s: (lambda mod: (s.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s)))(importlib.util.spec_from_file_location(n, os.path.join(HERE, f)))
x3 = _load("x3", "x3_patchscope.py")
QS = (0.05, 0.01)


def batched_patched_generate(q, cid, pos, U, n_gen, target_layer):
    """greedy n_gen tokens for B activations U [B, d] patched (replace) at position `pos` of the carrier cid [1, T]
    after block target_layer; == x3.patched_generate(q, cid, l, pos, U[b], n_gen, "replace", 1.0, None, target_layer)
    for every row b. Returns a [B, n_gen] long tensor."""
    B = U.shape[0]; T = cid.shape[1]; ad = q.w.emb.dtype if q.act_dtype is None else q.act_dtype
    with torch.no_grad():
        x = q.resid(cid, target_layer)[0, pos].float()
        d = torch.zeros(B, T, U.shape[1], dtype=torch.float32, device=cid.device); d[:, pos] = U.float() - x
        m = torch.zeros(B, T, dtype=torch.bool, device=cid.device); m[:, pos] = True
        hL, cache = forward(q.w, cid.expand(B, T), inject=(target_layer, m, d.to(ad)), act_dtype=q.act_dtype)
        out = []
        for _ in range(n_gen):
            t = true_logits(q.w, hL[:, -1]).argmax(-1); out.append(t)
            hL, cache = forward(q.w, t.view(B, 1), None, cache, act_dtype=q.act_dtype)
    return torch.stack(out, 1)


def agree_scores(gens, kind, tok=None):
    """(k_agree, k_agree_multi, the agreed string) of one activation's carrier generations (raw texts): the largest
    multiplicity of an identical non-empty normalised cut string; `multi` restricts to strings of >= 2 tokens (or >= 2
    words when no tokenizer is given)."""
    cnt = {}
    for g in gens:
        s = x3.norm(x3.cut(g, kind))
        if s: cnt[s] = cnt.get(s, 0) + 1
    if not cnt: return 0, 0, ""
    best = max(cnt, key=lambda s: (cnt[s], s)); k = cnt[best]
    is_multi = lambda s: (len(tok.encode(" " + s, add_special_tokens=False)) >= 2) if tok is not None else (len(s.split()) >= 2)
    km = max([c for s, c in cnt.items() if is_multi(s)] or [0])
    return k, km, best


def certify(scores_cal, scores_test, rec_test, q):
    """tau_q on the calibration scores; FPR on the test scores; the R7 line with N_eff = test RECORDS."""
    tau = conformal_threshold(torch.tensor(scores_cal, dtype=torch.float64), q)
    fp = [float(s > tau) for s in scores_test]; fpr = sum(fp) / len(fp) if fp else None
    by_rec = {}
    for r, f in zip(rec_test, fp): by_rec.setdefault(r, []).append(f)
    n_eff = len(by_rec); se = math.sqrt(q * (1 - q) / n_eff) if n_eff else None
    rec_fpr = sum(sum(v) / len(v) for v in by_rec.values()) / n_eff if n_eff else None
    ok = fpr is not None and abs(fpr - q) <= 2 * se; conservative = fpr is not None and fpr < q - 2 * se
    return {"q": q, "tau": tau, "fpr": fpr, "fpr_by_record_mean": rec_fpr, "n_test": len(fp), "n_eff_records": n_eff, "se_records": se,
            "pass_two_sided": bool(ok), "conservative_below": bool(conservative), "pass_one_sided_le": bool(fpr is not None and fpr <= q + 2 * se)}


def item_side(run_dir, layer, kind, target, taus, tok, log):
    """offline: k_agree per stored row of `<kind>:replace[:t<target>]` at `layer` in runs/<run>/results.json, per
    control; coverage / certified-correct / certified-wrong at every tau of `taus` ({(q, score_name): tau})."""
    f = os.path.join(run_dir, "results.json")
    if not os.path.exists(f): log(f"{run_dir}: no results.json; skipped"); return None
    res = json.load(open(f, encoding="utf-8")); blk = res.get("layers", {}).get(str(layer))
    if not blk: log(f"{run_dir}: no layer {layer}; skipped"); return None
    arm = f"{kind}:replace" + ("" if target == "same" else f":t{target}")
    out = {"run": os.path.relpath(run_dir, ROOT), "track": res.get("track"), "arm": arm, "controls": {}}
    for r in blk["rows"]:
        if r["arm"] != arm or not r.get("gens"): continue
        k, km, s = agree_scores(r["gens"], kind, tok)
        out["controls"].setdefault(r["control"], []).append({"id": r["id"], "k": k, "km": km, "agreed": s, "vote_exact": int(r["vote_exact"]), "string": r["string"]})
    for ctrl, rows in out["controls"].items():
        n = len(rows); summ = {"n": n, "vote_exact": sum(r["vote_exact"] for r in rows) / n if n else None, "k_hist": {str(k): sum(r["k"] == k for r in rows) for k in range(0, 9)}}
        for (qq, name), tau in taus.items():
            key = "k" if name == "k_agree" else "km"; cov = [r for r in rows if r[key] > tau]
            summ[f"q{qq}_{name}"] = {"tau": tau, "coverage": len(cov) / n if n else None, "certified_correct": sum(r["vote_exact"] for r in cov) / n if n else None,
                                     "certified_wrong": sum(1 - r["vote_exact"] for r in cov) / n if n else None, "precision_given_certified": (sum(r["vote_exact"] for r in cov) / len(cov)) if cov else None}
        out["controls"][ctrl] = {"summary": summ, "rows": rows}
        log(f"  {out['run']} {arm} [{ctrl}]: n {n}, exact {summ['vote_exact']:.3f}; " + "; ".join(f"q={qq} {name}: tau {v['tau']:.1f} coverage {v['coverage']:.3f} certified-correct {v['certified_correct']:.3f} certified-wrong {v['certified_wrong']:.3f}" for (qq, name), _ in taus.items() for v in [summ[f'q{qq}_{name}']]))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-14B"); p.add_argument("--dtype", default="bf16-mixed"); p.add_argument("--device", default="cuda")
    p.add_argument("--layers", default="32"); p.add_argument("--target-layer", default="4,same", help="comma list: same and/or block indices (single-layer replace, as R3 / A12)")
    p.add_argument("--carriers", default="identity3", help="one identity-format kind (identity3 | identity3_long)"); p.add_argument("--n-carriers", type=int, default=8); p.add_argument("--gen", type=int, default=8)
    p.add_argument("--contexts", default="runs/contexts/qwen3_T128_256.pt"); p.add_argument("--T", type=int, default=128); p.add_argument("--n-records", type=int, default=96, help=">= 64 (R7); half calibration, half test, by record")
    p.add_argument("--per-record", type=int, default=24, help="activations sampled per record from the valid window (>= 2000 in all, R7)"); p.add_argument("--batch", type=int, default=16)
    p.add_argument("--items-runs", default="", help="comma list of runs/<tag> dirs whose results.json rows store gens (w3_* / fresh runs): the item side, offline")
    p.add_argument("--limit", type=int, default=0, help="smoke: use only this many records"); p.add_argument("--seed", type=int, default=0); p.add_argument("--tag", default="e5c_certificate"); a = p.parse_args()
    out_dir = os.path.join(ROOT, "runs", a.tag); os.makedirs(out_dir, exist_ok=True); out_f = os.path.join(out_dir, "results.json")
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True); log(" ".join(sys.argv))
    kind = a.carriers; assert kind in x3.IDENTITY_KINDS, f"--carriers {kind!r}: an identity-format kind"
    x3.assert_no_overlap(kind)
    tok, m, q = load_q(a.model, a.dtype, a.device)
    cpath = os.path.join(ROOT, a.contexts) if os.path.exists(os.path.join(ROOT, a.contexts)) else (a.contexts if os.path.exists(a.contexts) else None)
    if cpath:  # the records file written by scripts/make_contexts.py (compute nodes are offline): bg records first, then ctx
        o = torch.load(cpath); assert o["T"] == a.T, f"contexts file has T={o['T']}, asked for {a.T}"
        recs = [c.to(a.device) for c in list(o["bg"]) + list(o["ctx"])]
    else:
        ctx, bg = contexts(tok, a.n_records, a.n_records, a.T, a.device); recs = bg + ctx
    recs = recs[: a.n_records]
    if a.limit: recs = recs[: a.limit]
    n_rec = len(recs); g = torch.Generator().manual_seed(a.seed); order = torch.randperm(n_rec, generator=g).tolist(); cal_recs = set(order[: n_rec // 2])
    mask = valid_mask(a.T, 16); valid = mask.nonzero().view(-1)
    car = [tok(s, return_tensors="pt").input_ids.to(a.device) for s in x3.CARRIERS[kind][: a.n_carriers]]; cpos = [x3.patch_pos(tok, c, kind) for c in car]
    targets = x3.parse_targets(a.target_layer)
    results = {"model": a.model, "dtype": a.dtype, "kind": kind, "n_carriers": len(car), "gen": a.gen, "T": a.T, "n_records": n_rec, "per_record": a.per_record, "n_cal_records": len(cal_recs), "n_test_records": n_rec - len(cal_recs),
               "score": "k_agree = max multiplicity of one non-empty cut+norm string over the carriers; k_agree_multi = the same over strings of >= 2 tokens", "split": "records permuted with --seed; first half calibration, second half test",
               "prereg": "docs/prereg_R.yaml R7 + A25 (E5c)", "layers": {}}
    for l in [int(x) for x in a.layers.split(",")]:
        t0 = time.time(); acts, rec_of, pos_of = [], [], []
        for ri, r in enumerate(recs):
            with torch.no_grad(): hl = q.resid(r.view(1, -1), l)[0]
            gp = torch.Generator().manual_seed(a.seed * 100003 + ri); sel = valid[torch.randperm(len(valid), generator=gp)[: a.per_record]]
            for p_ in sel.tolist(): acts.append(hl[p_].float().clone()); rec_of.append(ri); pos_of.append(p_)
        H = torch.stack(acts); log(f"L{l}: {H.shape[0]} background activations from {n_rec} records ({len(cal_recs)} calibration / {n_rec - len(cal_recs)} test)")
        Lres = {"n_activations": int(H.shape[0]), "targets": {}}
        for t in targets:
            tl = l if t == "same" else int(t); tkey = "same" if t == "same" else f"t{tl}"
            gens = [[None] * len(car) for _ in range(H.shape[0])]
            for ci, (cid, cp) in enumerate(zip(car, cpos)):
                for b0 in range(0, H.shape[0], a.batch):
                    U = H[b0: b0 + a.batch].to(a.device); ids = batched_patched_generate(q, cid, cp, U, a.gen, tl)
                    for bi in range(ids.shape[0]): gens[b0 + bi][ci] = tok.decode(ids[bi].tolist())
                log(f"  L{l} {tkey} carrier {ci + 1}/{len(car)} done ({time.time() - t0:.0f}s)")
            sc = [agree_scores(gs, kind, tok) for gs in gens]
            k = [s[0] for s in sc]; km = [s[1] for s in sc]
            cal = [i for i in range(len(k)) if rec_of[i] in cal_recs]; tst = [i for i in range(len(k)) if rec_of[i] not in cal_recs]
            block = {"k_hist_all": {str(v): k.count(v) for v in range(0, len(car) + 1)}, "k_multi_hist_all": {str(v): km.count(v) for v in range(0, len(car) + 1)},
                     "agreed_examples": [{"record": rec_of[i], "pos": pos_of[i], "k": k[i], "string": sc[i][2]} for i in sorted(range(len(k)), key=lambda i: -k[i])[:12]],
                     "certificates": {}, "taus": {}}
            for name, s in (("k_agree", k), ("k_agree_multi", km)):
                for qq in QS:
                    c = certify([s[i] for i in cal], [s[i] for i in tst], [rec_of[i] for i in tst], qq); block["certificates"][f"{name}_q{qq}"] = c; block["taus"][f"{name}_q{qq}"] = c["tau"]
                    log(f"  L{l} {tkey} {name}: q {qq} tau {c['tau']:.1f} FPR {c['fpr']:.4f} (N_eff {c['n_eff_records']} records, 2 se {2 * c['se_records']:.4f}) -> {'PASS' if c['pass_two_sided'] else ('conservative' if c['conservative_below'] else 'MISS')}")
            block["samples"] = [{"record": rec_of[i], "pos": pos_of[i], "k": k[i], "km": km[i], "gens": gens[i]} for i in range(0, len(k), max(1, len(k) // 40))]
            if a.items_runs:
                taus = {(qq, name): block["taus"][f"{name}_q{qq}"] for qq in QS for name in ("k_agree", "k_agree_multi")}
                block["items"] = [it_ for run in a.items_runs.split(",") if run for it_ in [item_side(os.path.join(ROOT, run) if not os.path.isabs(run) else run, l, kind, t, taus, tok, log)] if it_ is not None]
            Lres["targets"][tkey] = block; results["layers"][str(l)] = Lres; json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
        Lres["seconds"] = time.time() - t0; json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
    lines = ["| layer | target | score | q | tau | FPR (test) | N_eff records | 2 se | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for l, L in results["layers"].items():
        for tkey, b in L["targets"].items():
            for key, c in b["certificates"].items():
                lines.append(f"| {l} | {tkey} | {key.rsplit('_q', 1)[0]} | {c['q']} | {c['tau']:.1f} | {c['fpr']:.4f} | {c['n_eff_records']} | {2 * c['se_records']:.4f} | {'PASS' if c['pass_two_sided'] else ('conservative (below q - 2 se)' if c['conservative_below'] else 'MISS')} |")
    open(os.path.join(out_dir, "table.md"), "w").write("\n".join(lines) + "\n"); log("\n".join(lines)); log(f"-> {out_f}")


if __name__ == "__main__":
    main()
