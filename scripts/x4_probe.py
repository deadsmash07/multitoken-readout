"""X4 (N4 probe ceiling; N6 bridge readers at the first-hop position): is s2 / the string LINEARLY present in h?

Two phases. GPU (or CPU for smokes): gate every item as e12 does and cache h at the last prompt token for all gated
items at --layers (and, for curated bridges, at the first-hop entity's last token, e12.firsthop_pos) into
runs/<tag>/acts.pt. CPU: fit multinomial logistic-regression probes (torch, full-batch L-BFGS, L2 = lam/2 ||W||^2 on
standardised features) and evaluate:
  (a) answers, h -> s2 token id, classes = the observed s2 vocabulary of the gated answer items; trained on the
      TUNING half of the split, evaluated on the HELD-OUT half (top-1 / top-10 of the true class among the K
      classes, mid-rank; held-out items whose class never occurs in training count as misses and are counted).
  (b) answers, h -> full string; same split; also the within-first-piece-family top-1 (rank among the classes that
      share s1; chance = mean 1/|family|), the E14 statistic, so the probe and the dictionary are comparable.
  (c) bridges, h -> bridge string among the bridge strings, --folds-fold cross-validation BY ITEM (bridges are not in
      the split; folds seeded), at the last token and at the first-hop position (N6), by category.
Controls: shuffled labels (the training labels permuted with --seed, same L2, same evaluation); L2 chosen by
--folds-fold CV on the tuning half (a, b) or inside each training fold (c), over --l2-grid, by top-1. Bootstrap
CIs are item-level (1,000 resamples). Pass line (amendment A7, ANALYSIS_DAY1 N4): held-out top-10 >= 0.30 with the
shuffled-label control <= 0.05 for (a) at some layer. --fit-only re-fits from the cached activations."""
import argparse, importlib.util, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.eval.common import load_q

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("e12", os.path.join(HERE, "e12_second_token.py")); e12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e12)
PASS_TOP10, CTRL_MAX = 0.30, 0.05


def fit_lr(X, y, K, lam, iters=200):
    """multinomial logistic regression on standardised X [n, d] (float32, CPU): returns (W [d, K], b [K], mu, sd)."""
    mu = X.mean(0, keepdim=True); sd = X.std(0, keepdim=True).clamp_min(1e-6); Xs = (X - mu) / sd
    W = torch.zeros(X.shape[1], K, requires_grad=True); b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, tolerance_grad=1e-6, tolerance_change=1e-9, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(Xs @ W + b, y) + 0.5 * lam * (W * W).sum(); loss.backward(); return loss
    opt.step(closure)
    return W.detach(), b.detach(), mu, sd


def scores(model, X):
    W, b, mu, sd = model; return ((X - mu) / sd) @ W + b


def midrank(S, y):
    tv = S[torch.arange(len(y)), y][:, None]; return (S > tv).sum(1) + 1 + ((S == tv).sum(1) - 1) / 2


def boot_mean(xs, n=1000, seed=0):
    if len(xs) == 0: return None, None, None
    t = torch.tensor(xs, dtype=torch.float64); g = torch.Generator().manual_seed(seed)
    bs = torch.stack([t[torch.randint(0, len(t), (len(t),), generator=g)].mean() for _ in range(n)])
    return float(t.mean()), float(bs.quantile(0.025)), float(bs.quantile(0.975))


def rank_stats(rank, seed=0):
    r = rank.tolist(); e = {"n": len(r), "median_rank": float(torch.tensor(r).median()) if r else None}
    e["top1"], e["top1_lo"], e["top1_hi"] = boot_mean([float(x <= 1) for x in r], seed=seed)
    e["top10"], e["top10_lo"], e["top10_hi"] = boot_mean([float(x <= 10) for x in r], seed=seed)
    return e


def folds_of(n, k, seed):
    g = torch.Generator().manual_seed(seed); p = torch.randperm(n, generator=g); return [p[i::k].tolist() for i in range(k)]


def choose_l2(X, y, K, grid, folds, seed):
    """lam with the best mean CV top-1 on (X, y); ties go to the larger lam."""
    best = None
    for lam in grid:
        acc = []
        for te in folds_of(len(y), folds, seed):
            tr = [i for i in range(len(y)) if i not in set(te)]
            if not tr or not te: continue
            mdl = fit_lr(X[tr], y[tr], K, lam); acc.append(float((scores(mdl, X[te]).argmax(1) == y[te]).float().mean()))
        m = sum(acc) / len(acc) if acc else 0.0
        if best is None or m >= best[1]: best = (lam, m)
    return best


def evaluate(S, y, seed, fam=None):
    """rank statistics of the truth in the score rows; with fam (class -> family class list) also within-family top-1."""
    rk = midrank(S, y); e = rank_stats(rk, seed)
    if fam is not None:
        within, chance = [], []
        for i, t in enumerate(y.tolist()):
            f = fam[t]
            if len(f) < 2: continue
            sub = S[i, f]; mx = sub.max(); within.append(float(sub[f.index(t)] >= mx) / int((sub == mx).sum())); chance.append(1 / len(f))
        e["n_family"] = len(within); e["within_family_top1"] = sum(within) / len(within) if within else None; e["within_family_chance"] = sum(chance) / len(chance) if chance else None
    return e, rk


def by_cat(rk, cats):
    per = {}
    for r, c in zip(rk.tolist(), cats): per.setdefault(c, []).append(r)
    return {c: {"n": len(v), "top1": sum(x <= 1 for x in v) / len(v), "top10": sum(x <= 10 for x in v) / len(v)} for c, v in sorted(per.items())}


def split_probe(X, y, K, tr, te, cats_te, grid, folds, seed, fam=None, log=print, name=""):
    """train on tr, evaluate on te with a shuffled-label control; returns the block."""
    ytr, yte = y[tr], y[te]; seen = set(ytr.tolist())
    lam, cv = choose_l2(X[tr], ytr, K, grid, folds, seed)
    mdl = fit_lr(X[tr], ytr, K, lam); e, rk = evaluate(scores(mdl, X[te]), yte, seed, fam)
    g = torch.Generator().manual_seed(seed); ysh = ytr[torch.randperm(len(ytr), generator=g)]
    ctl = fit_lr(X[tr], ysh, K, lam); ec, _ = evaluate(scores(ctl, X[te]), yte, seed, fam)
    out = {"n_train": len(tr), "n_test": len(te), "n_classes": K, "n_classes_seen_in_train": len(seen), "n_test_unseen_class": int(sum(t not in seen for t in yte.tolist())),
           "l2": lam, "cv_top1": cv, "heldout": e, "control_shuffled_labels": ec, "by_category": by_cat(rk, cats_te), "ranks": rk.tolist()}
    log(f"  {name}: K {K} (seen {len(seen)}), lam {lam:g} (cv top-1 {cv:.3f}); held-out top-1 {e['top1']:.3f} [{e['top1_lo']:.3f}, {e['top1_hi']:.3f}] top-10 {e['top10']:.3f} "
        f"[{e['top10_lo']:.3f}, {e['top10_hi']:.3f}]" + (f" within-family top-1 {e['within_family_top1']} (chance {e['within_family_chance']}, n {e['n_family']})" if fam is not None else "")
        + f" | shuffled-label control top-1 {ec['top1']:.3f} top-10 {ec['top10']:.3f}; unseen-class test items {out['n_test_unseen_class']}")
    return out


def cv_probe(X, y, K, cats, grid, folds, seed, log=print, name=""):
    """cross-validated by item: every item is scored by a model that never saw it; shuffled-label control inside each fold."""
    S = torch.zeros(len(y), K); Sc = torch.zeros(len(y), K); lams = []
    for te in folds_of(len(y), folds, seed):
        tr = [i for i in range(len(y)) if i not in set(te)]
        if not tr or not te: continue
        lam, _ = choose_l2(X[tr], y[tr], K, grid, min(folds, max(2, len(tr) // 2)), seed); lams.append(lam)
        S[te] = scores(fit_lr(X[tr], y[tr], K, lam), X[te])
        g = torch.Generator().manual_seed(seed + len(lams)); ysh = y[tr][torch.randperm(len(tr), generator=g)]
        Sc[te] = scores(fit_lr(X[tr], ysh, K, lam), X[te])
    e, rk = evaluate(S, y, seed); ec, _ = evaluate(Sc, y, seed)
    n_single = int(sum(1 for t in y.tolist() if (y == t).sum() == 1))
    out = {"n": len(y), "n_classes": K, "n_items_singleton_class": n_single, "l2_per_fold": lams, "cv": e, "control_shuffled_labels": ec, "by_category": by_cat(rk, cats), "ranks": rk.tolist()}
    log(f"  {name}: n {len(y)}, K {K} ({n_single} items are the only example of their class), lams {lams}; CV top-1 {e['top1']:.3f} [{e['top1_lo']:.3f}, {e['top1_hi']:.3f}] top-10 {e['top10']:.3f} "
        f"| shuffled-label control top-1 {ec['top1']:.3f} top-10 {ec['top10']:.3f} | by category " + ", ".join(f"{c} {v['top1']:.2f} (n {v['n']})" for c, v in out["by_category"].items()))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--items", default=os.path.join(os.path.dirname(HERE), "data", "h2a_items.json"))
    p.add_argument("--split-file", default=os.path.join(os.path.dirname(HERE), "data", "h2a_split.json"))
    p.add_argument("--layers", default="10,14,18,22,26"); p.add_argument("--kinds", default="answer,bridge"); p.add_argument("--positions", default="last,firsthop")
    p.add_argument("--min-p1-chars", type=int, default=0); p.add_argument("--per-category", type=int, default=0)
    p.add_argument("--max-scan", type=int, default=0, help="cap the items BEFORE gating (smokes)")
    p.add_argument("--l2-grid", default="0.001,0.01,0.1,1,10"); p.add_argument("--folds", type=int, default=5); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="fp32"); p.add_argument("--device", default="cuda"); p.add_argument("--tag", default="x4_probe")
    p.add_argument("--fit-only", action="store_true", help="skip the model: refit from runs/<tag>/acts.pt"); p.add_argument("--track", default="", help="A (h2a sub-word) | B (phrase) | S (single); default from the items file name"); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(HERE), "runs", a.tag); os.makedirs(out_dir, exist_ok=True); acts_f = os.path.join(out_dir, "acts.pt")
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True); log(" ".join(sys.argv))
    layers = [int(x) for x in a.layers.split(",")]; pos_rules = [x for x in a.positions.split(",") if x]
    if a.fit_only and os.path.exists(acts_f):
        A = torch.load(acts_f, weights_only=True); log(f"loaded {acts_f}: {len(A['items'])} items, layers {A['layers']}")
    else:
        tok, m, q = load_q(a.model, a.dtype, a.device)
        items = e12.subset(e12.load_items(tok, a.items), a)
        if a.max_scan: items = items[: a.max_scan]
        kept = []
        for it in items:
            pid = tok(it["prompt"], return_tensors="pt").input_ids.to(a.device)
            ok, gated, _ = e12.gate(q, tok, it, pid)
            if ok and gated: it["prompt_ids"] = pid; it["firsthop_pos"] = e12.firsthop_pos(tok, it, pid[0].tolist()); kept.append(it)
        log(f"{len(items)} items, {len(kept)} gated ({sum(it['kind'] == 'answer' for it in kept)} answer, {sum(it['kind'] == 'bridge' for it in kept)} bridge)")
        A = {"model": a.model, "layers": layers, "items": [{k: it.get(k) for k in ("id", "kind", "category", "string", "pieces", "pieces_text", "p1_chars", "firsthop_pos", "source")} for it in kept],
             "h": {l: torch.zeros(len(kept), q.d) for l in layers}, "h_firsthop": {l: {} for l in layers}}
        t0 = time.time()
        for n, it in enumerate(kept):
            with torch.no_grad():
                for l in layers:
                    x = q.resid(it["prompt_ids"], l)[0]; A["h"][l][n] = x[-1].float().cpu()
                    if it["kind"] == "bridge" and it.get("firsthop_pos") is not None: A["h_firsthop"][l][n] = x[it["firsthop_pos"]].float().cpu()
            if n % 100 == 0: log(f"cached {n}/{len(kept)} ({time.time() - t0:.0f}s)")
        torch.save(A, acts_f); log(f"activations -> {acts_f}")
    items = A["items"]; sp = json.load(open(a.split_file)); tun, held = set(sp["tuning"]), set(sp["heldout"])
    grid = [float(x) for x in a.l2_grid.split(",")]
    ans = [i for i, it in enumerate(items) if it["kind"] == "answer" and len(it["pieces"]) >= 2]
    tr = [i for i in ans if items[i]["id"] in tun]; te = [i for i in ans if items[i]["id"] in held]
    br = [i for i, it in enumerate(items) if it["kind"] == "bridge"]
    results = {"model": A.get("model"), "track": e12.track_of(a.items, a.track), "items_file": a.items, "split_file": a.split_file, "n_gated": len(items), "n_answer_train": len(tr), "n_answer_test": len(te), "n_bridge": len(br),
               "l2_grid": grid, "folds": a.folds, "seed": a.seed, "prereg": "docs/prereg_R.yaml amendment A7", "layers": {}}
    assert not (set(tr) & set(te)), "tuning / held-out overlap"
    for l in A["layers"]:
        log(f"L{l}: probes"); r = {}; X = A["h"][l]
        if tr and te:
            ans_set = set(ans)  # labels over ALL cached items (0 for non-answer rows, which tr / te never index)
            s2 = sorted({items[i]["pieces"][1] for i in ans}); s2i = {t: k for k, t in enumerate(s2)}
            y = torch.tensor([s2i[items[i]["pieces"][1]] if i in ans_set else 0 for i in range(len(items))])
            cats = [items[i]["category"] for i in te]
            r["s2_token"] = split_probe(X, y, len(s2), tr, te, cats, grid, a.folds, a.seed, log=log, name="answers h -> s2 token")
            st = sorted({items[i]["string"] for i in ans}); sti = {t: k for k, t in enumerate(st)}; ys = torch.tensor([sti[items[i]["string"]] if i in ans_set else 0 for i in range(len(items))])
            p1 = {}; first = {sti[items[i]["string"]]: items[i]["pieces"][0] for i in ans}
            for c, w in first.items(): p1.setdefault(w, []).append(c)
            fam = {c: p1[w] for c, w in first.items()}
            r["string"] = split_probe(X, ys, len(st), tr, te, cats, grid, a.folds, a.seed, fam=fam, log=log, name="answers h -> string")
            r["s2_token"]["ids_test"] = [items[i]["id"] for i in te]
        if br:
            bs = sorted({items[i]["string"] for i in br}); bsi = {t: k for k, t in enumerate(bs)}
            yb = torch.tensor([bsi[items[i]["string"]] for i in br]); cb = [items[i]["category"] for i in br]
            r["bridge"] = {"last": cv_probe(X[br], yb, len(bs), cb, grid, a.folds, a.seed, log=log, name="bridges h(last) -> bridge string")}
            if "firsthop" in pos_rules:
                fh = [i for i in br if i in A["h_firsthop"][l]]
                if fh:
                    Xf = torch.stack([A["h_firsthop"][l][i] for i in fh]); yf = torch.tensor([bsi[items[i]["string"]] for i in fh]); cf = [items[i]["category"] for i in fh]
                    r["bridge"]["firsthop"] = cv_probe(Xf, yf, len(bs), cf, grid, a.folds, a.seed, log=log, name="bridges h(first-hop) -> bridge string")
                    sel = [br.index(i) for i in fh]; r["bridge"]["last_same_items"] = rank_stats(torch.tensor(r["bridge"]["last"]["ranks"])[sel], a.seed)
                    r["bridge"]["n_firsthop"] = len(fh); r["bridge"]["n_no_firsthop"] = len(br) - len(fh)
            r["bridge"]["ids"] = [items[i]["id"] for i in br]
        if "s2_token" in r:
            e, c = r["s2_token"]["heldout"], r["s2_token"]["control_shuffled_labels"]
            r["verdict"] = {"pass": bool(e["top10"] >= PASS_TOP10 and c["top10"] <= CTRL_MAX), "rule": "A7: held-out s2 top-10 >= 0.30 with the shuffled-label control <= 0.05"}
            log(f"L{l} verdict (A7): pass {r['verdict']['pass']}")
        results["layers"][str(l)] = r; json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
