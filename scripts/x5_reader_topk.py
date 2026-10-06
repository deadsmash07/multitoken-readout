"""X5: the released J-lens's own top-k readout of the answer token(s) at the last prompt position, per layer, beside
the logit lens (N5 single-token set; also the multi-token / phrase sets).

Per gated item (e12 gate; answer items: greedy continuation reproduces `pieces`; for 1-token strings that is the
greedy first token) and layer l, h = residual at the last prompt token:
  jlens       scores D_l h with D_l = W_U J_l (the released lens, no contexts, no controls): rank of the first
              piece over the vocabulary; top-1 / top-10 with item-level bootstrap CIs, by category.
  logitlens   rank of the first piece in lm_head(final_norm(h_l)) - the logit lens through the true head.
  jlens_cos   (agent H) the J-lens with WorkspaceBench's readout of record: (W_U J h) / ||J^T W_U[t]|| per token, i.e.
              D h divided by the row norms of D (produce/methods.py::JLens.read of camilablank/workspace-bench,
              commit 92d763e). The `jlens` block above is the raw dot product D h (the wave-2 convention); both are
              reported so "the original J-lens" number is the bench's.
For strings with >= 2 pieces (the phrase set's bag baseline) each reader also reports: rank of every piece,
bag_all_top10 (EVERY piece in the reader's top-10: the bag reads the whole phrase without order), order_ok (the
pieces' ranks are increasing in string order, given they are all in the top-10), and the rank of the second piece.
No pass / kill: descriptive (amendment A8), it calibrates each reader per category so X1 / X3 / E14 numbers can be
read against what the plain lens already gives."""
import argparse, importlib.util, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import logits as true_logits
from sjlens.lens import jlens
from sjlens.eval.common import load_q
from sjlens.eval.phase_a import token_matrix32
from sjlens.wsbench_regex import jlens_cosine_scores
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("e12", os.path.join(HERE, "e12_second_token.py")); e12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e12)


def rank_of(v, w):
    return float((v > v[w]).sum()) + 1 + float((v == v[w]).sum() - 1) / 2


def boot(xs, n=1000, seed=0):
    if not xs: return None, None, None
    t = torch.tensor(xs, dtype=torch.float64); g = torch.Generator().manual_seed(seed)
    b = torch.stack([t[torch.randint(0, len(t), (len(t),), generator=g)].mean() for _ in range(n)])
    return float(t.mean()), float(b.quantile(0.025)), float(b.quantile(0.975))


def summarise(rows, reader, seed=0):
    rk = [r[reader]["rank_p1"] for r in rows]; e = {"n": len(rows), "median_rank_p1": float(torch.tensor(rk).median())}
    e["top1"], e["top1_lo"], e["top1_hi"] = boot([float(x <= 1) for x in rk], seed=seed); e["top10"], e["top10_lo"], e["top10_hi"] = boot([float(x <= 10) for x in rk], seed=seed)
    multi = [r for r in rows if len(r["pieces"]) >= 2]
    if multi:
        e["n_multi"] = len(multi); e["bag_all_top10"], e["bag_all_top10_lo"], e["bag_all_top10_hi"] = boot([float(r[reader]["bag_all_top10"]) for r in multi], seed=seed)
        inb = [r for r in multi if r[reader]["bag_all_top10"]]; e["order_ok_given_bag"] = sum(r[reader]["order_ok"] for r in inb) / len(inb) if inb else None
        e["p2_top10"] = sum(r[reader]["rank_p2"] <= 10 for r in multi) / len(multi)
    per = {}
    for r in rows: per.setdefault(r["category"], []).append(r)
    e["by_category"] = {}
    for c, v in sorted(per.items()):
        rc = [x[reader]["rank_p1"] for x in v]; cb = {"n": len(v), "top1": sum(x <= 1 for x in rc) / len(v), "top10": sum(x <= 10 for x in rc) / len(v)}
        mv = [x for x in v if len(x["pieces"]) >= 2]
        if mv: cb["bag_all_top10"] = sum(x[reader]["bag_all_top10"] for x in mv) / len(mv)
        e["by_category"][c] = cb
    return e


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--items", default=os.path.join(os.path.dirname(HERE), "data", "single_items.json"))
    p.add_argument("--split", default="heldout"); p.add_argument("--split-file", default=os.path.join(os.path.dirname(HERE), "data", "single_split.json"))
    p.add_argument("--layers", default="10,14,18,22,26"); p.add_argument("--kinds", default="answer"); p.add_argument("--min-p1-chars", type=int, default=0); p.add_argument("--per-category", type=int, default=0)
    p.add_argument("--max-scan", type=int, default=0); p.add_argument("--max-items", type=int, default=0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="fp32"); p.add_argument("--device", default="cuda"); p.add_argument("--tag", default="x5_reader"); p.add_argument("--track", default="", help="A (h2a sub-word) | B (phrase) | S (single); default from the items file name"); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(HERE), "runs", a.tag); os.makedirs(out_dir, exist_ok=True); out_f = os.path.join(out_dir, "results.json")
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True); log(" ".join(sys.argv))
    tok, m, q = load_q(a.model, a.dtype, a.device)
    J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))
    items = e12.split_items(e12.subset(e12.load_items(tok, a.items), a), a.split_file, a.split)
    if a.max_scan: items = items[: a.max_scan]
    kept = []
    for it in items:
        pid = tok(it["prompt"], return_tensors="pt").input_ids.to(a.device)
        ok, gated, _ = e12.gate(q, tok, it, pid)
        if ok and gated: it["prompt_ids"] = pid; kept.append(it)
    if a.max_items: kept = kept[: a.max_items]
    log(f"{len(items)} items ({a.split}); {len(kept)} gated")
    results = {"model": a.model, "items_file": a.items, "track": e12.track_of(a.items, a.track), "split": a.split, "n_items": len(items), "n_kept": len(kept), "prereg": "docs/prereg_R.yaml amendment A8 (descriptive)", "layers": {}}
    for l in [int(x) for x in a.layers.split(",")]:
        t0 = time.time(); D = token_matrix32(q.w, J[l]); rows = []
        for it in kept:
            with torch.no_grad():
                h = q.resid(it["prompt_ids"], l)[0, -1]; sj = (D @ h.float()).double(); sl = true_logits(q.w, h).float().double(); sjc = jlens_cosine_scores(D, h.float()).double()
            row = {"id": it["id"], "category": it["category"], "string": it["string"], "pieces": it["pieces"], "shared_s1_group": it.get("shared_s1_group")}
            for name, v in (("jlens", sj), ("logitlens", sl), ("jlens_cos", sjc)):
                ranks = [rank_of(v, t) for t in it["pieces"]]
                row[name] = {"rank_p1": ranks[0], "ranks": ranks, "rank_p2": ranks[1] if len(ranks) > 1 else None, "bag_all_top10": int(all(r <= 10 for r in ranks)),
                             "order_ok": int(all(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1))), "top5": [tok.decode([t]) for t in torch.topk(v, 5).indices.tolist()]}
            rows.append(row)
        summ = {"rows": rows, "seconds": time.time() - t0, "jlens": summarise(rows, "jlens", a.seed), "logitlens": summarise(rows, "logitlens", a.seed), "jlens_cos": summarise(rows, "jlens_cos", a.seed)}
        results["layers"][str(l)] = summ; json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
        for rd in ("jlens", "logitlens", "jlens_cos"):
            e = summ[rd]; bag = f", bag all-in-top-10 {e['bag_all_top10']:.3f} (order ok given bag {e['order_ok_given_bag']}, n {e['n_multi']})" if "n_multi" in e else ""
            log(f"L{l} {rd:9s}: n {e['n']}, first-piece top-1 {e['top1']:.3f} [{e['top1_lo']:.3f}, {e['top1_hi']:.3f}] top-10 {e['top10']:.3f} [{e['top10_lo']:.3f}, {e['top10_hi']:.3f}] median rank {e['median_rank_p1']:.0f}{bag}")


if __name__ == "__main__":
    main()
