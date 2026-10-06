"""E14 / dictionary reader: can a backward phrase dictionary read a multi-token string from one activation?

Candidates: every distinct `string` of data/h2a_items.json (~1.2k multi-token strings; other items' answers of the same
category are the hard distractors). For each candidate s with pieces (p1..pn), at layer l, from N generic contexts:
  pure        v_z(s) = sum_i J(s_<i)^T W_U[s_i]                  (lens/backward.py sj_vectors, mode 'z')
  hybrid_nm   d(p1) * ||t1|| / ||d(p1)|| + (v_z(s) - t1),  t1 = v_z(p1)   (step 1 = the lens row, norm-matched)
  first_lens  d(p1)                                              (token-lens control)
  mean_lens   mean_i d(p_i)                                      (token-lens bag control)
Queries: items the model gets right (e12's gate), h = residual at the last prompt token. Score z_bg(v, h) =
<v, h - mu> / sd_bg(<v, g>) with background activations; rank of the item's own string among all candidates.
Ranks are mid-ranks (ties split). Reports top-1 / top-10 / median rank per method, per kind (answer: the string is the model's upcoming output;
bridge: a hidden intermediate) and per category; chance top-1 = 1 / n_candidates. Saves vectors for reuse.
Wave 2 (N6, C1): --positions last,firsthop also scores every bridge query at the last token of its FIRST-HOP entity
(e12.firsthop_pos: "... whose capital is Gaborone is" -> the last token of "Gaborone"; curated templates only, items
without a template position are skipped and counted) - results under layers[l]["by_position"]["firsthop"], the
last-token block stays where it was. The shuffled-h control is the fixed within-category derangement that never pairs
two items with the same string (queries with no valid partner are dropped from the control and counted), and a
cross-category control (control_crosscat) is added. --items / --split-file make it runnable on the single-token and
phrase sets (data/single_items.json, data/phrase_items.json)."""
import argparse, importlib.util, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min
from sjlens.lens import jlens
from sjlens.lens.jlens import valid_mask
from sjlens.lens.backward import sj_vectors
from sjlens.eval.common import load_hf, load_q, background_activations
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("e12", os.path.join(HERE, "e12_second_token.py")); e12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e12)
NUMERIC_CATS = ("arith_digits", "year", "number_word", "order-ops", "order-ops-target")  # strings are numbers / number words (C2: within-family chance is not the honest number there)


def is_numeric_cat(cat, strings=()):
    """a category whose strings are numbers: by name, or if at least half of its strings contain a digit."""
    return cat in NUMERIC_CATS or (len(strings) > 0 and sum(any(ch.isdigit() for ch in s) for s in strings) * 2 >= len(strings))


def midrank(Z, truth):
    """mid-rank of the true candidate per query (ties split): [nq] float."""
    tv = Z[torch.arange(len(truth)), truth][:, None]
    return (Z > tv).sum(1) + 1 + ((Z == tv).sum(1) - 1) / 2


def family_metrics(Z, truth, cand_p1, q_cat, numeric=()):
    """P2 / C2. Z [nq, nc] scores, truth [nq] candidate index, cand_p1 [nc] first piece per candidate, q_cat [nq]
    category per query, numeric: categories excluded from the 'ex_numeric' variant. A family = candidates sharing the
    first piece. Returns unique-first-piece top-1 (first-piece identification suffices there), within-family top-1
    (rank of the truth among its family only; chance = mean 1/|family|), both also excluding numeric categories, and
    by-category tables of all-candidate top-1/top-10, within-family top-1 and chance."""
    fam = {}
    for i, w in enumerate([int(x) for x in cand_p1]): fam.setdefault(w, []).append(i)
    rank = midrank(Z, truth); per = {}
    uniq, within, chance, uniq_x, within_x, chance_x = [], [], [], [], [], []
    for qi, ti in enumerate(truth.tolist()):
        f = fam[int(cand_p1[ti])]; c = q_cat[qi]; r = per.setdefault(c, {"n": 0, "top1": 0, "top10": 0, "n_unique": 0, "unique_top1": 0, "n_family": 0, "within_top1": 0, "chance": 0.0})
        r["n"] += 1; r["top1"] += int(rank[qi] <= 1); r["top10"] += int(rank[qi] <= 10); nx = c not in numeric
        if len(f) == 1:
            ok = int(Z[qi].argmax()) == ti; uniq.append(ok); r["n_unique"] += 1; r["unique_top1"] += ok
            if nx: uniq_x.append(ok)
        else:
            sub = Z[qi, f]; mx = sub.max(); ok = float(sub[f.index(ti)] >= mx) / int((sub == mx).sum())  # ties at the max share the credit
            within.append(ok); chance.append(1 / len(f)); r["n_family"] += 1; r["within_top1"] += ok; r["chance"] += 1 / len(f)
            if nx: within_x.append(ok); chance_x.append(1 / len(f))
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    for c, r in per.items():
        r["top1"] /= r["n"]; r["top10"] /= r["n"]; r["unique_top1"] = r["unique_top1"] / r["n_unique"] if r["n_unique"] else None
        r["within_top1"] = r["within_top1"] / r["n_family"] if r["n_family"] else None; r["chance"] = r["chance"] / r["n_family"] if r["n_family"] else None; r["numeric"] = c in numeric
    return {"n_unique": len(uniq), "unique_top1": mean(uniq), "n_family": len(within), "within_family_top1": mean(within), "within_family_chance": mean(chance),
            "n_unique_ex_numeric": len(uniq_x), "unique_top1_ex_numeric": mean(uniq_x), "n_family_ex_numeric": len(within_x), "within_family_top1_ex_numeric": mean(within_x),
            "within_family_chance_ex_numeric": mean(chance_x), "by_category": {c: per[c] for c in sorted(per)}}


shuffled_perm = e12.shuffled_perm  # within-category derangement (shuffled-h control), shared with x1 / x3; -1 = no valid partner (C1)
cross_perm = e12.cross_perm  # cross-category derangement (crosscat control)


def random_queries(Hq, mu, seed=0):
    """norm-matched random directions: Gaussian rows scaled to ||h - mu|| of each query, then re-centred at mu."""
    g = torch.Generator().manual_seed(seed); R = torch.randn(Hq.shape, generator=g, dtype=Hq.dtype)
    return mu + R / R.norm(dim=1, keepdim=True) * (Hq - mu).norm(dim=1, keepdim=True)


def rank_stats(rank):
    rs = rank.tolist(); return {"n": len(rs), "top1": sum(x <= 1 for x in rs) / len(rs), "top10": sum(x <= 10 for x in rs) / len(rs), "median_rank": float(torch.tensor(rs).float().median())}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--items", default=os.path.join(os.path.dirname(HERE), "data", "h2a_items.json"))
    p.add_argument("--layers", default="14"); p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=32); p.add_argument("--n-bg", type=int, default=16)
    p.add_argument("--max-cands", type=int, default=0); p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--max-scan", type=int, default=0, help="cap the items considered BEFORE gating (smokes: gating greedy-decodes every item)")
    p.add_argument("--device", default="cpu"); p.add_argument("--contexts", default="contexts/qwen3_T128.pt"); p.add_argument("--tag", default="e14_dictionary")
    p.add_argument("--dtype", default="fp32", help="fp32 | bf16 | bf16-mixed (bf16 weights, fp32 activations; = fp32 for Qwen3, half the memory)")
    p.add_argument("--split", default="all", help="all | tuning | heldout: restrict the answer QUERIES to a half of data/h2a_split.json (candidates stay all strings)")
    p.add_argument("--split-file", default=os.path.join(os.path.dirname(HERE), "data", "h2a_split.json")); p.add_argument("--seed", type=int, default=0, help="controls (shuffled pairing, random directions)")
    p.add_argument("--positions", default="last", help="last | last,firsthop (bridge queries also at the first-hop entity's last token, N6)"); p.add_argument("--track", default="", help="A (h2a sub-word) | B (phrase) | S (single); default from the items file name"); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(HERE), "runs", a.tag); os.makedirs(out_dir, exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    tok, m, q = load_q(a.model, a.dtype, a.device)
    J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))
    ctx, bg = contexts(tok, a.n_ctx, a.n_bg, a.T, a.device, path=a.contexts if os.path.exists(a.contexts) else None)
    mask = valid_mask(a.T, 16).to(a.device)
    items = e12.load_items(tok, a.items)
    if a.max_scan: items = items[: a.max_scan]
    cands = {}
    for it in items: cands.setdefault(it["string"], it["pieces"])
    names = sorted(cands)[: a.max_cands] if a.max_cands else sorted(cands); idx = {s: i for i, s in enumerate(names)}
    queries = []; keep = None if a.split == "all" else set(json.load(open(a.split_file))[a.split])
    for it in items:
        if it["string"] not in idx or (keep is not None and it["kind"] == "answer" and it["id"] not in keep): continue
        ids = tok(it["prompt"], return_tensors="pt").input_ids.to(a.device)
        ok, gated, _ = e12.gate(q, tok, it, ids)
        it["firsthop_pos"] = e12.firsthop_pos(tok, it, ids[0].tolist())
        if ok and gated: queries.append((it, ids))
    if a.max_queries: queries = queries[: a.max_queries]
    log(f"{len(names)} candidate strings; {len(queries)} gated queries ({sum(it['kind'] == 'answer' for it, _ in queries)} answer)")
    q_cat = [it["category"] for it, _ in queries]; cand_p1 = [cands[s][0] for s in names]; q_str = [it["string"] for it, _ in queries]
    pos_rules = [x for x in a.positions.split(",") if x]
    fh = [i for i, (it, _) in enumerate(queries) if it["kind"] == "bridge" and it.get("firsthop_pos") is not None]  # bridge queries with a first-hop position
    if "firsthop" in pos_rules: log(f"first-hop position known for {len(fh)} of {sum(it['kind'] == 'bridge' for it, _ in queries)} bridge queries")
    by_cat = {}
    for it in items: by_cat.setdefault(it["category"], []).append(it["string"])
    numeric = sorted(c for c in by_cat if is_numeric_cat(c, by_cat[c])); log(f"numeric categories (excluded in the ex_numeric family variant): {numeric}")
    results = {"model": a.model, "dtype": a.dtype, "split": a.split, "seed": a.seed, "track": e12.track_of(a.items, a.track), "n_ctx": len(ctx), "n_cands": len(names), "n_queries": len(queries), "numeric_categories": numeric,
               "n_bridge_leak_excluded": sum(bool(it.get("bridge_leak")) and it.get("correct_pre_leak", False) for it in items), "positions": pos_rules, "n_firsthop_queries": len(fh), "layers": {}}
    for l in [int(x) for x in a.layers.split(",")]:
        t0 = time.time(); D = token_matrix32(q.w, J[l]).double().cpu()
        H = background_activations(q, bg, l, mask).double().cpu(); mu = H.mean(0); Hc = H - mu
        V = {k: [] for k in ("pure", "hybrid_nm", "first_lens", "mean_lens")}
        for n, s in enumerate(names):
            pc = cands[s]; full = sj_vectors(q, pc, ctx, [l], 16, mode="z")[l]; t1 = sj_vectors(q, pc[:1], ctx, [l], 16, mode="z")[l]
            d1 = D[pc[0]]
            V["pure"].append(full); V["hybrid_nm"].append(d1 * (t1.norm() / d1.norm()) + (full - t1))
            V["first_lens"].append(d1); V["mean_lens"].append(D[pc].mean(0))
            if n % 100 == 0: log(f"L{l} vectors {n}/{len(names)} ({time.time() - t0:.0f}s)")
        V = {k: torch.stack(v) for k, v in V.items()}; torch.save({"names": names, **{k: v.float() for k, v in V.items()}}, os.path.join(out_dir, f"vectors_L{l}.pt"))
        with torch.no_grad(): Hq = torch.stack([q.resid(ids, l)[0, -1] for _, ids in queries]).double().cpu()
        r = {"seconds_vectors": time.time() - t0, "methods": {}}
        truth = torch.tensor([idx[it["string"]] for it, _ in queries]); Zs = {}
        perm = shuffled_perm(q_cat, a.seed, q_str); xperm = cross_perm(q_cat, a.seed, q_str); Hr = random_queries(Hq, mu, a.seed)  # P2 controls: another item's h (same category, never the same string; C1) / another category / a norm-matched random direction
        r["n_shuffled_dropped"] = int((perm < 0).sum()); r["n_crosscat_dropped"] = int((xperm < 0).sum())
        if "firsthop" in pos_rules and fh:
            with torch.no_grad(): Hf = torch.stack([q.resid(queries[i][1], l)[0, queries[i][0]["firsthop_pos"]] for i in fh]).double().cpu()
        def ctrl_block(Z, prm, tr, cats):
            """rank stats + family metrics of the ORIGINAL truths scored with the partner's rows (partners < 0 dropped)."""
            v = (prm >= 0).nonzero().flatten()
            if v.numel() == 0: return {"n": 0}
            o = rank_stats(midrank(Z[prm[v]], tr[v])); o["family"] = {kk: vv for kk, vv in family_metrics(Z[prm[v]], tr[v], cand_p1, [cats[i] for i in v.tolist()], numeric).items() if kk != "by_category"}; return o
        for k, M in V.items():
            sd = (Hc @ M.T).std(0).clamp_min(1e-12); Z = ((Hq - mu) @ M.T) / sd  # [nq, nc]
            rank = midrank(Z, truth); Zs[k] = Z.float()  # mid-rank: candidates with an identical score (e.g. first_lens of strings sharing p1) split the tie
            per = {}
            for (it, _), rk in zip(queries, rank.tolist()):
                for key in (it["kind"], f"{it['kind']}:{it['category']}"): per.setdefault(key, []).append(rk)
            f = lambda rs: rank_stats(torch.tensor(rs))
            r["methods"][k] = {"all": f(rank.tolist()), **{kk: f(v) for kk, v in sorted(per.items())}, "ranks": rank.tolist()}
            fm = family_metrics(Z, truth, cand_p1, q_cat, numeric); r["methods"][k]["family"] = fm
            for kind in ("answer", "bridge"):
                sel = [i for i, c in enumerate(q_cat) if queries[i][0]["kind"] == kind]
                if sel: r["methods"][k]["family"][kind] = {kk: v for kk, v in family_metrics(Z[sel], truth[sel], cand_p1, [q_cat[i] for i in sel], numeric).items() if kk != "by_category"}
            r["methods"][k]["control_shuffled"] = ctrl_block(Z, perm, truth, q_cat); r["methods"][k]["control_crosscat"] = ctrl_block(Z, xperm, truth, q_cat)
            Zr = ((Hr - mu) @ M.T) / sd; r["methods"][k]["control_random"] = rank_stats(midrank(Zr, truth)); r["methods"][k]["control_random"]["family"] = {kk: v for kk, v in family_metrics(Zr, truth, cand_p1, q_cat, numeric).items() if kk != "by_category"}
            if "firsthop" in pos_rules and fh:  # N6: the same bridge queries read at the first-hop entity's last token
                Zf = ((Hf - mu) @ M.T) / sd; tf = truth[fh]; cf = [q_cat[i] for i in fh]; sf = [q_str[i] for i in fh]
                fb = {"n": len(fh), "bridge": rank_stats(midrank(Zf, tf)), "ranks": midrank(Zf, tf).tolist(), "family": {kk: v for kk, v in family_metrics(Zf, tf, cand_p1, cf, numeric).items() if kk != "by_category"}}
                fb["bridge_last_same_items"] = rank_stats(rank[fh]); fb["by_category"] = {}
                for c in sorted(set(cf)):
                    sel = [j for j, cc in enumerate(cf) if cc == c]; fb["by_category"][c] = {"firsthop": rank_stats(midrank(Zf, tf)[sel]), "last": rank_stats(rank[fh][sel])}
                pf = shuffled_perm(cf, a.seed, sf); xf = cross_perm(cf, a.seed, sf)
                fb["control_shuffled"] = ctrl_block(Zf, pf, tf, cf); fb["control_crosscat"] = ctrl_block(Zf, xf, tf, cf)
                Zfr = ((random_queries(Hf, mu, a.seed) - mu) @ M.T) / sd; fb["control_random"] = rank_stats(midrank(Zfr, tf))
                r.setdefault("by_position", {}).setdefault("firsthop", {})[k] = fb; Zs[k + ":firsthop"] = Zf.float()
                log(f"L{l} {k:10s} bridges at FIRST-HOP position: top-1 {fb['bridge']['top1']:.3f} top-10 {fb['bridge']['top10']:.3f} (same items at last token: top-1 {fb['bridge_last_same_items']['top1']:.3f}); "
                    f"shuffled {fb['control_shuffled'].get('top1', float('nan')):.3f}, crosscat {fb['control_crosscat'].get('top1', float('nan')):.3f}, random {fb['control_random']['top1']:.3f}")
            ex = [(it["string"], names[int(Z[i].argmax())]) for i, (it, _) in enumerate(queries[:8])]
            log(f"L{l} {k:10s}: top-1 {r['methods'][k]['all']['top1']:.3f} top-10 {r['methods'][k]['all']['top10']:.3f} median rank {r['methods'][k]['all']['median_rank']:.0f} "
                f"| answer top-1 {r['methods'][k].get('answer', {}).get('top1', float('nan')):.3f} bridge top-1 {r['methods'][k].get('bridge', {}).get('top1', float('nan')):.3f} | e.g. {ex[:4]}")
            g_ = lambda x: "-" if x is None else f"{x:.3f}"
            log(f"L{l} {k:10s} family: unique-p1 top-1 {g_(fm['unique_top1'])} (n {fm['n_unique']}); within-family top-1 {g_(fm['within_family_top1'])} vs chance {g_(fm['within_family_chance'])} (n {fm['n_family']}); "
                f"ex numeric {g_(fm['within_family_top1_ex_numeric'])} vs {g_(fm['within_family_chance_ex_numeric'])} (n {fm['n_family_ex_numeric']}) | controls top-1/top-10: shuffled-h "
                f"{r['methods'][k]['control_shuffled'].get('top1', float('nan')):.3f}/{r['methods'][k]['control_shuffled'].get('top10', float('nan')):.3f} (within-family {g_(r['methods'][k]['control_shuffled'].get('family', {}).get('within_family_top1'))}), "
                f"crosscat {r['methods'][k]['control_crosscat'].get('top1', float('nan')):.3f}/{r['methods'][k]['control_crosscat'].get('top10', float('nan')):.3f}, "
                f"random {r['methods'][k]['control_random']['top1']:.3f}/{r['methods'][k]['control_random']['top10']:.3f}")
        torch.save({"names": names, "truth": truth, "perm": perm, "xperm": xperm, "q_cat": q_cat, "cand_p1": cand_p1, "firsthop_query_idx": fh, **Zs}, os.path.join(out_dir, f"scores_L{l}.pt"))
        r["queries"] = [{"id": it["id"], "kind": it["kind"], "category": it["category"], "string": it["string"]} for it, _ in queries]
        r["chance_top1"] = 1 / len(names); r["seconds"] = time.time() - t0; results["layers"][str(l)] = r
        json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
