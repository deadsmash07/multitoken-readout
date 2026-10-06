"""E12 / H2a: second-token recovery, prompt-blind.

Items: --items is either data/h2a_items.json (built by scripts/make_h2a_items.py: `kind` answer / bridge, `string`,
`pieces`, `target`; readout at the last prompt token) or the old glob over anthropics/jacobian-lens
data/evaluations/lens-eval-*.json (every ' '+intermediate with >= 2 pieces, kind bridge, target = the item's target).
Gate (unless --no-gate): the model's greedy continuation of the prompt must reproduce the item's `pieces` exactly
(answer items) or, for bridge items, one of the target's tokenisations (with / without a leading space) as a prefix
or the target string anywhere in the first 6 greedy tokens (the model likes " Uzbekistan som"). Items without a
target (association, poetry, typo) are kept ungated and flagged.
For each kept item and layer: h = residual at the readout position; p1, p2 = first two pieces. The step readout
num(w) = W_U J(p1) h over generic contexts (exact JVP; the prompt never enters) ranks all tokens; we record the rank
of p2, raw and z-scored with m Hutchinson null directions (centred background activations), and the rank of p1 in
the exact step-1 lens readout D h (the precondition: the lens must see the first piece before the second can be
asked for). Per layer: p2 top-1 / top-10 overall, the count of items with p1 in the lens top-k (--pre-k), and top-1 /
top-10 conditional on it, overall and per kind. "any_layer": for each item the layer where p1's lens rank is best
(ties: first listed layer) and p2's rank there - this is a per-item selection over layers, reported separately from
the per-layer numbers and not a pre-registered statistic. Pass: second piece top-1 >= 50%; kill < 30%.
Saves runs/<tag>/results.json after every item; --resume skips (layer, id) pairs already in it."""
import argparse, glob, json, os, random, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min, forward, pseudo_logits, logits as true_logits
from sjlens.lens import jlens
from sjlens.lens.jlens import valid_mask
from sjlens.lens.forward import jvp_exact
from sjlens.lens.score import hutchinson_sigma, nnomp
from sjlens.eval.common import load_hf, load_q, background_activations
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download


def load_items(tok, path):
    if path.endswith(".json") and os.path.isfile(path):
        items = json.load(open(path)); assert isinstance(items, list) and "pieces" in items[0], f"{path} is not an h2a_items.json"
        for it in items: it.setdefault("name", "")
        return items
    items = []  # old behaviour: Anthropic evaluation files, bridge items only
    for f in sorted(glob.glob(path)):
        slug = os.path.basename(f)[len("lens-eval-"):-5]
        for it in json.load(open(f))["items"]:
            if "target" not in it or isinstance(it["prompt"], list): continue
            for s in it["intermediates"]:
                pieces = tok.encode(" " + s, add_special_tokens=False)
                if len(pieces) >= 2 and tok.decode([pieces[0]]).strip():
                    items.append({"id": f"lens_eval:{slug}:{it.get('name', '')}:{s}", "source": "lens_eval", "category": slug, "kind": "bridge", "name": it.get("name", ""),
                                  "prompt": it["prompt"], "readout": "last_prompt_token", "string": " " + s, "pieces": pieces, "pieces_text": [tok.decode([t]) for t in pieces],
                                  "target": it["target"], "p1_chars": len(tok.decode([pieces[0]]).strip())})
    return items


def subset(items, a):
    if a.kinds: items = [it for it in items if it["kind"] in a.kinds.split(",")]
    if getattr(a, "track_filter", ""): items = [it for it in items if it.get("track") == a.track_filter]  # fresh_items.json carries both tracks (A24)
    if a.min_p1_chars: items = [it for it in items if it.get("p1_chars", 9) >= a.min_p1_chars]
    if a.per_category:  # stratified: at most n per (source, category), seeded
        by = {}
        for it in items: by.setdefault((it["source"], it["category"]), []).append(it)
        rng = random.Random(a.seed); items = [it for k in sorted(by) for it in (rng.sample(by[k], a.per_category) if len(by[k]) > a.per_category else by[k])]
    return items


def greedy(q, ids, k):
    out = []
    with torch.no_grad():
        hL, cache = q.prefill(ids)
        for _ in range(k):
            t = int(true_logits(q.w, hL[0, -1]).argmax()); out.append(t)
            hL, cache = forward(q.w, torch.tensor([[t]], device=ids.device), None, cache, act_dtype=q.act_dtype)
    return out


def _contains(g, v):
    """token-level containment of tuple v in list g (no substring matches: target "1" does not match "16")."""
    return any(tuple(g[i:i + len(v)]) == v for i in range(len(g) - len(v) + 1))


def bridge_leak(it, txt, tg):
    """P1 / C1: the bridge is verbalised in the greedy output (" Uzbekistani som"), or the target starts with the
    bridge's first piece (" Uzbek" -> "Uzbek"): in both cases the last-token state trivially carries the bridge."""
    s = it["string"].strip().lower(); p1 = it["pieces_text"][0].strip().lower()
    return (bool(s) and s in txt.lower()) or any(bool(p1) and t.strip().lower().startswith(p1) for t in tg)


def gate(q, tok, it, ids):
    """(correct, gated, greedy text): answer items must reproduce `pieces`; bridge items one tokenisation of a target
    as a prefix or contained token-wise in the first 6 greedy tokens, and no leak (it["bridge_leak"] is set)."""
    if it["kind"] == "answer": vs = {tuple(it["pieces"])}
    else:
        tg = it["target"] if isinstance(it["target"], list) else [it["target"]]; tg = [t for t in tg if t]
        if not tg: return True, False, ""
        vs = {tuple(tok.encode(v, add_special_tokens=False)) for t in tg for v in (t, " " + t.lstrip(), t.lstrip())}
    g = greedy(q, ids, max(len(v) for v in vs) if it["kind"] == "answer" else max(6, *(len(v) for v in vs))); txt = tok.decode(g)
    ok = any(tuple(g[: len(v)]) == v for v in vs)
    if it["kind"] == "bridge":
        ok = ok or any(_contains(g, v) for v in vs)  # " Uzbekistan som": demonym-prefixed answers count, token-wise
        it["bridge_leak"] = bridge_leak(it, txt, tg); it["correct_pre_leak"] = ok; ok = ok and not it["bridge_leak"]
    return ok, True, txt


def _derange(idx, ok, g, tries=200):
    """a random bijection q of the index list idx with ok(i, q[i]) for every i, found by seeded random draws plus
    pairwise swap repairs; positions that cannot be satisfied are -1 (the caller drops them and reports the count)."""
    n = len(idx)
    if n < 2: return [-1] * n
    best, best_bad = None, n + 1
    for _ in range(tries):
        q = [idx[k] for k in torch.randperm(n, generator=g).tolist()]
        for _r in range(4):  # repair: swap an offending position with one that keeps both partners valid
            bad = [k for k in range(n) if not ok(idx[k], q[k])]
            if not bad: break
            for k in bad:
                for m in torch.randperm(n, generator=g).tolist():
                    if m != k and ok(idx[k], q[m]) and ok(idx[m], q[k]): q[k], q[m] = q[m], q[k]; break
        bad = [k for k in range(n) if not ok(idx[k], q[k])]
        if len(bad) < best_bad: best, best_bad = q, len(bad)
        if not bad: break
    return [j if ok(i, j) else -1 for i, j in zip(idx, best)]


def _cat_seed(seed, cat):
    import zlib
    return int(seed) * 1000003 + zlib.crc32(str(cat).encode()) % 1000003


def shuffled_perm(cats, seed=0, strings=None):
    """within-category derangement of the item indices: the shuffled-h control (another item's h, SAME category).
    C1 (ANALYSIS_DAY1): partner i -> j requires j != i and, when `strings` is given, strings[j] != strings[i] (two
    '120' arithmetic items are the same answer); singleton categories and positions with no valid partner are -1
    and must be dropped from the control comparison (the callers report the count). Seeded per category
    (seed, category name) so one category's pairing does not depend on which other categories are present."""
    perm = torch.full((len(cats),), -1, dtype=torch.long)
    for cat in sorted(set(cats)):
        idx = [i for i, c in enumerate(cats) if c == cat]
        ok = lambda i, j: i != j and (strings is None or strings[i] != strings[j])
        q = _derange(idx, ok, torch.Generator().manual_seed(_cat_seed(seed, cat)))
        for i, j in zip(idx, q): perm[i] = j
    return perm


def cross_perm(cats, seed=0, strings=None):
    """cross-category derangement (control `crosscat`): partner from a DIFFERENT category (and a different string),
    seeded; -1 where impossible (a single category; or, since the pairing is a bijection, the excess of a category
    that holds more than half of the items). Separates the category-level content of h (this control) from
    the item-level content (the same-category control) - ANALYSIS_DAY1 section 2."""
    idx = list(range(len(cats)))
    ok = lambda i, j: i != j and cats[i] != cats[j] and (strings is None or strings[i] != strings[j])
    return torch.tensor(_derange(idx, ok, torch.Generator().manual_seed(_cat_seed(seed, "__cross__"))) if idx else [], dtype=torch.long)


def track_of(items_path, override=""):
    """the item-set track a run belongs to (reported in every wave-2 results.json): A = sub-word multi-token
    (data/h2a_items.json: word fragments, digits), B = word-phrase (data/phrase_items.json: every piece a word token),
    S = single-token calibration (data/single_items.json); --track overrides."""
    if override: return override
    b = os.path.basename(str(items_path))
    return {"h2a_items.json": "A", "phrase_items.json": "B", "single_items.json": "S"}.get(b, "other:" + b)


FIRSTHOP_TAILS = (" is the", " is", " was")


def firsthop_pos(tok, it, prompt_ids=None):
    """position (index into the prompt tokens) of the LAST token of the first-hop entity of a curated bridge prompt,
    or None. The curated bridge templates (make_h2a_items.py) all end "... <entity> is" / "... <entity> is the" /
    "... <entity> was" where <entity> is the mentioned capital / currency / element symbol / description of the work
    ("Fact: The continent of the country whose capital is Gaborone is" -> the last token of "Gaborone"; "Fact: The
    nationality of the composer of the ballet Swan Lake was" -> the last token of "Lake"). The tail is stripped, the
    remaining prefix is tokenised and required to be a token-prefix of the full prompt (else None). Non-curated
    bridges (lens_eval / probe_swap: free-form prompts) have no template and return None; callers report the count."""
    if it.get("kind") != "bridge" or it.get("source") not in ("curated", "fresh"): return None  # fresh (A24): the same templates
    pr = it["prompt"]
    for tail in FIRSTHOP_TAILS:
        if pr.endswith(tail):
            pre = pr[: -len(tail)]; break
    else: return None
    full = prompt_ids if prompt_ids is not None else tok(pr, return_tensors="pt").input_ids[0].tolist()
    if hasattr(full, "tolist"): full = full.view(-1).tolist()
    pids = tok(pre, return_tensors="pt").input_ids[0].tolist()
    if len(pids) == 0 or len(pids) >= len(full) or pids != full[: len(pids)]: return None
    return len(pids) - 1


def split_items(items, split_path, which):
    """which = all | tuning | heldout. The split (data/h2a_split.json) covers ANSWER items only; items of another
    kind (bridges) are never in it and are kept as they are, so --kinds bridge is unaffected by --split."""
    if which == "all": return items
    keep = set(json.load(open(split_path))[which]); return [it for it in items if it["kind"] != "answer" or it["id"] in keep]


def _stats(rows, pre_k):
    t = lambda k: torch.tensor([r[k] for r in rows]).float()
    rz, rr, r1 = t("rank_p2_z"), t("rank_p2_raw"), t("rank_p1_lens"); pre = r1 <= pre_k
    f = lambda x: float(x.float().mean()) if x.numel() else None
    return {"n": len(rows), "p2_top1_z": f(rz == 1), "p2_top10_z": f(rz <= 10), "p2_median_rank_z": float(rz.median()), "p2_top1_raw": f(rr == 1), "p2_top10_raw": f(rr <= 10),
            "p1_median_rank_lens": float(r1.median()), "n_pre": int(pre.sum()), "p1_in_lens_topk": f(pre),
            "p2_top1_z_given_pre": f(rz[pre] == 1), "p2_top10_z_given_pre": f(rz[pre] <= 10), "p2_top1_raw_given_pre": f(rr[pre] == 1), "p2_top10_raw_given_pre": f(rr[pre] <= 10)}


def summarise(rows, pre_k):
    if not rows: return {"n": 0}
    s = _stats(rows, pre_k)
    for kind in ("answer", "bridge"):
        sub = [r for r in rows if r["kind"] == kind]
        if sub: s[kind] = {k: v for k, v in _stats(sub, pre_k).items() if k in ("n", "p2_top1_z", "p2_top10_z", "n_pre", "p2_top1_z_given_pre", "p2_top10_z_given_pre")}
    return s


def any_layer(results, pre_k):
    """per item: the run layer with the best p1 lens rank (first on ties), and p2's rank there (a selection)."""
    best = {}
    for l, summ in results["layers"].items():
        for r in summ["rows"]:
            if r["id"] not in best or r["rank_p1_lens"] < best[r["id"]]["rank_p1_lens"]: best[r["id"]] = dict(r, layer=int(l))
    s = summarise(list(best.values()), pre_k); s["note"] = "layer selected per item by best p1 lens rank; not a per-layer statistic"
    s["layer_hist"] = {str(l): sum(r["layer"] == l for r in best.values()) for l in sorted({r["layer"] for r in best.values()})}
    return s


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--items", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "h2a_items.json"), help="h2a_items.json or a lens-eval-*.json glob")
    p.add_argument("--layers", default="11"); p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=32); p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--m-null", type=int, default=16); p.add_argument("--max-items", type=int, default=0); p.add_argument("--per-category", type=int, default=0, help="stratified cap per (source, category)")
    p.add_argument("--kinds", default="", help="answer,bridge"); p.add_argument("--min-p1-chars", type=int, default=0); p.add_argument("--pre-k", type=int, default=10); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inject", default="hc", help="vector pushed through J(p1): hc = h - mu (Prop. 4 z_bg), h (raw; uncentred), hJ = NNOMP projection of hc"); p.add_argument("--no-gate", action="store_true"); p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cpu"); p.add_argument("--contexts", default="contexts/qwen3_T128.pt"); p.add_argument("--tag", default="e12_second_token")
    p.add_argument("--dtype", default="fp32", help="fp32 | bf16 | bf16-mixed (bf16 weights, fp32 activations; = fp32 for Qwen3, half the memory)")
    p.add_argument("--split", default="all", help="all | tuning | heldout (answer items of data/h2a_split.json)")
    p.add_argument("--split-file", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "h2a_split.json")); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.tag); os.makedirs(out_dir, exist_ok=True); out_f = os.path.join(out_dir, "results.json")
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    tok, m, q = load_q(a.model, a.dtype, a.device)
    J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))
    ctx, bg = contexts(tok, a.n_ctx, 8, a.T, a.device, path=a.contexts if os.path.exists(a.contexts) else None)
    ids = torch.stack(ctx); mask = valid_mask(a.T, 16).to(a.device)
    items = split_items(subset(load_items(tok, a.items), a), a.split_file, a.split); log(f"{len(items)} items from {a.items} after subset filters (split {a.split})")
    kept, gate_rows = [], []
    for it in items:
        pid = tok(it["prompt"], return_tensors="pt").input_ids.to(a.device)
        it["correct"], it["gated"], it["model_greedy"] = gate(q, tok, it, pid); it["prompt_ids"] = pid
        gate_rows.append({k: it.get(k) for k in ("id", "kind", "category", "string", "target", "correct", "gated", "model_greedy", "bridge_leak", "correct_pre_leak")})
        if it["correct"] or a.no_gate: kept.append(it)
    nc = sum(it["correct"] for it in items); log(f"{len(kept)} kept after gating ({nc} greedy-correct of {len(items)}; {sum(not it['gated'] for it in items)} ungated); "
                                                 f"per kind: " + ", ".join(f"{k} {sum(it['correct'] for it in items if it['kind'] == k)}/{sum(it['kind'] == k for it in items)}" for k in ("answer", "bridge"))
                                                 + f"; bridges excluded as leaks (P1): {sum(bool(it.get('bridge_leak')) and it.get('correct_pre_leak', False) for it in items)}")
    if a.max_items: kept = kept[: a.max_items]
    results = {"model": a.model, "dtype": a.dtype, "split": a.split, "n_ctx": len(ctx), "m_null": a.m_null, "pre_k": a.pre_k, "n_items": len(items), "n_correct": nc, "n_kept": len(kept), "gate": gate_rows, "layers": {}}
    done = set()
    if a.resume and os.path.exists(out_f):
        old = json.load(open(out_f)); results["layers"] = old.get("layers", {}); done = {(int(l), r["id"]) for l, s in results["layers"].items() for r in s["rows"]}; log(f"resuming: {len(done)} (layer, item) rows")
    layers = [int(x) for x in a.layers.split(",")]
    for l in layers:
        t0 = time.time(); D = token_matrix32(q.w, J[l]); H = background_activations(q, bg, l, mask); mu = H.mean(0)
        g = torch.Generator().manual_seed(l); nulls = (H - mu)[torch.randperm(H.shape[0], generator=g)[: a.m_null]]
        summ = results["layers"].setdefault(str(l), {"rows": []}); rows = summ["rows"]
        for n, it in enumerate(kept):
            if (l, it["id"]) in done: continue
            with torch.no_grad(): h = q.resid(it["prompt_ids"], l)[0, -1]
            p1, p2 = it["pieces"][0], it["pieces"][1]
            dh = D @ h; rank_p1 = int((dh > dh[p1]).sum()) + 1
            step = lambda u: pseudo_logits(q.w, torch.stack([jvp_exact(q, ids[c0:c0 + a.chunk], l, mask, u, prefix=[p1]) for c0 in range(0, len(ctx), a.chunk)]).mean(0))
            hc = h - mu; u = {"hc": hc, "h": h, "hJ": nnomp(D, hc, 25)[1] if a.inject == "hJ" else None}[a.inject]
            num = step(u); nn_ = torch.stack([step(gj) for gj in nulls])
            z = num / hutchinson_sigma(nn_); rank_raw = int((num > num[p2]).sum()) + 1; rank_z = int((z > z[p2]).sum()) + 1
            rows.append({"id": it["id"], "kind": it["kind"], "category": it["category"], "name": it["name"], "string": it["string"], "pieces": it["pieces_text"], "gated": it["gated"],
                         "rank_p1_lens": rank_p1, "rank_p1_lens_c": int(((D @ hc) > (D @ hc)[p1]).sum()) + 1, "inject": a.inject, "rank_p2_raw": rank_raw, "rank_p2_z": rank_z, "z_p2": float(z[p2]),
                         "top5_lens": [tok.decode([t]) for t in torch.topk(dh, 5).indices.tolist()],
                         "top5_raw": [tok.decode([t]) for t in torch.topk(num, 5).indices.tolist()], "top5_z": [tok.decode([t]) for t in torch.topk(z, 5).indices.tolist()]})
            log(f"L{l} {n+1}/{len(kept)} {it['string']!r} [{it['kind']}]: p1 lens rank {rank_p1}, p2 rank raw {rank_raw} z {rank_z}; top5 z {rows[-1]['top5_z']}")
            summ.update(summarise(rows, a.pre_k), seconds=time.time() - t0); json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
        summ.update(summarise(rows, a.pre_k)); results["any_layer"] = any_layer(results, a.pre_k); json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
        if summ["n"]: log(f"L{l}: n {summ['n']}; second piece top-1 z {summ['p2_top1_z']:.2f} raw {summ['p2_top1_raw']:.2f} (pass >= 0.5, kill < 0.3), top-10 z {summ['p2_top10_z']:.2f}, "
                          f"median rank z {summ['p2_median_rank_z']:.0f}; first piece in lens top-{a.pre_k}: {summ['n_pre']} items ({summ['p1_in_lens_topk']:.2f}); "
                          f"given that: top-1 z {summ['p2_top1_z_given_pre']} raw {summ['p2_top1_raw_given_pre']}, top-10 z {summ['p2_top10_z_given_pre']}; {summ['seconds']:.0f}s")
    al = results.get("any_layer", {"n": 0})
    if al["n"]: log(f"any-layer (best-p1 layer per item; a selection): n {al['n']}, p2 top-1 z {al['p2_top1_z']:.2f}, top-10 z {al['p2_top10_z']:.2f}; "
                    f"n with p1 in top-{a.pre_k} at that layer {al['n_pre']}; given that: top-1 z {al['p2_top1_z_given_pre']}, top-10 z {al['p2_top10_z_given_pre']}; layer hist {al['layer_hist']}")


if __name__ == "__main__":
    main()
