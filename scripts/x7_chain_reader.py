"""X7 / P3: an UNCONDITIONAL, prompt-blind multi-token "chain reader", scored WorkspaceBench-style, beside the
ORIGINAL released J-lens and the logit lens in the SAME output format on the SAME items with the SAME scoring.

The reader sees only residual activations at the item's last prompt token (never the prompt, never s1):
  h_read  at the READ layer l (--layers), where the gap-1 boost of X1 is read;
  h_first at the FIRST layer (--first-layer: "same" = l, or a deeper block such as 26 / 36), where the first piece is
          read by a plain single-vector reader (--first logitlens | jlens (bench cosine readout) | jlens_raw).
Step 1  first-piece candidates = top-k1 (--k1 5) of the first reader over the allowed vocabulary (no special tokens,
        no whitespace-only / newline tokens, no pure-punctuation tokens except - ' . / &; ALLOWED_PUNCT).
Step 2+ for every partial string p (a token list), the next-piece candidates are the top-b (--beam 5) tokens by the
        gap-1 BOOST at the read layer: Delta log p(. | c, p) = mean over 64 generic carriers of log p with h_read
        REPLACING the carrier residual at its last context position (X1's finite replace arm, delta_logp) minus the
        clean value. Beam width --beam over all partial strings, up to --max-pieces (4). Beam (search) score S = the
        SUM of the per-step boosts Delta log p (nats; the a-priori choice, A11) - the first piece contributes 0 and ties
        are broken by the first reader's rank. A partial string whose last boost is < tau_min = 0 is never extended.
        Every prefix the search visits is STORED (ids, text, S, per-step boosts, first rank), so any ranking rule can
        be applied offline.
Ranking --rank last | sum | mean (A11c; default `last`): the rule that orders the stored prefixes into samples.
        `sum` = S (the registered A11 rule: it prefers longer prefixes, exact N = 10 = 0.000 on both tuning tracks);
        `last` = the LAST step's boost of a multi-piece prefix (A11c, chosen on the tuning half: A .050 / B .155);
        `mean` = the mean step boost. A single-token prefix (no step boosts) scores its S (= 0 for P3; the plain
        chains' rank-coded S) under every rule; ties break by the first reader's rank, then by length (shorter).
Samples every stored prefix string is a candidate "sample". N = 10: the first reader's top-1 token alone, then the
        nine best distinct prefix strings under --rank (the bench scores any sample of a cell). N = 1: the best prefix
        under --rank whose EVERY step has Delta >= tau (the stop threshold, OPTIONAL: fixed on the TUNING half with
        --fit-tau, applied at summary time with --tau / --tau-from / --rescore; with no tau the best multi-piece prefix
        under --rank), else the first reader's top-1 token alone. No space-initial stop: word phrases and the bench's own
        targets ("Aghlabid dynasty") are multi-word, so the stop is the threshold, the token filter and the length cap.
        --rescore --rank R [--tau t] re-ranks and re-scores a finished run from its stored prefixes on the CPU.
Scoring exact string after the bench's normalisation (fold: NFKD, combining marks dropped, casefold; whitespace
        collapsed; strip) against the item's string; per layer and ANY layer over the run layers (the bench's
        any-layer rule). Controls: shuffled-h (same category, fixed derangement, A1) and crosscat (A2) through the whole
        reader (both h_read and h_first from the partner). The "s1 fixes s2" split uses the clean carrier + TRUE s1
        distribution (rank_s2_clean, the teacher-forced X1 row that is also logged per item as the X1 continuity check).
Baselines (same items, same layers, same scoring, same row format; the ORIGINAL J-lens first):
  jlens_bag10 / logitlens_bag10  the bench's token-lens arm: the top-10 tokens of the released J-lens (cosine readout
        (W_U J h) / ||J^T W_U[t]||, produce/methods.py::JLens) or of the logit lens, each token ONE sample. Exact-string
        scoring lets it pass only single-token strings (the bench's raw-regex asymmetry); it is reported anyway.
  jlens_chain / logitlens_chain  the plain-lens "ordered bag" chain: first candidates = top-k1 of the lens, next pieces
        = the remaining top-10 bag tokens in rank order (no prefix conditioning exists for a plain lens), prefixes up to
        --max-pieces; N = 10 samples by (first rank, length); N = 1 = the ordered top-1 + top-2 pair. When --first-layer
        is not "same", the cross-layer variant (*_chain_x: first pieces from the FIRST layer's reader, continuation from
        the READ layer's bag) shares P3's first-piece source and differs from P3 only in the continuation reader.
  X3 identity2 exact strings are reported from the wave-2 runs (P1c), not re-run here.
Pass / kill (A11): pass if P3 exact (N = 10, any layer) >= 0.30 on track B and >= 0.15 on track A with both controls
<= 0.05; kill if P3 exact (N = 10, any layer) <= the best ORIGINAL J-lens baseline (jlens_bag10, jlens_chain,
jlens_chain_x) + 0.05 on BOTH tracks. The same-layer single-vector variant (--first-layer same) is the bench-faithful
secondary; the cross-layer variant is the primary (a two-vector reader, declared as such).
--backend hf runs the same reader through sjlens/model/backend.py::HFHooksBackend (Qwen3.6-27B); --backend min is
Qwen3Min (bit-identical to X1 / X3). results.json is written after every item (--resume).
"""
import argparse, importlib.util, json, math, os, re, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.lens import jlens
from sjlens.eval.common import load_q, load_hf
from sjlens.eval.phase_a import contexts, token_matrix32
from sjlens.model.backend import MinBackend, HFHooksBackend
from sjlens.wsbench_regex import fold, jlens_cosine_scores
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
_load = lambda n, f: (lambda s: (lambda mod: (s.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s)))(importlib.util.spec_from_file_location(n, os.path.join(HERE, f)))
e12 = _load("e12", "e12_second_token.py"); x1 = _load("x1", "x1_gap1_patch.py")
PASS_B, PASS_A, CTRL_MAX, KILL_MARGIN = 0.30, 0.15, 0.05, 0.05  # A11 (P3)
ALLOWED_PUNCT = "-'’./&"
TAU_DEFAULT = None  # no stop threshold unless --tau / --tau-from is given (N0: tau is optional; --fit-tau writes tau.json)
TAU_GRID = [round(0.25 * i, 2) for i in range(0, 25)]
RANKS = ("last", "sum", "mean"); RANK_DEFAULT = "last"  # A11c


def rank_score(p, rank=RANK_DEFAULT):
    """the ranking score of a stored prefix under --rank: sum = S; last = the last step boost; mean = the mean step
    boost; a prefix without step boosts (a single first-piece token, or a plain-chain prefix) scores its S under
    every rule."""
    if rank not in RANKS: raise ValueError(f"--rank {rank!r}: one of {RANKS}")
    d = p.get("d") or []
    if rank == "sum" or not d: return p["S"]
    return d[-1] if rank == "last" else sum(d) / len(d)


def _order_key(rank):
    return lambda x: (-rank_score(x, rank), x["fr"], len(x["ids"]))


def normexact(s):
    """the bench's fold (NFKD, no combining marks, casefold) + whitespace collapse + strip: the exact-string key."""
    return re.sub(r"\s+", " ", fold(s)).strip()


def allowed_tokens(tok, V):
    """[V] bool: tokens a chain may emit (see the module docstring)."""
    ok = torch.ones(V, dtype=torch.bool); special = set(getattr(tok, "all_special_ids", []) or [])
    for t in range(V):
        if t in special: ok[t] = False; continue
        s = tok.decode([t])
        st = s.strip()
        if not st or "\n" in s or "\r" in s or not any(ch.isalnum() or ch in ALLOWED_PUNCT for ch in st): ok[t] = False
    return ok


def first_scores(kind, backend, h, D=None):
    """[V] float64 scores of the first-piece reader from one vector: logitlens = lm_head(norm(h)); jlens = the bench's
    cosine J-lens readout; jlens_raw = D h (the wave-2 X5 convention)."""
    if kind == "logitlens": return backend.logits(h).double()
    if kind == "jlens": return jlens_cosine_scores(D, h.float()).double()
    if kind == "jlens_raw": return (D @ h.float()).double()
    raise ValueError(kind)


def clean_mean(backend, ctx_ids, suffix, chunk=8):
    """mean over carriers of the CLEAN log p(. | c, suffix) at the last position: [V] float64 (x1.clean_logp)."""
    C = ctx_ids.shape[0]; out = None
    suf = torch.tensor(list(suffix), dtype=torch.long, device=ctx_ids.device).view(1, -1)
    for c0 in range(0, C, chunk):
        cid = ctx_ids[c0:c0 + chunk]; ids = torch.cat([cid, suf.expand(cid.shape[0], -1)], 1) if suf.numel() else cid
        lp = backend.last_logp(ids).sum(0); out = lp if out is None else out + lp
    return out / C


def boost(backend, ctx_ids, layer, h, suffix, clean, xlast, chunk=8):
    """(Delta log p [V], patched mean log p [V]) with h REPLACING the carrier residual after block `layer` at the last
    context position (X1 finite replace, x1.delta_logp with target = source layer)."""
    C, T = ctx_ids.shape; out = None
    suf = torch.tensor(list(suffix), dtype=torch.long, device=ctx_ids.device).view(1, -1)
    for c0 in range(0, C, chunk):
        cid = ctx_ids[c0:c0 + chunk]; b = cid.shape[0]
        ids = torch.cat([cid, suf.expand(b, -1)], 1) if suf.numel() else cid
        m = torch.zeros(b, ids.shape[1], dtype=torch.bool, device=ids.device); m[:, T - 1] = True
        d = (h.view(1, -1).to(xlast.dtype) - xlast[c0:c0 + b]).view(b, 1, -1).expand(b, ids.shape[1], -1)
        lp = backend.last_logp(ids, (layer, m, d)).sum(0); out = lp if out is None else out + lp
    patched = out / C
    return patched - clean, patched


class CleanCache:
    """LRU of clean_mean by suffix tuple (shared across items and controls; suffixes repeat a lot on word phrases)."""
    def __init__(s, backend, ctx_ids, chunk, cap=512):
        s.b, s.ids, s.chunk, s.cap, s.d, s.hits, s.misses = backend, ctx_ids, chunk, cap, {}, 0, 0

    def get(s, suffix):
        k = tuple(int(x) for x in suffix)
        if k in s.d:
            s.hits += 1; v = s.d.pop(k); s.d[k] = v; return v
        s.misses += 1; v = clean_mean(s.b, s.ids, k, s.chunk); s.d[k] = v
        if len(s.d) > s.cap: s.d.pop(next(iter(s.d)))
        return v


def chain_read(backend, tok, ctx_ids, layer, h_read, first_z, ok, xlast, cache, k1=5, beam=5, max_pieces=4, chunk=8, tau_min=0.0):
    """The P3 reader on ONE activation pair (h_read at `layer`, first_z = first-reader scores from h_first). No prompt,
    no item text. Returns {"cands": [...], "prefixes": [...], "n_boosts": int}; every prefix carries its token ids, text,
    score S (sum of step boosts), the per-step boosts and the first candidate's rank (1 = best)."""
    V = first_z.shape[0]; z = first_z.clone(); z[~ok.to(z.device)] = -float("inf")
    top = torch.topk(z, k1)
    cands = [{"id": int(w), "t": tok.decode([int(w)]), "rank": i + 1, "score": float(v)} for i, (v, w) in enumerate(zip(top.values, top.indices))]
    prefixes = [{"ids": [c["id"]], "t": c["t"], "S": 0.0, "d": [], "fr": c["rank"]} for c in cands]
    beams = list(prefixes); n_boosts = 0; okd = ok.to(first_z.device)
    for step in range(2, max_pieces + 1):
        new = []
        for bm in beams:
            if bm["d"] and bm["d"][-1] < tau_min: continue
            cl = cache.get(bm["ids"]); dlp, _ = boost(backend, ctx_ids, layer, h_read, bm["ids"], cl, xlast, chunk); n_boosts += 1
            dd = dlp.clone(); dd[~okd] = -float("inf"); tk = torch.topk(dd, beam)
            for r, (v, w) in enumerate(zip(tk.values, tk.indices)):
                ids = bm["ids"] + [int(w)]
                new.append({"ids": ids, "t": tok.decode(ids), "S": bm["S"] + float(v), "d": bm["d"] + [float(v)], "fr": bm["fr"]})
        if not new: break
        new.sort(key=lambda x: (-x["S"], x["fr"], len(x["ids"])))
        prefixes += new; beams = new[:beam]
    return {"cands": cands, "prefixes": prefixes, "n_boosts": n_boosts}


def samples_topn(prefixes, n, rank=RANK_DEFAULT):
    """the N-sample set: the first reader's top-1 token alone is ALWAYS sample 1 (so a one-token string is never pushed
    out by longer positive-boost strings), then the best distinct prefix strings under --rank (ties: first rank, then
    shorter)."""
    out, seen = [], set()
    firsts = [p for p in prefixes if not p["d"]]
    order = ([min(firsts, key=lambda x: x["fr"])] if firsts else []) + sorted(prefixes, key=_order_key(rank))
    for p in order:
        k = normexact(p["t"])
        if k in seen: continue
        seen.add(k); out.append(p["t"])
        if len(out) == n: break
    return out


def sample_top1(prefixes, tau, rank=RANK_DEFAULT):
    """the N = 1 sample: the best multi-piece prefix under --rank whose every step boost is >= tau (tau None = no
    threshold), else the first reader's top-1 token alone."""
    valid = [p for p in prefixes if p["d"] and (tau is None or all(v >= tau for v in p["d"]))]
    if valid: return min(valid, key=_order_key(rank))["t"]
    firsts = [p for p in prefixes if not p["d"]]
    return min(firsts, key=lambda x: x["fr"])["t"] if firsts else ""


def ordered_bag_chain(tok, first_z, cont_z, ok, k1=5, max_pieces=4, bag=10):
    """the plain-lens chain: first candidates = top-k1 of first_z; continuation = the top-`bag` tokens of cont_z in rank
    order, skipping the first candidate; prefixes up to max_pieces. Returns (cands, prefixes) in chain_read's format
    (S = -(first rank) - 0.01 * length so that samples_topn orders by first rank then length; d = [] so that
    sample_top1 is defined separately: the ordered top-1 + top-2 pair)."""
    z = first_z.clone(); z[~ok.to(z.device)] = -float("inf"); top = torch.topk(z, k1)
    zc = cont_z.clone(); zc[~ok.to(zc.device)] = -float("inf"); cont = [int(w) for w in torch.topk(zc, bag).indices]
    cands, prefixes = [], []
    for i, (v, w) in enumerate(zip(top.values, top.indices)):
        c = int(w); cands.append({"id": c, "t": tok.decode([c]), "rank": i + 1, "score": float(v)})
        ids = [c]; prefixes.append({"ids": list(ids), "t": tok.decode(ids), "S": -(i + 1) - 0.01, "d": [], "fr": i + 1})
        for w2 in [x for x in cont if x != c][: max_pieces - 1]:
            ids = ids + [w2]; prefixes.append({"ids": list(ids), "t": tok.decode(ids), "S": -(i + 1) - 0.01 * len(ids), "d": [], "fr": i + 1})
    return cands, prefixes


def bag_pair_top1(prefixes):
    """the plain chain's N = 1 sample: the first candidate of rank 1 followed by the bag's next token (two pieces)."""
    two = [p for p in prefixes if p["fr"] == 1 and len(p["ids"]) == 2]
    return two[0]["t"] if two else next((p["t"] for p in prefixes if p["fr"] == 1), "")


def gate_backend(backend, tok, it, ids):
    """e12.gate for answer items through a backend (greedy continuation must reproduce `pieces`)."""
    g = backend.greedy(ids, len(it["pieces"])); return tuple(g) == tuple(it["pieces"]), True, tok.decode(g)


def _score_row(row, tau, rank=RANK_DEFAULT):
    """exact flags of one reader row under (tau, rank), recomputed from the STORED prefixes (CPU-only rescoring): the
    N-sample set (n = row["n_samples"], default 10) and the N = 1 sample are rebuilt under `rank`."""
    key = row["key"]; pre = row["prefixes"]; n = int(row.get("n_samples") or 10)
    top10 = samples_topn(pre, n, rank); top1 = sample_top1(pre, tau, rank) if row["reader"] == "p3" else bag_pair_top1(pre)
    return {"exact_top1": int(normexact(top1) == key), "exact_top10": int(any(normexact(s) == key for s in top10)),
            "exact_anyprefix": int(any(normexact(p["t"]) == key for p in pre)), "sample_top1": top1, "samples10": top10}


def summarise(rows, tau, seed=0, rank=RANK_DEFAULT):
    """per (reader, control): exact N = 1 / N = 10 / any-prefix rates with CIs, first-piece rates, paired real - control
    (top-10 and top-1) with McNemar, split by s1-fixes-s2 (clean rank <= 1) and clean rank <= 10, per category."""
    out = {}
    for r in rows: out.setdefault(r["reader"], {}).setdefault(r["control"], []).append(r)
    summ = {}
    for reader, byc in out.items():
        summ[reader] = {}
        for ctrl, rs in byc.items():
            sc = [_score_row(r, tau, rank) for r in rs]; e = {"n": len(rs), "tau": tau, "rank": rank}
            for k in ("exact_top1", "exact_top10", "exact_anyprefix"):
                e[k], e[k + "_lo"], e[k + "_hi"] = x1.boot([float(s[k]) for s in sc], lambda t: t.mean(), seed=seed)
            e["first_top1"] = sum(r["first_hit_rank"] == 1 for r in rs) / len(rs); e["first_topk"] = sum(r["first_hit_rank"] is not None for r in rs) / len(rs)
            for name, sel in (("s1_fixes_s2", lambda r: r.get("rank_s2_clean") is not None and r["rank_s2_clean"] <= 1),
                              ("s1_not_fix", lambda r: r.get("rank_s2_clean") is not None and r["rank_s2_clean"] > 1),
                              ("clean_le10", lambda r: r.get("rank_s2_clean") is not None and r["rank_s2_clean"] <= 10),
                              ("clean_gt10", lambda r: r.get("rank_s2_clean") is not None and r["rank_s2_clean"] > 10)):
                sub = [s for r, s in zip(rs, sc) if sel(r)]
                e["split_" + name] = {"n": len(sub), "exact_top10": sum(s["exact_top10"] for s in sub) / len(sub) if sub else None, "exact_top1": sum(s["exact_top1"] for s in sub) / len(sub) if sub else None}
            per = {}
            for r, s in zip(rs, sc): per.setdefault(r["category"], []).append(s)
            e["by_category"] = {c: {"n": len(v), "exact_top10": sum(s["exact_top10"] for s in v) / len(v), "exact_top1": sum(s["exact_top1"] for s in v) / len(v)} for c, v in sorted(per.items())}
            if reader == "p3":
                tf = [r["rank_s2_tf"] for r in rs if r.get("rank_s2_tf") is not None]
                if tf: e["tf_s2_top10"] = sum(x <= 10 for x in tf) / len(tf); e["tf_s2_top1"] = sum(x <= 1 for x in tf) / len(tf)
                e["mean_boosts_per_item"] = sum(r["n_boosts"] for r in rs) / len(rs)
            summ[reader][ctrl] = e
        if "none" in byc:
            real = {r["id"]: _score_row(r, tau, rank) for r in byc["none"]}
            for ctrl, rs in byc.items():
                if ctrl == "none": continue
                pairs = [(real[r["id"]], _score_row(r, tau, rank)) for r in rs if r["id"] in real]; p = {"n_pairs": len(pairs)}
                for k in ("exact_top10", "exact_top1"):
                    a = [float(x[k]) for x, y in pairs]; b = [float(y[k]) for x, y in pairs]
                    if a: p[k + "_diff"], p[k + "_diff_lo"], p[k + "_diff_hi"] = x1.paired_boot(a, b, seed=seed); p[k + "_mcnemar"] = x1.mcnemar(a, b)
                summ[reader][ctrl]["paired"] = p
    return summ


def any_layer(results, tau, seed=0, rank=RANK_DEFAULT):
    """the bench's any-layer rule: an item passes a reader / control if it passes at ANY run layer."""
    flags = {}
    for l, s in results["layers"].items():
        for r in s["rows"]:
            sc = _score_row(r, tau, rank); k = (r["reader"], r["control"]); f = flags.setdefault(k, {}).setdefault(r["id"], {"exact_top1": 0, "exact_top10": 0, "exact_anyprefix": 0, "category": r["category"]})
            for kk in ("exact_top1", "exact_top10", "exact_anyprefix"): f[kk] = max(f[kk], sc[kk])
    out = {}
    for (reader, ctrl), d in flags.items():
        e = {"n": len(d)}
        for kk in ("exact_top1", "exact_top10", "exact_anyprefix"):
            e[kk], e[kk + "_lo"], e[kk + "_hi"] = x1.boot([float(v[kk]) for v in d.values()], lambda t: t.mean(), seed=seed)
        out.setdefault(reader, {})[ctrl] = e
    for reader, byc in out.items():
        if "none" in byc:
            real = flags[(reader, "none")]
            for ctrl in byc:
                if ctrl == "none": continue
                d = flags[(reader, ctrl)]; ids = [i for i in d if i in real]
                if ids:
                    a = [float(real[i]["exact_top10"]) for i in ids]; b = [float(d[i]["exact_top10"]) for i in ids]
                    diff, lo, hi = x1.paired_boot(a, b, seed=seed); byc[ctrl]["paired"] = {"n_pairs": len(ids), "exact_top10_diff": diff, "exact_top10_diff_lo": lo, "exact_top10_diff_hi": hi, "exact_top10_mcnemar": x1.mcnemar(a, b)}
    out["layers"] = sorted(int(l) for l in results["layers"])
    return out


def verdict(al, track):
    """A11 P3 lines on the any-layer block; the kill is relative to the ORIGINAL J-lens baselines."""
    if "p3" not in al or "none" not in al["p3"]: return {"note": "no p3 rows"}
    p3 = al["p3"]["none"]["exact_top10"]; ctrl = max([al["p3"][c]["exact_top10"] for c in al["p3"] if c != "none"] or [0.0])
    base = {k: al[k]["none"]["exact_top10"] for k in ("jlens_bag10", "jlens_chain", "jlens_chain_x") if k in al and "none" in al[k]}
    best_j = max(base.values()) if base else 0.0
    line = PASS_B if track == "B" else PASS_A
    return {"track": track, "p3_exact_top10_any_layer": p3, "control_max": ctrl, "jlens_baselines": base, "best_jlens_baseline": best_j,
            "pass": bool(p3 >= line and ctrl <= CTRL_MAX), "pass_line": line, "kill_this_track": bool(p3 <= best_j + KILL_MARGIN),
            "rule": "A11: pass exact(N=10, any layer) >= 0.30 (B) / 0.15 (A) with both controls <= 0.05; kill <= best original J-lens baseline + 0.05 on BOTH tracks (cross-track call in the write-up)"}


def fit_tau(results, log=print, rank=RANK_DEFAULT):
    """choose tau on a TUNING run: the grid value maximising p3 exact N = 1 (control none, under `rank`) pooled over
    the run's layers."""
    rows = [r for l, s in results["layers"].items() for r in s["rows"] if r["reader"] == "p3" and r["control"] == "none"]
    best = None
    for tau in TAU_GRID:
        acc = sum(_score_row(r, tau, rank)["exact_top1"] for r in rows) / max(len(rows), 1)
        log(f"tau {tau:.2f}: exact top-1 {acc:.3f} (n {len(rows)})")
        if best is None or acc > best[1] + 1e-12: best = (tau, acc)
    return {"tau": best[0], "exact_top1": best[1], "n_rows": len(rows), "grid": TAU_GRID, "rank": rank, "rule": "argmax exact N=1 over the tuning rows; ties -> the smaller tau"}


def load_backend(kind, model, dtype, device):
    if kind == "min":
        tok, m, q = load_q(model, dtype, device); return tok, m, MinBackend(tok, q)
    wd = {"fp32": torch.float32, "bf16": torch.bfloat16, "bf16-mixed": torch.bfloat16}[dtype]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    try: m = AutoModelForCausalLM.from_pretrained(model, dtype=wd, device_map=device)
    except TypeError: m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=wd).to(device)
    m.eval()
    for p_ in m.parameters(): p_.requires_grad_(False)
    return tok, m, HFHooksBackend(m, tok)


def resummarise(results, tau, seed=0, rank=RANK_DEFAULT):
    """re-rank (every row's samples10 / exact flags / N = 1 sample are rebuilt from its stored prefixes under `rank`
    and `tau`) and re-summarise a run; records rank and tau in results."""
    for l, s in results["layers"].items():
        for r in s["rows"]: r.update(_score_row(r, tau, rank))
        s["summary"] = summarise(s["rows"], tau, seed, rank)
    results["any_layer"] = any_layer(results, tau, seed, rank); results["verdict"] = verdict(results["any_layer"], results.get("track")); results["tau"] = tau; results["rank"] = rank
    return results


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--items", default=os.path.join(os.path.dirname(HERE), "data", "h2a_items.json")); p.add_argument("--split", default="heldout")
    p.add_argument("--split-file", default=os.path.join(os.path.dirname(HERE), "data", "h2a_split.json"))
    p.add_argument("--layers", default="22", help="READ layers (the gap-1 boost)"); p.add_argument("--first", default="logitlens", help="first-piece reader: logitlens | jlens (bench cosine) | jlens_raw")
    p.add_argument("--target-layer", type=int, default=-1, help="block after which h is PATCHED into the carrier (-1 = the read layer, the registered default; e.g. 4: early-target patch, A11b)")
    p.add_argument("--first-layer", default="same", help="same | block index of the FIRST-piece vector (e.g. 26 / 36: the two-vector primary reader)")
    p.add_argument("--k1", type=int, default=5); p.add_argument("--beam", type=int, default=5); p.add_argument("--max-pieces", type=int, default=4); p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--tau", type=float, default=None, help="stop threshold for the N = 1 sample (OPTIONAL; none = the best prefix under --rank)"); p.add_argument("--tau-from", default="", help="tau.json written by --fit-tau on a tuning run (optional)")
    p.add_argument("--rank", default=RANK_DEFAULT, choices=RANKS, help="sample ranking rule (A11c): last = last step boost (default), sum = registered A11 sum, mean")
    p.add_argument("--fit-tau", action="store_true", help="CPU: choose tau on runs/<tag>/results.json (a TUNING run) under --rank and write runs/<tag>/tau.json")
    p.add_argument("--rescore", action="store_true", help="CPU: re-rank (--rank) and re-summarise runs/<tag>/results.json under --tau / --tau-from from the stored prefixes")
    p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=64); p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--controls", default="", help="comma list of shuffled,crosscat,random"); p.add_argument("--kinds", default="answer")
    p.add_argument("--min-p1-chars", type=int, default=0); p.add_argument("--per-category", type=int, default=0); p.add_argument("--max-items", type=int, default=0); p.add_argument("--max-scan", type=int, default=0)
    p.add_argument("--shard", default=""); p.add_argument("--summarise", default="", help="comma list of run tags to merge (shards)"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--track", default=""); p.add_argument("--dtype", default="fp32"); p.add_argument("--device", default="cuda"); p.add_argument("--backend", default="min", help="min (Qwen3Min) | hf (forward hooks; Qwen3.6-27B)")
    p.add_argument("--contexts", default="runs/contexts/qwen3_T128_256.pt"); p.add_argument("--tag", default="x7_chain"); p.add_argument("--resume", action="store_true")
    p.add_argument("--no-teacher-forced", action="store_true", help="skip the X1 continuity row (boost with the TRUE s1)")
    a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(HERE), "runs", a.tag); os.makedirs(out_dir, exist_ok=True); out_f = os.path.join(out_dir, "results.json")
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True); log(" ".join(sys.argv))
    tau = a.tau if a.tau is not None else (json.load(open(a.tau_from))["tau"] if a.tau_from else TAU_DEFAULT)  # None = no threshold (tau optional)
    log(f"rank {a.rank} (A11c default: last); tau {tau}")
    if a.fit_tau:
        r = json.load(open(out_f)); ft = fit_tau(r, log, a.rank); json.dump(ft, open(os.path.join(out_dir, "tau.json"), "w"), indent=1); log(f"tau {ft['tau']} (rank {a.rank}) -> {out_dir}/tau.json"); return
    if a.rescore or a.summarise:
        if a.summarise:
            merged = None
            for tag in [t for t in a.summarise.split(",") if t]:
                r = json.load(open(os.path.join(os.path.dirname(HERE), "runs", tag, "results.json")))
                if merged is None: merged = {k: v for k, v in r.items() if k != "layers"}; merged["layers"] = {}; merged["merged_from"] = []
                merged["merged_from"].append(tag)
                for l, s in r["layers"].items():
                    m_ = merged["layers"].setdefault(l, {"rows": []}); seen = {(x["id"], x["reader"], x["control"]) for x in m_["rows"]}
                    m_["rows"] += [x for x in s["rows"] if (x["id"], x["reader"], x["control"]) not in seen]
            r = merged
        else: r = json.load(open(out_f))
        resummarise(r, tau, a.seed, a.rank); json.dump(r, open(out_f, "w"), indent=1, ensure_ascii=False)
        v = r["verdict"]; log(f"rescored under rank {a.rank}, tau {tau}: p3 any-layer exact top-10 {v.get('p3_exact_top10_any_layer')} vs J-lens baselines {v.get('jlens_baselines')} -> pass {v.get('pass')} kill {v.get('kill_this_track')}"); return

    tok, m, backend = load_backend(a.backend, a.model, a.dtype, a.device)
    dev = backend.device
    J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))
    W_U = backend.unembed()
    ctx, _bg = contexts(tok, a.n_ctx, 8, a.T, dev, path=a.contexts if os.path.exists(a.contexts) else None); ids = torch.stack(ctx)
    items = e12.split_items(e12.subset(e12.load_items(tok, a.items), a), a.split_file, a.split)
    if a.max_scan: items = items[: a.max_scan]
    kept = []
    for it in items:
        pid = tok(it["prompt"], return_tensors="pt").input_ids.to(dev)
        it["correct"], it["gated"], it["model_greedy"] = gate_backend(backend, tok, it, pid); it["prompt_ids"] = pid
        if it["correct"] and it["gated"]: kept.append(it)
    if a.max_items: kept = kept[: a.max_items]
    log(f"{len(items)} items ({a.split}); {len(kept)} gated")
    ok = allowed_tokens(tok, backend.V); log(f"allowed vocabulary: {int(ok.sum())} of {backend.V}")
    layers = [int(x) for x in a.layers.split(",")]; ctrls = ["none"] + [c for c in a.controls.split(",") if c]
    shard = [int(x) for x in a.shard.split("/")] if a.shard else None; run_idx = [n for n in range(len(kept)) if shard is None or n % shard[1] == shard[0]]
    strings = [it["string"] for it in kept]; cats = [it["category"] for it in kept]
    perm = e12.shuffled_perm(cats, a.seed, strings); xperm = e12.cross_perm(cats, a.seed, strings)
    results = {"model": a.model, "backend": backend.name, "dtype": a.dtype, "split": a.split, "track": e12.track_of(a.items, a.track), "n_ctx": len(ctx), "T": a.T, "seed": a.seed,
               "first": a.first, "first_layer": a.first_layer, "k1": a.k1, "beam": a.beam, "max_pieces": a.max_pieces, "n_samples": a.n_samples, "tau": tau, "tau_from": a.tau_from, "rank": a.rank,
               "controls": ctrls, "n_items": len(items), "n_kept": len(kept), "n_run": len(run_idx), "shard": a.shard, "n_shuffled_dropped": int((perm < 0).sum()), "n_crosscat_dropped": int((xperm < 0).sum()),
               "readers": ["p3", "jlens_bag10", "logitlens_bag10", "jlens_chain", "logitlens_chain"] + (["jlens_chain_x", "logitlens_chain_x"] if a.first_layer != "same" else []),
               "prereg": "docs/prereg_R.yaml amendment A11 (P3)", "layers": {}}
    done = set()
    if a.resume and os.path.exists(out_f):
        old = json.load(open(out_f)); results["layers"] = old.get("layers", {}); done = {(int(l), r["id"], r["reader"], r["control"]) for l, s in results["layers"].items() for r in s["rows"]}; log(f"resuming: {len(done)} rows")
    # residuals at every needed layer (read layers + the first layer), one forward per item per layer
    need = sorted(set(layers) | ({int(a.first_layer)} if a.first_layer != "same" else set()))
    if a.first_layer != "same" and int(a.first_layer) not in J:  # the released lens file may stop before the deepest block (1.7B: 0..26)
        assert a.first != "jlens" and a.first != "jlens_raw", f"--first {a.first} at layer {a.first_layer}: the lens file has no layer {a.first_layer} (has {min(J)}..{max(J)})"
        results["readers"] = [r for r in results["readers"] if r != "jlens_chain_x"]; log(f"lens file has no layer {a.first_layer}: jlens_chain_x dropped (logitlens_chain_x kept)")
    H = {l: {} for l in need}
    for it in kept:
        for l in need: H[l][it["id"]] = backend.resid(it["prompt_ids"], l)[0, -1].float()
    D = {}
    def Dl(l):
        if l not in D: D[l] = W_U @ J[l].float().to(W_U.device)  # = phase_a.token_matrix32(w, J[l]): D_l = W_U J_l in fp32
        return D[l]
    for l in layers:
        t0 = time.time(); lf = l if a.first_layer == "same" else int(a.first_layer)
        tl = l if a.target_layer < 0 else a.target_layer  # injection layer of the boost (the READ vector h still comes from layer l)
        xlast = backend.resid(ids, tl)[:, a.T - 1].clone(); cache = CleanCache(backend, ids, a.chunk)
        summ = results["layers"].setdefault(str(l), {"rows": [], "first_layer": lf, "target_layer": tl}); rows = summ["rows"]
        g = torch.Generator().manual_seed(a.seed + l)
        for n in range(len(kept)):
            it = kept[n]; h = H[l][it["id"]]; hf = H[lf][it["id"]]
            r = torch.randn(h.shape, generator=g, dtype=torch.float32).to(h.device); rf = torch.randn(hf.shape, generator=g, dtype=torch.float32).to(hf.device)
            if n not in run_idx: continue
            hs = {"none": (h, hf), "random": (r / r.norm() * h.norm(), rf / rf.norm() * hf.norm())}
            if int(perm[n]) >= 0: pj = kept[int(perm[n])]["id"]; hs["shuffled"] = (H[l][pj], H[lf][pj])
            if int(xperm[n]) >= 0: pj = kept[int(xperm[n])]["id"]; hs["crosscat"] = (H[l][pj], H[lf][pj])
            key = normexact(it["string"]); p1key = normexact(it["pieces_text"][0])
            base = {"id": it["id"], "category": it["category"], "string": it["string"], "pieces_text": it["pieces_text"], "p1_chars": it.get("p1_chars"), "key": key, "layer": l, "first_layer": lf}
            # the X1 continuity / s1-fixes-s2 row: clean carrier + TRUE s1, and the boost of s2 with the real h (teacher-forced)
            tf = {}
            if len(it["pieces"]) >= 2:
                cl1 = cache.get([it["pieces"][0]]); tf["rank_s2_clean"] = x1.rank_of(cl1, it["pieces"][1]); tf["logp_s2_clean"] = float(cl1[it["pieces"][1]])
                if not a.no_teacher_forced:
                    dlp1, pat1 = boost(backend, ids, tl, h, [it["pieces"][0]], cl1, xlast, a.chunk); tf["rank_s2_tf"] = x1.rank_of(dlp1, it["pieces"][1]); tf["rank_s2_abs_tf"] = x1.rank_of(pat1, it["pieces"][1]); tf["dlp_s2_tf"] = float(dlp1[it["pieces"][1]])
            for ctrl in ctrls:
                if ctrl not in hs: continue
                u, uf = hs[ctrl]
                todo = [rd for rd in results["readers"] if (l, it["id"], rd, ctrl) not in done]
                if not todo: continue
                zf = first_scores(a.first, backend, uf, Dl(lf) if a.first != "logitlens" else None)
                z_j, z_lg = jlens_cosine_scores(Dl(l), u).double(), backend.logits(u).double()
                zf_lg = backend.logits(uf).double() if lf != l else z_lg; zf_j = (jlens_cosine_scores(Dl(lf), uf).double() if lf in J else None) if lf != l else z_j
                outs = {}
                if "p3" in todo:
                    res = chain_read(backend, tok, ids, tl, u, zf, ok, xlast, cache, a.k1, a.beam, a.max_pieces, a.chunk)
                    outs["p3"] = (res["cands"], res["prefixes"], res["n_boosts"])
                for rd, (zf_, zc_) in (("jlens_bag10", (z_j, z_j)), ("logitlens_bag10", (z_lg, z_lg)), ("jlens_chain", (z_j, z_j)), ("logitlens_chain", (z_lg, z_lg)), ("jlens_chain_x", (zf_j, z_j)), ("logitlens_chain_x", (zf_lg, z_lg))):
                    if rd not in todo or zf_ is None: continue
                    if rd.endswith("bag10"):
                        zz = zf_.clone(); zz[~ok.to(zz.device)] = -float("inf"); top = torch.topk(zz, 10)
                        cands = [{"id": int(w), "t": tok.decode([int(w)]), "rank": i + 1, "score": float(v)} for i, (v, w) in enumerate(zip(top.values, top.indices))]
                        outs[rd] = (cands, [{"ids": [c["id"]], "t": c["t"], "S": -c["rank"], "d": [], "fr": c["rank"]} for c in cands], 0)
                    else:
                        cands, pre = ordered_bag_chain(tok, zf_, zc_, ok, a.k1, a.max_pieces); outs[rd] = (cands, pre, 0)
                for rd, (cands, pre, nb) in outs.items():
                    fr = next((c["rank"] for c in cands if normexact(c["t"]) == p1key), None)
                    row = {**base, **tf, "reader": rd, "control": ctrl, "cands": cands, "prefixes": pre, "n_boosts": nb, "first_hit_rank": fr, "n_samples": a.n_samples}
                    row.update(_score_row(row, tau, a.rank)); rows.append(row)
            shown = [r_ for r_ in rows if r_["id"] == it["id"] and r_["control"] == "none" and r_["reader"] == "p3"]
            if shown:
                r_ = shown[-1]; log(f"L{l} {run_idx.index(n)+1}/{len(run_idx)} {it['string']!r} ({it['category']}): p3 top1 {r_['sample_top1']!r} exact {r_['exact_top1']}/{r_['exact_top10']}; cands {[c['t'] for c in r_['cands']]}; "
                                  f"best {[(p['t'], round(p['S'], 2)) for p in sorted(r_['prefixes'], key=lambda x: -x['S'])[:3]]}; clean s2 rank {tf.get('rank_s2_clean')}; tf s2 rank {tf.get('rank_s2_tf')}; boosts {r_['n_boosts']}; cache {cache.hits}/{cache.misses}")
            summ["summary"] = summarise(rows, tau, a.seed, a.rank); summ["seconds"] = time.time() - t0
            json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
        summ["summary"] = summarise(rows, tau, a.seed, a.rank); summ["seconds"] = time.time() - t0
        results["any_layer"] = any_layer(results, tau, a.seed, a.rank); results["verdict"] = verdict(results["any_layer"], results["track"])
        json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
        for rd, byc in sorted(summ["summary"].items()):
            for ctrl, e in sorted(byc.items()):
                pr = e.get("paired", {}); pd = f", paired real-ctrl top-10 {pr['exact_top10_diff']:+.3f} [{pr['exact_top10_diff_lo']:+.3f}, {pr['exact_top10_diff_hi']:+.3f}]" if pr.get("exact_top10_diff") is not None else ""
                log(f"L{l} {rd:16s} [{ctrl}]: n {e['n']}, exact N=1 {e['exact_top1']:.3f} [{e['exact_top1_lo']:.3f}, {e['exact_top1_hi']:.3f}], N=10 {e['exact_top10']:.3f} [{e['exact_top10_lo']:.3f}, {e['exact_top10_hi']:.3f}], "
                    f"any-prefix {e['exact_anyprefix']:.3f}, first top-1 {e['first_top1']:.3f} top-k {e['first_topk']:.3f}; s1-fixes-s2 n {e['split_s1_fixes_s2']['n']} N=10 {e['split_s1_fixes_s2']['exact_top10']}; not n {e['split_s1_not_fix']['n']} {e['split_s1_not_fix']['exact_top10']}{pd}")
    v = results["verdict"]; log(f"any-layer verdict (A11 P3, rank {a.rank}, tau {tau}): p3 exact N=10 {v.get('p3_exact_top10_any_layer')}, control max {v.get('control_max')}, original J-lens baselines {v.get('jlens_baselines')}: pass {v.get('pass')}, kill (this track) {v.get('kill_this_track')}")


if __name__ == "__main__":
    main()
