"""X1 (R1 locality, R2 linearity): is s2 recoverable from h placed at GAP 1, and does the tangent suffice?

The SJ-lens perturbs sources 16..tau-2, so no sequence term contains gap 0 or gap 1 (audit C13 / plan 2.4); G8 killed
the far-gap version on both models. Here h sits at the carrier's LAST context position (index tau-1), which is gap 1
to the position that predicts s2 and gap 0 to the one that predicts s1 - where the bench reads h.

Per gated answer item (e12's load_items / subset / gate; --split heldout for headline numbers) and layer l:
  h      = residual at the last prompt token at layer l (the prompt never enters any arm below)
  arms   finite   : patch h into each carrier at tau-1 (mode add: delta = alpha * mean_src_norm / ||h|| * h, so
                    ||delta|| = alpha * mean carrier source-residual norm; mode replace: delta = h - x[tau-1];
                    mode replace_nm: h rescaled to the mean carrier residual norm at the TARGET layer, then replace),
                    append s1, and read dlp = log_softmax(lm_head(norm(h_L))) at the s1 position, patched - clean,
                    averaged over carriers. Rank of s2 over the whole vocabulary; also mean dlp(s2).
          linear  : jvp_step(mask = onehot(tau-1), prefix=[s1], mode lp) - the exact tangent of the SAME readout
                    (patch P4's jvp_gap; mode z gives the pseudo-logit version, reported beside it).
          fargap  : the G8 reference, jvp_step with the lens mask 16..tau-2 and prefix [s1], same items.
  controls shuffled: h of another gated item of the same category (fixed derangement, never the same string, C1);
                    crosscat: h of a gated item of a DIFFERENT category (cross-category derangement); random: a
                    norm-matched Gaussian direction. All run through every arm that --controls lists.
  frames  plain (the carrier as it is) | answer (" The answer is" inserted before s1, a question-like frame).

Wave 2 (ANALYSIS_DAY1 N1; docs/prereg_R.yaml amendments):
  --target-layer same,4,8,12,16   h read at the SOURCE layer l is patched into the carrier AFTER block l' (target),
                    still at position tau-1, so the s1 position can read it through blocks l'+1..L (C3). Arms of the
                    same-layer path keep their Day-1 keys (finite:replace:rep:plain); other targets append ":t<l'>".
  ranks per row     rank_s2 (rank of s2 in Delta log p, the pre-registered readout statistic), rank_s2_abs (rank of
                    s2 in the PATCHED mean log-prob: does the model now predict s2?), rank_s2_clean (rank of s2 in the
                    clean carrier + s1 distribution, no patch; identical across arms, computed once per item and
                    frame), logp_s2_patched, logp_s2_clean (C6).
  paired statistics real minus each control (top-10 and top-1, also for the absolute rank) with a paired item-level
                    bootstrap CI and exact McNemar counts, over the items that have both rows.
  --pieces 2,3      optional third-piece arm (":p3"): for items with >= 3 pieces, s1 s2 are appended and s3 is scored
                    (open question 5: answer buffer or one piece ahead). In ":p3" rows rank_s2 / dlp_s2 refer to s3.
  --shard i/n       run items i, i+n, ... of the gated set (partners and controls are drawn from the FULL gated set,
                    so shards are mergeable); --summarise tagA,tagB merges the rows of several runs into one summary.
results.json is written after every item (--resume skips finished (layer, arm, item) rows). The summary gives, per
layer / arm / alpha / mode / control, top-1, top-10, median rank and mean dlp(s2) with item-level bootstrap CIs,
overall and per category, and states pass / kill against docs/prereg_R.yaml (R1: pass top-10 >= 0.30 with controls
<= 0.05; kill top-10 <= 0.05 everywhere and within 2 points of the controls), plus the SECONDARY effect-size line
(amendment A6: real - same-category >= 0.20 with the paired CI lower bound >= 0.10, and real - random >= 0.30).
"""
import argparse, importlib.util, json, math, os, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min, forward, logits as true_logits
from sjlens.lens import jlens
from sjlens.lens.jlens import valid_mask
from sjlens.lens.forward import jvp_step
from sjlens.eval.common import load_q, background_activations
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("e12", os.path.join(HERE, "e12_second_token.py")); e12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e12)
PASS_TOP10, KILL_TOP10, CTRL_MAX = 0.30, 0.05, 0.05  # docs/prereg_R.yaml R1
SEC_GAP, SEC_LO, SEC_RANDOM = 0.20, 0.10, 0.30  # amendment A6 (secondary effect-size line, ANALYSIS_DAY1 section 3)


def frame_ids(tok, s1, frame):
    """the tokens appended to a carrier: [s1], or the question-like frame " The answer is" + s1."""
    return [s1] if frame == "plain" else tok.encode(" The answer is", add_special_tokens=False) + [s1]


F32 = lambda z: z.float() if z.dtype in (torch.bfloat16, torch.float16) else z  # as lens/forward._lsm: never downcast fp64


def clean_logp(q, ctx_ids, suffix, chunk=8):
    """mean over carriers of the CLEAN log p(. | c, suffix) at the last suffix position: [V]. It does not depend on the
    patched direction, so it is computed once per (item, frame) and shared by every arm and control."""
    C = ctx_ids.shape[0]; out = torch.zeros(q.w.lm.shape[0], dtype=torch.float64, device=ctx_ids.device)
    suf = torch.tensor(suffix, dtype=torch.long, device=ctx_ids.device).view(1, -1)
    for c0 in range(0, C, chunk):
        cid = ctx_ids[c0:c0 + chunk]
        with torch.no_grad():
            h0, _ = forward(q.w, torch.cat([cid, suf.expand(cid.shape[0], -1)], 1), act_dtype=q.act_dtype)
            out += F.log_softmax(F32(true_logits(q.w, h0[:, -1])), -1).double().sum(0)
    return out / C


def delta_logp(q, ctx_ids, layer, h, suffix, alpha=1.0, mode="add", hn=None, chunk=8, clean=None, xlast=None, target_layer=None, return_patched=False):
    """mean over carriers of log p(. | c, suffix) with h patched at the carrier's LAST context position (tau-1),
    minus the clean value: [V]. mode add: norm-matched addition (||delta|| = alpha * hn); replace: overwrite;
    replace_nm: overwrite with h * hn / ||h|| (hn = the mean carrier residual norm at the target layer).
    target_layer: the layer AFTER whose block the patch is applied (default: `layer`, the source layer; the source
    layer itself only names where h came from). clean: clean_logp for this suffix; xlast: [C, d] clean residual at
    tau-1 at the TARGET layer (by causality it does not depend on the appended suffix, so one per layer serves every
    item). Both are recomputed when not supplied. return_patched: also return the patched mean log-prob (= out + clean)."""
    tl = layer if target_layer is None else target_layer
    C, T = ctx_ids.shape; out = torch.zeros(q.w.lm.shape[0], dtype=torch.float64, device=ctx_ids.device)
    suf = torch.tensor(suffix, dtype=torch.long, device=ctx_ids.device).view(1, -1)
    if clean is None: clean = clean_logp(q, ctx_ids, suffix, chunk)
    if mode != "add" and xlast is None:
        with torch.no_grad(): xlast = q.resid(ctx_ids, tl)[:, T - 1]
    for c0 in range(0, C, chunk):
        cid = ctx_ids[c0:c0 + chunk]; b = cid.shape[0]
        ids = torch.cat([cid, suf.expand(b, -1)], 1)
        m = torch.zeros(b, ids.shape[1], dtype=torch.bool, device=ids.device); m[:, T - 1] = True
        with torch.no_grad():
            if mode == "add":
                d = (alpha * hn / h.norm().clamp_min(1e-12)) * h
            else:  # replace: delta = h - the clean residual at tau-1, per carrier (replace_nm: h rescaled to hn first)
                hh = h * (hn / h.norm().clamp_min(1e-12)) if mode == "replace_nm" else h
                d = (hh.view(1, -1) - xlast[c0:c0 + b]).view(b, 1, -1).expand(b, ids.shape[1], -1)
            hA, _ = forward(q.w, ids, inject=(tl, m, d.to(q.act_dtype or q.w.emb.dtype)), act_dtype=q.act_dtype)
            out += F.log_softmax(F32(true_logits(q.w, hA[:, -1])), -1).double().sum(0)
    patched = out / C
    return (patched - clean, patched) if return_patched else patched - clean


def rank_of(v, w):
    """rank of token w in v (mid-rank on ties)."""
    return float((v > v[w]).sum()) + 1 + float((v == v[w]).sum() - 1) / 2


def boot(xs, f, n=1000, seed=0):
    """(point, lo, hi) of f over an item-level bootstrap of the list xs."""
    if not xs: return None, None, None
    t = torch.tensor(xs, dtype=torch.float64); g = torch.Generator().manual_seed(seed)
    b = torch.stack([f(t[torch.randint(0, len(t), (len(t),), generator=g)]) for _ in range(n)])
    return float(f(t)), float(b.quantile(0.025)), float(b.quantile(0.975))


def paired_boot(a, b, n=1000, seed=0):
    """mean(a) - mean(b) over paired items with an item-level bootstrap CI: (diff, lo, hi)."""
    if not a: return None, None, None
    d = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64); g = torch.Generator().manual_seed(seed)
    bs = torch.stack([d[torch.randint(0, len(d), (len(d),), generator=g)].mean() for _ in range(n)])
    return float(d.mean()), float(bs.quantile(0.025)), float(bs.quantile(0.975))


def mcnemar(a, b):
    """exact McNemar test of paired binary outcomes: b_ = a hit & b miss, c_ = a miss & b hit, two-sided binomial p."""
    b_ = sum(int(x and not y) for x, y in zip(a, b)); c_ = sum(int(y and not x) for x, y in zip(a, b)); n = b_ + c_
    if n == 0: return {"b": b_, "c": c_, "p": 1.0}
    k = min(b_, c_); p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"b": b_, "c": c_, "p": p}


def _rank_block(ranks, seed, prefix=""):
    e = {prefix + "median_rank": float(torch.tensor(ranks).median())}
    e[prefix + "top1"], e[prefix + "top1_lo"], e[prefix + "top1_hi"] = boot(ranks, lambda t: (t <= 1).double().mean(), seed=seed)
    e[prefix + "top10"], e[prefix + "top10_lo"], e[prefix + "top10_hi"] = boot(ranks, lambda t: (t <= 10).double().mean(), seed=seed)
    return e


def summarise(rows, seed=0):
    """per (layer, arm, control) and per category: top-1, top-10, median rank, mean dlp(s2), with bootstrap CIs; the
    same three statistics for the absolute patched rank (rank_s2_abs) and the clean rank (rank_s2_clean) where the
    rows carry them; and, per control, the paired real-minus-control differences (top-10 / top-1, Delta-rank and
    absolute-rank) with paired bootstrap CIs and exact McNemar counts over the items that have both rows."""
    out = {}
    for r in rows: out.setdefault(r["arm"], {}).setdefault(r["control"], []).append(r)
    summ = {}
    for arm, byc in out.items():
        summ[arm] = {}
        for ctrl, rs in byc.items():
            ranks = [r["rank_s2"] for r in rs]; dlp = [r["dlp_s2"] for r in rs]
            e = {"n": len(rs), **_rank_block(ranks, seed)}
            e["mean_dlp_s2"], e["mean_dlp_s2_lo"], e["mean_dlp_s2_hi"] = boot(dlp, lambda t: t.mean(), seed=seed)
            for key, pre in (("rank_s2_abs", "abs_"), ("rank_s2_clean", "clean_")):
                v = [r[key] for r in rs if r.get(key) is not None]
                if v: e.update(_rank_block(v, seed, pre))
            lp = [r["logp_s2_patched"] for r in rs if r.get("logp_s2_patched") is not None]
            if lp: e["mean_logp_s2_patched"] = float(torch.tensor(lp).mean())
            per = {}
            for r in rs: per.setdefault(r["category"], []).append(r)
            e["by_category"] = {}
            for c, v in sorted(per.items()):
                rk = [x["rank_s2"] for x in v]; cb = {"n": len(v), "top1": sum(x <= 1 for x in rk) / len(v), "top10": sum(x <= 10 for x in rk) / len(v),
                                                       "median_rank": float(torch.tensor(rk).float().median())}
                ab = [x["rank_s2_abs"] for x in v if x.get("rank_s2_abs") is not None]
                if ab: cb["abs_top1"] = sum(x <= 1 for x in ab) / len(ab); cb["abs_top10"] = sum(x <= 10 for x in ab) / len(ab)
                e["by_category"][c] = cb
            summ[arm][ctrl] = e
        if "none" in byc:  # paired statistics: real minus each control over the items that have both rows
            real = {r["id"]: r for r in byc["none"] if r.get("id") is not None}
            for ctrl, rs in byc.items():
                if ctrl == "none": continue
                pairs = [(real[r["id"]], r) for r in rs if r.get("id") in real]
                p = {"n_pairs": len(pairs)}
                for key, pre in (("rank_s2", ""), ("rank_s2_abs", "abs_")):
                    pp = [(x, y) for x, y in pairs if x.get(key) is not None and y.get(key) is not None]
                    if not pp: continue
                    for k, name in ((10, "top10"), (1, "top1")):
                        a = [float(x[key] <= k) for x, y in pp]; b = [float(y[key] <= k) for x, y in pp]
                        p[pre + name + "_diff"], p[pre + name + "_diff_lo"], p[pre + name + "_diff_hi"] = paired_boot(a, b, seed=seed)
                        p[pre + name + "_mcnemar"] = mcnemar(a, b)
                summ[arm][ctrl]["paired"] = p
    return summ


def verdict(summ):
    """R1 pass / kill for one layer's summary (the cross-layer / cross-model call is made in the write-up), plus the
    SECONDARY effect-size line (amendment A6; labelled secondary, it does not replace the pre-registered verdict):
    real - same-category shuffled top-10 >= 0.20 with the paired bootstrap lower bound >= 0.10 and, when the random
    control was run, real - random >= 0.30. Holm across layers is applied in the write-up."""
    best, out = None, {}
    for arm, byc in summ.items():
        if arm.startswith("linear") or arm.startswith("fargap") or "none" not in byc: continue
        t10 = byc["none"]["top10"]; ctrl = max([byc[c]["top10"] for c in byc if c != "none"] or [0.0])
        out[arm] = {"top10": t10, "control_top10_max": ctrl, "pass": bool(t10 >= PASS_TOP10 and ctrl <= CTRL_MAX),
                    "kill": bool(t10 <= KILL_TOP10 and t10 - ctrl <= 0.02)}
        sec = {"label": "SECONDARY (amendment A6, not the pre-registered verdict)"}
        if "shuffled" in byc and "paired" in byc["shuffled"] and byc["shuffled"]["paired"].get("top10_diff") is not None:
            pr = byc["shuffled"]["paired"]; sec["gap_same_cat"] = pr["top10_diff"]; sec["gap_same_cat_lo"] = pr["top10_diff_lo"]
            ok = pr["top10_diff"] >= SEC_GAP and pr["top10_diff_lo"] >= SEC_LO
            if "random" in byc and "paired" in byc["random"] and byc["random"]["paired"].get("top10_diff") is not None:
                sec["gap_random"] = byc["random"]["paired"]["top10_diff"]; ok = ok and sec["gap_random"] >= SEC_RANDOM
            sec["pass"] = bool(ok)
        else: sec["pass"] = None
        out[arm]["secondary"] = sec
        if best is None or t10 > best[1]: best = (arm, t10)
    return {"arms": out, "best_arm": best[0] if best else None, "any_pass": any(v["pass"] for v in out.values()),
            "all_kill": bool(out) and all(v["kill"] for v in out.values()),
            "any_secondary_pass": any(v["secondary"].get("pass") for v in out.values())}


def parse_targets(s):
    """--target-layer list: 'same' (the source layer) or ints; 'same,4,8'. '-1' means same (old x3 convention)."""
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x: continue
        out.append("same" if x in ("same", "-1") else int(x))
    return out or ["same"]


def arm_name(mode, alpha, frame, target, piece=2):
    """finite arm key: Day-1 keys for the same-layer 2-piece arm; ':t<l>' for another target, ':p3' for the third piece."""
    a = f"finite:{mode}:{'a%g' % alpha if alpha is not None else 'rep'}:{frame}"
    if target != "same": a += f":t{target}"
    if piece != 2: a += f":p{piece}"
    return a


def merge_summaries(tags, root, seed=0):
    """--summarise: merge the rows of several runs (shards) into one summary per layer."""
    merged = None
    for tag in tags:
        r = json.load(open(os.path.join(root, tag, "results.json")))
        if merged is None: merged = {k: v for k, v in r.items() if k != "layers"}; merged["layers"] = {}; merged["merged_from"] = list(tags)
        for l, s in r["layers"].items():
            m = merged["layers"].setdefault(l, {"rows": [], "hn": s.get("hn")}); seen = {(x["id"], x["arm"], x["control"]) for x in m["rows"]}
            m["rows"] += [x for x in s["rows"] if (x["id"], x["arm"], x["control"]) not in seen]
    for l, s in merged["layers"].items():
        s["summary"] = summarise(s["rows"], seed); s["verdict"] = verdict(s["summary"])
    return merged


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--items", default=os.path.join(os.path.dirname(HERE), "data", "h2a_items.json"))
    p.add_argument("--split", default="heldout", help="heldout | tuning | all (answer ids of data/h2a_split.json)")
    p.add_argument("--split-file", default=os.path.join(os.path.dirname(HERE), "data", "h2a_split.json"))
    p.add_argument("--layers", default="22"); p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=64); p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--alphas", default="0.3,1"); p.add_argument("--modes", default="add,replace", help="comma list of add, replace, replace_nm"); p.add_argument("--frame", default="plain", help="plain | answer | plain,answer")
    p.add_argument("--target-layer", default="same", help="comma list: same and/or block indices the patch is applied after (h is always read at --layers)")
    p.add_argument("--pieces", default="2", help="2 | 2,3: also score s3 given s1 s2 on items with >= 3 pieces (arm suffix :p3)")
    p.add_argument("--linear", action="store_true", help="also the exact gap-1 JVP (R2's tangent arm; same-layer only)")
    p.add_argument("--linear-modes", default="lp,z", help="which linear readouts to run: lp (log-prob tangent, primary) and/or z (pseudo-logit)")
    p.add_argument("--fargap", action="store_true", help="also the far-gap G8 reference (mask 16..tau-2)")
    p.add_argument("--controls", default="", help="comma list of shuffled,crosscat,random (empty: none)")
    p.add_argument("--kinds", default="answer"); p.add_argument("--min-p1-chars", type=int, default=0); p.add_argument("--per-category", type=int, default=0)
    p.add_argument("--max-items", type=int, default=0, help="cap the items AFTER gating"); p.add_argument("--max-scan", type=int, default=0, help="cap the items BEFORE gating (smokes: gating greedy-decodes every item)")
    p.add_argument("--shard", default="", help="i/n: run gated items i, i+n, ... (controls drawn from the full gated set)")
    p.add_argument("--summarise", default="", help="comma list of run tags: merge their rows into runs/<tag>/results.json and exit (CPU)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--track", default="", help="A (h2a sub-word) | B (phrase) | S (single); default from the items file name")
    p.add_argument("--dtype", default="fp32", help="fp32 | bf16 | bf16-mixed (bf16 weights, fp32 activations; = fp32 for Qwen3)")
    p.add_argument("--device", default="cuda"); p.add_argument("--contexts", default="runs/contexts/qwen3_T128_256.pt")
    p.add_argument("--tag", default="x1_gap1"); p.add_argument("--resume", action="store_true"); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(HERE), "runs", a.tag); os.makedirs(out_dir, exist_ok=True); out_f = os.path.join(out_dir, "results.json")
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True); log(" ".join(sys.argv))
    if a.summarise:
        merged = merge_summaries([t for t in a.summarise.split(",") if t], os.path.join(os.path.dirname(HERE), "runs"), a.seed)
        json.dump(merged, open(out_f, "w"), indent=1, ensure_ascii=False); log(f"merged {a.summarise} -> {out_f}"); return

    tok, m, q = load_q(a.model, a.dtype, a.device)
    J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))
    ctx, bg = contexts(tok, a.n_ctx, 8, a.T, a.device, path=a.contexts if os.path.exists(a.contexts) else None)
    ids = torch.stack(ctx); mask = valid_mask(a.T, 16).to(a.device)
    gap1 = torch.zeros(a.T, dtype=torch.bool); gap1[a.T - 1] = True  # the one-position mask: gap 1 to the s2 position
    items = e12.split_items(e12.subset(e12.load_items(tok, a.items), a), a.split_file, a.split)
    if a.max_scan: items = items[: a.max_scan]
    log(f"{len(items)} items ({a.split} split) after subset filters")
    kept = []
    for it in items:
        pid = tok(it["prompt"], return_tensors="pt").input_ids.to(a.device)
        it["correct"], it["gated"], it["model_greedy"] = e12.gate(q, tok, it, pid); it["prompt_ids"] = pid
        if it["correct"] and it["gated"] and len(it["pieces"]) >= 2: kept.append(it)
    if a.max_items: kept = kept[: a.max_items]
    log(f"{len(kept)} gated items with >= 2 pieces")
    alphas = [float(x) for x in a.alphas.split(",") if x]; modes = [x for x in a.modes.split(",") if x]
    frames = [x for x in a.frame.split(",") if x]; ctrls = ["none"] + [c for c in a.controls.split(",") if c]
    targets = parse_targets(a.target_layer); pieces = [int(x) for x in a.pieces.split(",") if x]
    shard = [int(x) for x in a.shard.split("/")] if a.shard else None
    run_idx = [n for n in range(len(kept)) if shard is None or n % shard[1] == shard[0]]
    results = {"model": a.model, "dtype": a.dtype, "split": a.split, "n_ctx": len(ctx), "T": a.T, "seed": a.seed, "alphas": alphas, "modes": modes,
               "frames": frames, "controls": ctrls, "target_layers": targets, "pieces": pieces, "shard": a.shard, "n_items": len(items), "n_kept": len(kept),
               "n_run": len(run_idx), "track": e12.track_of(a.items, a.track), "prereg": "docs/prereg_R.yaml (R1, R2; amendments A1-A6)", "layers": {}}
    done = set()
    if a.resume and os.path.exists(out_f):
        old = json.load(open(out_f)); results["layers"] = old.get("layers", {})
        done = {(int(l), r["id"], r["arm"], r["control"]) for l, s in results["layers"].items() for r in s["rows"]}; log(f"resuming: {len(done)} rows")
    # controls: another gated item's h from the same category (fixed derangement, never the same string; -1 = no valid
    # partner, dropped), from a different category (cross-category derangement), and a norm-matched random direction
    strings = [it["string"] for it in kept]
    perm = e12.shuffled_perm([it["category"] for it in kept], a.seed, strings); xperm = e12.cross_perm([it["category"] for it in kept], a.seed, strings)
    results["n_shuffled_dropped"] = int((perm < 0).sum()); results["n_crosscat_dropped"] = int((xperm < 0).sum())
    log(f"controls: shuffled partners missing for {results['n_shuffled_dropped']} items, crosscat for {results['n_crosscat_dropped']}")

    for l in [int(x) for x in a.layers.split(",")]:
        t0 = time.time(); D = token_matrix32(q.w, J[l]); tls = {t: (l if t == "same" else t) for t in targets}
        hn, xlast = {}, {}
        with torch.no_grad():
            for tl in sorted(set(tls.values()) | {l}):
                xl = q.resid(ids, tl); hn[tl] = float(xl[:, mask].float().norm(dim=-1).mean())  # mean carrier residual norm at this layer
                xlast[tl] = xl[:, a.T - 1].clone(); del xl  # clean residual at tau-1: the replace arm's baseline, one per target layer
        H = {}
        for it in kept:
            with torch.no_grad(): H[it["id"]] = q.resid(it["prompt_ids"], l)[0, -1]
        g = torch.Generator().manual_seed(a.seed + l)
        summ = results["layers"].setdefault(str(l), {"rows": [], "hn": hn[l], "hn_by_target": {str(k): v for k, v in hn.items()}}); rows = summ["rows"]
        for n in range(len(kept)):
            it = kept[n]; h = H[it["id"]]
            r = torch.randn(h.shape, generator=g, dtype=torch.float32).to(h.device, h.dtype)  # drawn for every item so shards share the seed stream
            if n not in run_idx: continue
            s1, s2 = it["pieces"][0], it["pieces"][1]
            dh = D @ h; rank_p1 = int((dh > dh[s1]).sum()) + 1
            hs = {"none": h, "random": r / r.norm() * h.norm()}
            if int(perm[n]) >= 0: hs["shuffled"] = H[kept[int(perm[n])]["id"]]
            if int(xperm[n]) >= 0: hs["crosscat"] = H[kept[int(xperm[n])]["id"]]
            base = {"id": it["id"], "category": it["category"], "kind": it["kind"], "string": it["string"], "p1_chars": it.get("p1_chars"),
                    "rank_p1_lens": rank_p1, "h_norm": float(h.norm()), "shuffled_from": kept[int(perm[n])]["id"] if int(perm[n]) >= 0 else None,
                    "crosscat_from": kept[int(xperm[n])]["id"] if int(xperm[n]) >= 0 else None}
            # the clean carrier + suffix distribution: shared by every arm and control, and the s1-only baseline (C6)
            suffixes = {}
            for f in frames:
                for pc in pieces:
                    if pc == 3 and len(it["pieces"]) < 3: continue
                    suffixes[(f, pc)] = frame_ids(tok, s1, f) + ([s2] if pc == 3 else [])
            cl = {k: clean_logp(q, ids, suf, a.chunk) for k, suf in suffixes.items()}
            cinfo = {(f, pc): {"rank_s2_clean": rank_of(cl[(f, pc)], it["pieces"][pc - 1]), "logp_s2_clean": float(cl[(f, pc)][it["pieces"][pc - 1]])} for (f, pc) in suffixes}
            for ctrl in ctrls:
                if ctrl not in hs: continue  # no valid partner for this control (reported in n_*_dropped)
                u = hs[ctrl]
                for (frame, pc), suffix in suffixes.items():
                    tgt = it["pieces"][pc - 1]
                    for target in targets:
                        tl = tls[target]
                        for mode in modes:
                            for alpha in (alphas if mode == "add" else [None]):
                                arm = arm_name(mode, alpha, frame, target, pc)
                                if (l, it["id"], arm, ctrl) in done: continue
                                dlp, pat = delta_logp(q, ids, l, u, suffix, alpha or 1.0, mode, hn[tl], a.chunk, clean=cl[(frame, pc)], xlast=xlast[tl], target_layer=tl, return_patched=True)
                                rows.append({**base, **cinfo[(frame, pc)], "arm": arm, "control": ctrl, "frame": frame, "target_layer": tl, "piece": pc,
                                             "rank_s2": rank_of(dlp, tgt), "dlp_s2": float(dlp[tgt]), "rank_s2_abs": rank_of(pat, tgt), "logp_s2_patched": float(pat[tgt]),
                                             "top5": [tok.decode([t]) for t in torch.topk(dlp, 5).indices.tolist()],
                                             "top5_abs": [tok.decode([t]) for t in torch.topk(pat, 5).indices.tolist()]})
                    if pc != 2: continue
                    if a.linear:
                        for mode_r, arm in (("lp", f"linear:lp:{frame}"), ("z", f"linear:z:{frame}")):
                            if (l, it["id"], arm, ctrl) in done or mode_r not in a.linear_modes.split(","): continue
                            v = torch.stack([jvp_step(q, ids[c0:c0 + a.chunk], l, gap1, u, prefix=suffix, mode=mode_r) for c0 in range(0, len(ctx), a.chunk)]).mean(0).double()
                            rows.append({**base, **cinfo[(frame, pc)], "arm": arm, "control": ctrl, "frame": frame, "target_layer": l, "piece": 2, "rank_s2": rank_of(v, s2), "dlp_s2": float(v[s2]),
                                         "top5": [tok.decode([t]) for t in torch.topk(v, 5).indices.tolist()]})
                    if a.fargap and frame == frames[0]:
                        arm = "fargap:lp"
                        if (l, it["id"], arm, ctrl) not in done:
                            v = torch.stack([jvp_step(q, ids[c0:c0 + a.chunk], l, mask, u, prefix=[s1], mode="lp") for c0 in range(0, len(ctx), a.chunk)]).mean(0).double()
                            rows.append({**base, **cinfo[(frame, pc)], "arm": arm, "control": ctrl, "frame": frames[0], "target_layer": l, "piece": 2, "rank_s2": rank_of(v, s2), "dlp_s2": float(v[s2]),
                                         "top5": [tok.decode([t]) for t in torch.topk(v, 5).indices.tolist()]})
            last = [r_ for r_ in rows if r_["id"] == it["id"] and r_["control"] == "none"]
            log(f"L{l} {run_idx.index(n)+1}/{len(run_idx)} {it['string']!r} ({it['category']}): p1 lens rank {rank_p1}; clean s2 rank {cinfo.get((frames[0], 2), {}).get('rank_s2_clean', float('nan')):.0f}; " +
                ", ".join(f"{r_['arm']} s2 rank {r_['rank_s2']:.0f} abs {r_.get('rank_s2_abs', float('nan')):.0f} dlp {r_['dlp_s2']:+.3f}" for r_ in last[:4]))
            summ["summary"] = summarise(rows, a.seed); summ["seconds"] = time.time() - t0
            json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
        summ["summary"] = summarise(rows, a.seed); summ["verdict"] = verdict(summ["summary"]); summ["seconds"] = time.time() - t0
        json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
        for arm, byc in sorted(summ["summary"].items()):
            for ctrl, e in sorted(byc.items()):
                ab = f", abs top-1 {e['abs_top1']:.3f} top-10 {e['abs_top10']:.3f} (clean {e['clean_top10']:.3f})" if "abs_top10" in e else ""
                pr = e.get("paired", {}); pd = f", paired real-ctrl top-10 {pr['top10_diff']:+.3f} [{pr['top10_diff_lo']:+.3f}, {pr['top10_diff_hi']:+.3f}] McNemar {pr['top10_mcnemar']['b']}/{pr['top10_mcnemar']['c']}" if pr.get("top10_diff") is not None else ""
                log(f"L{l} {arm} [{ctrl}]: n {e['n']}, s2 top-1 {e['top1']:.3f} [{e['top1_lo']:.3f}, {e['top1_hi']:.3f}], top-10 {e['top10']:.3f} "
                    f"[{e['top10_lo']:.3f}, {e['top10_hi']:.3f}], median rank {e['median_rank']:.0f}, mean dlp(s2) {e['mean_dlp_s2']:+.4f}{ab}{pd}")
        v = summ["verdict"]; log(f"L{l} verdict vs prereg R1 (pass top-10 >= {PASS_TOP10} with controls <= {CTRL_MAX}; kill <= {KILL_TOP10} and within 2 pts of control): "
                                 f"any_pass {v['any_pass']}, all_kill {v['all_kill']}, best arm {v['best_arm']}; SECONDARY line (A6: real - same-cat >= {SEC_GAP}, CI lower >= {SEC_LO}, "
                                 f"real - random >= {SEC_RANDOM}): any {v['any_secondary_pass']}; {summ['seconds']:.0f}s")


if __name__ == "__main__":
    main()
