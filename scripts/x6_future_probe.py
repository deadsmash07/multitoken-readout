"""X6 / P2: vocabulary-space probes (replaces the retired X4 class probes, A10 / ANALYSIS_WAVE2 W1-W2).

Both probes output a FULL-VOCABULARY logit vector through the model's own head, so an unseen label is scorable.

(i) Future-Lens-style probe (Pal et al. 2023, arXiv 2311.04897). For a layer l and an offset k, a linear map
    A_k, b_k : h_t^{(l)} -> y = the FINAL-layer residual (pre-norm) at position t + k - 1, whose logits
    lm_head(norm(y)) predict token t + k. k = 2 is the "two-ahead" probe (from h at the item's last prompt token it
    predicts s2 WITHOUT s1 being given), k = 3 the three-ahead one (s3), k = 1 the tuned-lens-like reference
    (predicts s1, beside the plain logit lens). Trained by ridge regression on generic wikitext records
    (--contexts: the ctx + bg splits of runs/contexts/qwen3_T128_256.pt and the disjoint qwen3_T128_B.pt = 704 records
    x 126 pairs = ~89k pairs per layer; add a bigger file to reach more), streaming: only S_xx = sum x x^T and
    S_xy = sum x y^T are accumulated (fp64), one forward per record batch, then A = (S_xx + lambda I)^-1 S_xy with a
    bias column (unpenalised). lambda is chosen per (l, k) on a held-out 10 % of the RECORDS (never on items) by the
    next-token NLL of the probe's logits over the grid --l2-grid x mean diag(S_xx). Evaluation on the gated items
    (held-out half for tracks A / B; all gated answers of track S): rank of s2 over the vocabulary from h at the last
    prompt token (k = 2), of s1 (k = 1 probe and the plain logit lens), of s3 (k = 3), and the chained pair (s1 top-1
    by the logit lens AND s2 top-1 by the k = 2 probe; also both in the top-10). Controls through the same probe:
    shuffled-h (same category, fixed derangement A1), crosscat (A2), norm-matched random direction. Baselines on the
    SAME items: the rank of s2 in the ORIGINAL released J-lens (bench cosine readout and the raw D h) and in the logit
    lens - the single-vector readers the probe must beat.
    A11 lines: pass if held-out s2 top-10 >= 0.30 with both shuffled controls <= 0.10 at some layer (Holm over
    layers in the write-up); kill if s2 top-10 <= the logit lens's s2 top-10 + 0.05 at every layer.
(ii) Entity-grouped bridge probe. Items: gated bridge items (e12 gate, no leak) with a first-hop position
    (e12.firsthop_pos). Features: h at the first-hop entity's last token, and separately h at the last prompt token.
    Target: the bridge string's FIRST token, read through lm_head(norm(A h + b)). A is a full d x d map initialised at
    the wikitext k = 1 probe of that layer (a general "next token" reader) and fine-tuned by cross-entropy through the
    frozen head with an L2 penalty ||A - A_init||^2 * lambda_b (Adam, full batch, --bridge-steps), lambda_b chosen by
    an inner GROUPED 4-fold CV on the training folds over --bridge-l2-grid. Folds: K = 5 entity-grouped folds; the
    group key joins the BRIDGE STRING and the FIRST-HOP TOKEN TEXT by union-find, so no country / person / element
    (and no capital / work title) appears in both a training and a test fold (the W1 leak). Control: training labels
    permuted within the training folds (same lambda_b). Baselines at the same positions: the logit lens and the J-lens
    rank of the bridge's first token (no training). Reported by category and for PERSON vs COUNTRY bridges.
    A11 lines: pass if person-bridge top-10 >= 0.30 (entity-grouped, first-hop position) at some layer; kill if
    <= 0.05 on both models.
Layers: 1.7B 14 / 18 / 22 / 26; 14B 24 / 32 / 36. Tracks A, B (answers + bridges) and S (single-token: s1 ranks and the
bridge probe only). runs/<tag>/results.json (+ probes_L*.pt with the fitted maps for reuse)."""
import argparse, importlib.util, json, math, os, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import forward
from sjlens.lens import jlens
from sjlens.eval.common import load_q
from sjlens.wsbench_regex import jlens_cosine_scores
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
_load = lambda n, f: (lambda s: (lambda mod: (s.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s)))(importlib.util.spec_from_file_location(n, os.path.join(HERE, f)))
e12 = _load("e12", "e12_second_token.py"); x1 = _load("x1", "x1_gap1_patch.py")
PASS_TOP10, CTRL_MAX, KILL_MARGIN, BR_PASS, BR_KILL = 0.30, 0.10, 0.05, 0.30, 0.05  # A11 (P2)


# ------------------------------------------------------------------ (i) streaming ridge probes
class RidgeAccum:
    """S_xx [d+1, d+1] and S_xy [d+1, d] in fp64 for one (layer, offset) pair; the last column of x is the bias 1."""
    def __init__(s, d, device):
        s.d = d; s.Sxx = torch.zeros(d + 1, d + 1, dtype=torch.float64, device=device); s.Sxy = torch.zeros(d + 1, d, dtype=torch.float64, device=device); s.n = 0

    def add(s, X, Y):
        X1 = torch.cat([X.double(), torch.ones(X.shape[0], 1, dtype=torch.float64, device=X.device)], 1)
        s.Sxx += X1.T @ X1; s.Sxy += X1.T @ Y.double(); s.n += X.shape[0]

    def solve(s, lam):
        """A [d+1, d] = (S_xx + lam * diag(1..1, 0))^-1 S_xy; returns (A [d, d], b [d]) in fp32."""
        R = torch.eye(s.d + 1, dtype=torch.float64, device=s.Sxx.device); R[s.d, s.d] = 0.0
        M = torch.linalg.solve(s.Sxx + lam * R, s.Sxy)
        return M[: s.d].float(), M[s.d].float()


def residual_layers(q, ids, layers):
    """one forward: {l: residual after block l [B, T, d]} for every l in layers plus the FINAL residual under key -1."""
    out = {}; x = None; prev = None
    for l in sorted(layers):
        x, _ = forward(q.w, ids, None, stop_layer=l, act_dtype=q.act_dtype) if prev is None else forward(q.w, ids, None, stop_layer=l, act_dtype=q.act_dtype, resume=(prev, x))
        out[l] = x.float(); prev = l
    hL, _ = forward(q.w, ids, None, act_dtype=q.act_dtype) if prev is None else forward(q.w, ids, None, act_dtype=q.act_dtype, resume=(prev, x))
    out[-1] = hL.float(); return out


def accumulate(q, records, layers, offsets, skip=1, batch=16, log=print):
    """streaming S_xx / S_xy over records [N, T] for every (layer, offset): x = h_t^{(l)}, y = h_L at t + k - 1,
    t in [skip, T - k]."""
    acc = {(l, k): RidgeAccum(q.d, q.w.emb.device) for l in layers for k in offsets}; t0 = time.time()
    for b0 in range(0, len(records), batch):
        ids = torch.stack(records[b0:b0 + batch]).to(q.w.emb.device); T = ids.shape[1]
        with torch.no_grad(): R = residual_layers(q, ids, layers)
        for l in layers:
            for k in offsets:
                X = R[l][:, skip:T - k].reshape(-1, q.d)  # h at t, t in [skip, T - k)
                Y = R[-1][:, skip + k - 1:T - 1].reshape(-1, q.d)  # the final residual at t + k - 1 (its logits predict token t + k)
                assert X.shape[0] == Y.shape[0]; acc[(l, k)].add(X, Y)
        if (b0 // batch) % 10 == 0: log(f"  accumulated {min(b0 + batch, len(records))}/{len(records)} records ({time.time() - t0:.0f}s)")
    return acc


def probe_logits(q, A, b, H):
    """lm_head(norm(A h + b)) for H [n, d] -> [n, V] float32."""
    from sjlens.model.qwen3_min import logits as true_logits
    with torch.no_grad(): return true_logits(q.w, (H.float() @ A + b).to(q.w.emb.dtype if q.act_dtype is None else q.act_dtype)).float()


def choose_lambda(q, acc, records_val, l, k, grid, skip=1, batch=16, log=print):
    """lambda (absolute) minimising the held-out next-token NLL of the probe over grid x mean diag(S_xx)."""
    scale = float(acc.Sxx.diagonal()[: acc.d].mean()); best = None
    with torch.no_grad():
        feats = []
        for b0 in range(0, len(records_val), batch):
            ids = torch.stack(records_val[b0:b0 + batch]).to(q.w.emb.device); T = ids.shape[1]; R = residual_layers(q, ids, [l])
            X = R[l][:, skip:T - k].reshape(-1, q.d); y = ids[:, skip + k:T].reshape(-1); feats.append((X, y))
        for g in grid:
            lam = g * scale; A, b = acc.solve(lam); nll = 0.0; top1 = 0; n = 0
            for X, y in feats:
                lp = F.log_softmax(probe_logits(q, A, b, X), -1); nll += float(-lp[torch.arange(len(y)), y].sum()); top1 += int((lp.argmax(-1) == y).sum()); n += len(y)
            log(f"    L{l} k{k} lambda {g:g} x {scale:.1f}: val NLL {nll / n:.3f} top-1 {top1 / n:.3f} (n {n})")
            if best is None or nll / n < best[1]: best = (g, nll / n, top1 / n)
    return best[0] * scale, {"grid_value": best[0], "scale": scale, "val_nll": best[1], "val_top1": best[2], "n_val_pairs": n}


# ------------------------------------------------------------------ (ii) entity-grouped bridge probe
class UnionFind:
    def __init__(s): s.p = {}
    def find(s, x):
        s.p.setdefault(x, x)
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def union(s, a, b): s.p[s.find(a)] = s.find(b)


def entity_groups(keys_per_item):
    """keys_per_item: list of key tuples per item (e.g. (bridge string, first-hop token text)); items sharing ANY key
    join one group (union-find). Returns group ids per item (0..G-1, in order of first appearance)."""
    uf = UnionFind()
    for i, ks in enumerate(keys_per_item):
        for k in ks: uf.union(("item", i), ("key", k))
    roots = {}; out = []
    for i in range(len(keys_per_item)):
        r = uf.find(("item", i)); out.append(roots.setdefault(r, len(roots)))
    return out


def grouped_folds(groups, K, seed):
    """K folds over items such that every group sits in ONE fold; groups are shuffled (seed) and dealt to the currently
    smallest fold (balances sizes). Returns fold index per item."""
    remap = {gi: j for j, gi in enumerate(sorted(set(groups)))}; groups = [remap[gi] for gi in groups]  # ids need not be contiguous (inner CV subsets)
    G = len(remap); g = torch.Generator().manual_seed(seed); order = torch.randperm(G, generator=g).tolist()
    members = {}
    for i, gi in enumerate(groups): members.setdefault(gi, []).append(i)
    fold_of_group = {}; sizes = [0] * K
    for gi in order:
        f = min(range(K), key=lambda x: sizes[x]); fold_of_group[gi] = f; sizes[f] += len(members[gi])
    return [fold_of_group[gi] for gi in groups]


def finetune_probe(q, A0, b0, X, y, lam, steps=200, lr=1e-3):
    """A, b from (A0, b0) by cross-entropy through the frozen head with lam * ||A - A0||^2 (Adam, full batch)."""
    A = A0.clone().requires_grad_(True); b = b0.clone().requires_grad_(True); opt = torch.optim.Adam([A, b], lr=lr)
    from sjlens.model.qwen3_min import logits as true_logits
    Xf = X.float()
    for _ in range(steps):
        opt.zero_grad(); z = true_logits(q.w, (Xf @ A + b).to(q.w.emb.dtype if q.act_dtype is None else q.act_dtype)).float()
        loss = F.cross_entropy(z, y) + lam * ((A - A0) ** 2).sum(); loss.backward(); opt.step()
    return A.detach(), b.detach()


def bridge_cv(q, A0, b0, X, y, groups, K, grid, steps, seed, log=print, name=""):
    """entity-grouped K-fold CV: every item is scored by a probe that never saw its group; lambda_b by an inner
    grouped 4-fold CV on the training folds (top-1); shuffled-label control inside each fold. Returns ranks [n] of the
    true token over the vocabulary (probe and control), the folds and the chosen lambdas."""
    folds = grouped_folds(groups, K, seed); n = len(y); ranks = torch.zeros(n); ranks_c = torch.zeros(n); lams = []
    for f in range(K):
        te = [i for i in range(n) if folds[i] == f]; tr = [i for i in range(n) if folds[i] != f]
        if not te or not tr: continue
        assert not ({groups[i] for i in te} & {groups[i] for i in tr}), "entity group in both train and test"
        # inner grouped CV for lambda_b
        inner = grouped_folds([groups[i] for i in tr], min(4, len({groups[i] for i in tr})), seed + 1); best = None
        for g in grid:
            hit = 0; tot = 0
            for fi in range(max(inner) + 1):
                itr = [tr[j] for j in range(len(tr)) if inner[j] != fi]; ite = [tr[j] for j in range(len(tr)) if inner[j] == fi]
                if not itr or not ite: continue
                A, b = finetune_probe(q, A0, b0, X[itr], y[itr], g, steps); z = probe_logits(q, A, b, X[ite]); hit += int((z.argmax(-1) == y[ite]).sum()); tot += len(ite)
            acc = hit / max(tot, 1)
            if best is None or acc > best[1]: best = (g, acc)
        lam = best[0]; lams.append(lam)
        A, b = finetune_probe(q, A0, b0, X[tr], y[tr], lam, steps); z = probe_logits(q, A, b, X[te]); ranks[te] = midrank(z, y[te])
        gsh = torch.Generator().manual_seed(seed + 7 + f); ysh = y[tr][torch.randperm(len(tr), generator=gsh)]
        Ac, bc = finetune_probe(q, A0, b0, X[tr], ysh, lam, steps); zc = probe_logits(q, Ac, bc, X[te]); ranks_c[te] = midrank(zc, y[te])
        log(f"    {name} fold {f}: train {len(tr)} test {len(te)} lambda_b {lam:g} (inner top-1 {best[1]:.2f}); test top-1 {float((ranks[te] <= 1).float().mean()):.3f} top-10 {float((ranks[te] <= 10).float().mean()):.3f} | shuffled-label {float((ranks_c[te] <= 10).float().mean()):.3f}")
    return ranks, ranks_c, folds, lams


def midrank(S, y):
    tv = S[torch.arange(len(y)), y][:, None]; return ((S > tv).sum(1) + 1 + ((S == tv).sum(1) - 1) / 2).float().cpu()


def rank_block(ranks, seed=0, prefix=""):
    r = [float(x) for x in ranks]; e = {prefix + "n": len(r)}
    if not r: return e
    e[prefix + "median_rank"] = float(torch.tensor(r).median())
    e[prefix + "top1"], e[prefix + "top1_lo"], e[prefix + "top1_hi"] = x1.boot(r, lambda t: (t <= 1).double().mean(), seed=seed)
    e[prefix + "top10"], e[prefix + "top10_lo"], e[prefix + "top10_hi"] = x1.boot(r, lambda t: (t <= 10).double().mean(), seed=seed)
    return e


def by_category(rows, key, cat="category"):
    per = {}
    for r in rows:
        if r.get(key) is not None: per.setdefault(r[cat], []).append(r[key])
    return {c: {"n": len(v), "top1": sum(x <= 1 for x in v) / len(v), "top10": sum(x <= 10 for x in v) / len(v)} for c, v in sorted(per.items())}


def person_country(cat):
    if "person" in cat: return "person"
    if "capital" in cat or "currency" in cat or "country" in cat or "state" in cat or "landmark" in cat or "company" in cat or "team" in cat: return "country"
    return "other"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--items", default=os.path.join(os.path.dirname(HERE), "data", "h2a_items.json")); p.add_argument("--split", default="heldout")
    p.add_argument("--split-file", default=os.path.join(os.path.dirname(HERE), "data", "h2a_split.json"))
    p.add_argument("--layers", default="14,18,22,26"); p.add_argument("--offsets", default="1,2,3"); p.add_argument("--contexts", default="runs/contexts/qwen3_T128_256.pt,runs/contexts/qwen3_T128_B.pt")
    p.add_argument("--max-records", type=int, default=0, help="cap the training records (smokes)"); p.add_argument("--val-frac", type=float, default=0.1); p.add_argument("--batch", type=int, default=16); p.add_argument("--skip", type=int, default=1)
    p.add_argument("--l2-grid", default="0.001,0.01,0.1,1,10"); p.add_argument("--kinds", default="answer,bridge"); p.add_argument("--controls", default="shuffled,crosscat,random")
    p.add_argument("--bridge-folds", type=int, default=5); p.add_argument("--bridge-l2-grid", default="0.01,0.1,1,10"); p.add_argument("--bridge-steps", type=int, default=200); p.add_argument("--no-bridge", action="store_true")
    p.add_argument("--min-p1-chars", type=int, default=0); p.add_argument("--per-category", type=int, default=0); p.add_argument("--max-scan", type=int, default=0); p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--track", default=""); p.add_argument("--dtype", default="fp32"); p.add_argument("--device", default="cuda"); p.add_argument("--tag", default="x6_probe"); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(HERE), "runs", a.tag); os.makedirs(out_dir, exist_ok=True); out_f = os.path.join(out_dir, "results.json")
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True); log(" ".join(sys.argv))
    tok, m, q = load_q(a.model, a.dtype, a.device); dev = q.w.emb.device
    J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))
    layers = [int(x) for x in a.layers.split(",")]; offsets = [int(x) for x in a.offsets.split(",")]; grid = [float(x) for x in a.l2_grid.split(",")]
    # training records: every ctx + bg record of every contexts file; the last val-frac of the (fixed) order is validation
    records = []
    for f in [x for x in a.contexts.split(",") if x]:
        o = torch.load(f, weights_only=False); records += [c.view(-1) for c in o["ctx"]] + [c.view(-1) for c in o["bg"]]
    if a.max_records: records = records[: a.max_records]
    nv = max(1, int(len(records) * a.val_frac)); rec_tr, rec_val = records[:-nv], records[-nv:]
    log(f"{len(records)} records ({len(rec_tr)} train / {len(rec_val)} val), T {records[0].numel()}; ~{len(rec_tr) * (records[0].numel() - 2)} pairs per layer")
    t0 = time.time(); acc = accumulate(q, rec_tr, layers, offsets, a.skip, a.batch, log); log(f"accumulated in {time.time() - t0:.0f}s")
    probes = {}; fit_info = {}
    for l in layers:
        for k in offsets:
            lam, info = choose_lambda(q, acc[(l, k)], rec_val, l, k, grid, a.skip, a.batch, log); A, b = acc[(l, k)].solve(lam)
            probes[(l, k)] = (A, b); fit_info[f"L{l}_k{k}"] = {"lambda": lam, **info, "n_pairs": acc[(l, k)].n}
            log(f"L{l} k{k}: lambda {lam:.3g}, n {acc[(l, k)].n}, val top-1 {info['val_top1']:.3f} NLL {info['val_nll']:.3f}")
    torch.save({f"L{l}_k{k}": (A.cpu(), b.cpu()) for (l, k), (A, b) in probes.items()}, os.path.join(out_dir, "probes.pt"))
    del acc
    # items
    items = e12.split_items(e12.subset(e12.load_items(tok, a.items), a), a.split_file, a.split)
    if a.max_scan: items = items[: a.max_scan]
    kept = []
    for it in items:
        pid = tok(it["prompt"], return_tensors="pt").input_ids.to(dev)
        it["correct"], it["gated"], it["model_greedy"] = e12.gate(q, tok, it, pid); it["prompt_ids"] = pid; it["firsthop_pos"] = e12.firsthop_pos(tok, it, pid[0].tolist())
        if it["correct"] and it["gated"]: kept.append(it)
    if a.max_items: kept = kept[: a.max_items]
    ans = [it for it in kept if it["kind"] == "answer"]; br = [it for it in kept if it["kind"] == "bridge" and it["firsthop_pos"] is not None]
    log(f"{len(items)} items ({a.split}); gated: {len(ans)} answers, {sum(it['kind'] == 'bridge' for it in kept)} bridges ({len(br)} with a first-hop position)")
    H = {l: {} for l in layers}; Hf = {l: {} for l in layers}
    for it in kept:
        with torch.no_grad(): R = residual_layers(q, it["prompt_ids"], layers)
        for l in layers:
            H[l][it["id"]] = R[l][0, -1]
            if it["kind"] == "bridge" and it["firsthop_pos"] is not None: Hf[l][it["id"]] = R[l][0, it["firsthop_pos"]]
    ctrls = [c for c in a.controls.split(",") if c]
    strings = [it["string"] for it in ans]; cats = [it["category"] for it in ans]
    perm = e12.shuffled_perm(cats, a.seed, strings); xperm = e12.cross_perm(cats, a.seed, strings)
    results = {"model": a.model, "dtype": a.dtype, "split": a.split, "track": e12.track_of(a.items, a.track), "contexts": a.contexts, "n_records": len(records), "n_val_records": nv, "offsets": offsets, "l2_grid": grid,
               "fit": fit_info, "n_items": len(items), "n_answers": len(ans), "n_bridges": len(br), "controls": ctrls, "n_shuffled_dropped": int((perm < 0).sum()), "n_crosscat_dropped": int((xperm < 0).sum()),
               "prereg": "docs/prereg_R.yaml amendment A11 (P2)", "layers": {}}
    W_U = q.w.lm.float()
    for l in layers:
        t0 = time.time(); D = W_U @ J[l].float().to(dev); blk = {"rows": [], "bridge": {}}
        g = torch.Generator().manual_seed(a.seed + l)
        for n, it in enumerate(ans):
            h = H[l][it["id"]]; r = torch.randn(h.shape, generator=g, dtype=torch.float32).to(dev)
            hs = {"none": h, "random": r / r.norm() * h.norm()}
            if int(perm[n]) >= 0 and "shuffled" in ctrls: hs["shuffled"] = H[l][ans[int(perm[n])]["id"]]
            if int(xperm[n]) >= 0 and "crosscat" in ctrls: hs["crosscat"] = H[l][ans[int(xperm[n])]["id"]]
            if "random" not in ctrls: hs.pop("random")
            pieces = it["pieces"]
            for ctrl, u in hs.items():
                row = {"id": it["id"], "category": it["category"], "string": it["string"], "pieces_text": it["pieces_text"], "control": ctrl, "n_pieces": len(pieces)}
                with torch.no_grad():
                    z_lg = probe_logits(q, torch.eye(q.d, device=dev), torch.zeros(q.d, device=dev), u.view(1, -1))[0].double()  # the plain logit lens
                    z_j = jlens_cosine_scores(D, u).double(); z_jr = (D @ u).double()
                    zp = {k: probe_logits(q, *probes[(l, k)], u.view(1, -1))[0].double() for k in offsets if (l, k) in probes}
                row["rank_s1_logitlens"] = x1.rank_of(z_lg, pieces[0]); row["rank_s1_jlens"] = x1.rank_of(z_j, pieces[0]); row["rank_s1_jlens_raw"] = x1.rank_of(z_jr, pieces[0])
                if 1 in zp: row["rank_s1_probe1"] = x1.rank_of(zp[1], pieces[0])
                if len(pieces) >= 2:
                    row["rank_s2_logitlens"] = x1.rank_of(z_lg, pieces[1]); row["rank_s2_jlens"] = x1.rank_of(z_j, pieces[1]); row["rank_s2_jlens_raw"] = x1.rank_of(z_jr, pieces[1])
                    if 2 in zp:
                        row["rank_s2_probe2"] = x1.rank_of(zp[2], pieces[1]); row["top5_probe2"] = [tok.decode([t]) for t in torch.topk(zp[2], 5).indices.tolist()]
                        row["chain_top1"] = int(row["rank_s1_logitlens"] <= 1 and row["rank_s2_probe2"] <= 1); row["chain_top10"] = int(row["rank_s1_logitlens"] <= 10 and row["rank_s2_probe2"] <= 10)
                if len(pieces) >= 3 and 3 in zp: row["rank_s3_probe3"] = x1.rank_of(zp[3], pieces[2]); row["rank_s3_logitlens"] = x1.rank_of(z_lg, pieces[2])
                blk["rows"].append(row)
            if n % 50 == 0: log(f"L{l} {n + 1}/{len(ans)} {it['string']!r}: s1 logit-lens rank {blk['rows'][-len(hs)]['rank_s1_logitlens']:.0f}; s2 probe2 rank {blk['rows'][-len(hs)].get('rank_s2_probe2')}")
        # summaries per control
        blk["summary"] = {}
        for ctrl in ["none"] + ctrls:
            rs = [r for r in blk["rows"] if r["control"] == ctrl]
            if not rs: continue
            e = {"n": len(rs)}
            for key in ("rank_s1_logitlens", "rank_s1_jlens", "rank_s1_jlens_raw", "rank_s1_probe1", "rank_s2_logitlens", "rank_s2_jlens", "rank_s2_jlens_raw", "rank_s2_probe2", "rank_s3_probe3", "rank_s3_logitlens"):
                v = [r[key] for r in rs if r.get(key) is not None]
                if v: e[key] = rank_block(v, a.seed); e[key]["by_category"] = by_category(rs, key)
            for key in ("chain_top1", "chain_top10"):
                v = [float(r[key]) for r in rs if r.get(key) is not None]
                if v: e[key], e[key + "_lo"], e[key + "_hi"] = x1.boot(v, lambda t: t.mean(), seed=a.seed)
            blk["summary"][ctrl] = e
        real = {r["id"]: r for r in blk["rows"] if r["control"] == "none"}
        for ctrl in ctrls:
            rs = [r for r in blk["rows"] if r["control"] == ctrl and r["id"] in real and r.get("rank_s2_probe2") is not None]
            if rs:
                a_ = [float(real[r["id"]]["rank_s2_probe2"] <= 10) for r in rs]; b_ = [float(r["rank_s2_probe2"] <= 10) for r in rs]
                d, lo, hi = x1.paired_boot(a_, b_, seed=a.seed); blk["summary"][ctrl]["paired_s2_probe2_top10"] = {"n_pairs": len(rs), "diff": d, "lo": lo, "hi": hi, "mcnemar": x1.mcnemar(a_, b_)}
        s = blk["summary"].get("none", {})
        if "rank_s2_probe2" in s:
            ctrl_max = max([blk["summary"][c]["rank_s2_probe2"]["top10"] for c in ("shuffled", "crosscat") if c in blk["summary"] and "rank_s2_probe2" in blk["summary"][c]] or [0.0])
            lg = s["rank_s2_logitlens"]["top10"]; jl = s["rank_s2_jlens"]["top10"]
            blk["verdict"] = {"s2_probe2_top10": s["rank_s2_probe2"]["top10"], "control_max": ctrl_max, "s2_logitlens_top10": lg, "s2_jlens_top10": jl,
                              "pass": bool(s["rank_s2_probe2"]["top10"] >= PASS_TOP10 and ctrl_max <= CTRL_MAX), "kill_this_layer": bool(s["rank_s2_probe2"]["top10"] <= lg + KILL_MARGIN),
                              "rule": "A11 P2(i): pass s2 top-10 >= 0.30 with shuffled and crosscat <= 0.10 (Holm over layers in the write-up); kill s2 top-10 <= logit lens + 0.05 at every layer"}
            log(f"L{l} answers: s2 top-10 probe {s['rank_s2_probe2']['top10']:.3f} [{s['rank_s2_probe2']['top10_lo']:.3f}, {s['rank_s2_probe2']['top10_hi']:.3f}] top-1 {s['rank_s2_probe2']['top1']:.3f} | logit lens {lg:.3f} | J-lens (cos) {jl:.3f} | controls " +
                ", ".join(f"{c} {blk['summary'][c]['rank_s2_probe2']['top10']:.3f}" for c in ctrls if c in blk["summary"] and "rank_s2_probe2" in blk["summary"][c]) +
                f" | s1 top-1 logit lens {s['rank_s1_logitlens']['top1']:.3f} probe1 {s.get('rank_s1_probe1', {}).get('top1')} | chain top-1 {s.get('chain_top1')} top-10 {s.get('chain_top10')} | verdict pass {blk['verdict']['pass']} kill {blk['verdict']['kill_this_layer']}")
        elif "rank_s1_logitlens" in s:
            log(f"L{l} single-token answers: s1 top-1 logit lens {s['rank_s1_logitlens']['top1']:.3f}, J-lens (cos) {s['rank_s1_jlens']['top1']:.3f}, probe1 {s.get('rank_s1_probe1', {}).get('top1')}")
        # (ii) entity-grouped bridge probe at the first-hop and the last token
        if br and not a.no_bridge and (l, 1) in probes:
            A0, b0 = probes[(l, 1)]; y = torch.tensor([it["pieces"][0] for it in br], device=dev)
            keys = [(("bridge", it["string"].strip().lower()), ("firsthop", tok.decode([int(it["prompt_ids"][0, it["firsthop_pos"]])]).strip().lower())) for it in br]
            groups = entity_groups(keys); cats_b = [it["category"] for it in br]
            bres = {"n": len(br), "n_groups": max(groups) + 1, "folds": a.bridge_folds}
            for pos, Xs in (("firsthop", torch.stack([Hf[l][it["id"]] for it in br])), ("last", torch.stack([H[l][it["id"]] for it in br]))):
                log(f"L{l} bridge probe at {pos}: {len(br)} items in {max(groups) + 1} entity groups")
                ranks, ranks_c, folds, lams = bridge_cv(q, A0, b0, Xs, y, groups, a.bridge_folds, [float(x) for x in a.bridge_l2_grid.split(",")], a.bridge_steps, a.seed, log, f"L{l} {pos}")
                with torch.no_grad():
                    z_lg = probe_logits(q, torch.eye(q.d, device=dev), torch.zeros(q.d, device=dev), Xs); r_lg = midrank(z_lg, y)
                    z_j = torch.stack([jlens_cosine_scores(D, x_) for x_ in Xs]); r_j = midrank(z_j, y)
                rows = [{"id": it["id"], "category": it["category"], "group": groups[i], "fold": folds[i], "string": it["string"], "rank_probe": float(ranks[i]), "rank_control": float(ranks_c[i]), "rank_logitlens": float(r_lg[i]), "rank_jlens": float(r_j[i]), "kind2": person_country(it["category"])} for i, it in enumerate(br)]
                e = {"probe": rank_block(ranks.tolist(), a.seed), "shuffled_labels": rank_block(ranks_c.tolist(), a.seed), "logitlens": rank_block(r_lg.tolist(), a.seed), "jlens": rank_block(r_j.tolist(), a.seed), "lambdas": lams,
                     "by_category": {k: by_category(rows, k) for k in ("rank_probe", "rank_control", "rank_logitlens", "rank_jlens")},
                     "by_kind": {k2: {k: rank_block([r[k] for r in rows if r["kind2"] == k2], a.seed) for k in ("rank_probe", "rank_control", "rank_logitlens", "rank_jlens")} for k2 in ("person", "country", "other") if any(r["kind2"] == k2 for r in rows)}, "rows": rows}
                bres[pos] = e
                pp = e["by_kind"].get("person", {}).get("rank_probe", {})
                log(f"L{l} bridge probe [{pos}]: top-1 {e['probe']['top1']:.3f} top-10 {e['probe']['top10']:.3f} | shuffled-label {e['shuffled_labels']['top10']:.3f} | logit lens {e['logitlens']['top10']:.3f} | J-lens {e['jlens']['top10']:.3f} | person top-10 {pp.get('top10')} (n {pp.get('n')})")
            pf = bres.get("firsthop", {}).get("by_kind", {}).get("person", {}).get("rank_probe", {})
            bres["verdict"] = {"person_firsthop_top10": pf.get("top10"), "person_n": pf.get("n"), "pass": bool(pf.get("top10") is not None and pf["top10"] >= BR_PASS), "kill_this_layer": bool(pf.get("top10") is not None and pf["top10"] <= BR_KILL),
                               "rule": "A11 P2(ii): pass person-bridge top-10 >= 0.30 at the first-hop position (entity-grouped folds) at some layer; kill <= 0.05 on both models"}
            blk["bridge"] = bres
        blk["seconds"] = time.time() - t0; results["layers"][str(l)] = blk
        json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
    log("done")


if __name__ == "__main__":
    main()
