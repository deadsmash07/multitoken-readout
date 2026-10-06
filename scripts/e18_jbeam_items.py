"""E18: J-beam step normaliser search and end-to-end phrase recovery on the gated H2a answer items.

Items: data/h2a_items.json answer items (the model's upcoming multi-token answer), gated by e12's greedy check,
restricted to one half of data/h2a_split.json (--half tuning | heldout | all; the split was fixed before any result).
h = residual at the last prompt token; the prompt never reaches the J-beam.

Stage "screen" (per layer; one FD step readout per (inject, rho) and one batched m-null readout per item, all
normaliser variants evaluated on the same numbers):
  (b) second-piece accuracy given the true first piece: rank of s2 in the normalised step readout after prefix [s1]
      for every variant x inject (hc, hJ) x rho (+ the exact JVP on the first --jvp-items items, and the FD error);
  seeds: rank of s1 under the bench seed rule (D h)[w]/||d(w)|| and the z rule (D hc)[w]/sigma_1(w);
  token-lens baseline: rank of s1 and of s2 in D h (no conditioning);
  (d) split-half reliability on --n-rel background activations (eval.common.sample_activations; prefix = the
      lens top-1 token, shared null directions, contexts split in halves): top-10 overlap and top-1 agreement of the
      normalised readout, per variant; the max-z over the vocabulary on these activations gives each variant's
      conformal tau_q (score.conformal_threshold) for the stopping rule.
Stage "beam" (per layer, --beam-configs "name:key=val,key=val;..."): the full J-beam on every item with shared
null caches and a per-item readout memo; (a) exact-string recall@1 / @10 among the emitted phrases (decoded
phrase == item string; also token-level and prefix matches), (c) junk rate of the emitted extension tokens:
junk_script = the token has no alphanumeric character, or contains a character outside the item's script
(item Latin: any char with ord > 0x24F; item non-Latin: any CJK/other char absent from the item string);
junk_lens = the token is in no lens top-1000 of --n-lens-vocab background activations. With stop=none and
coherence=0 (expand 1) every seed runs to max_len and the stopping rules (z_fw; tau_0.05; tau_0.2; none;
each with / without the coherence test) are evaluated offline from the logged step z's, exactly.
Saves runs/<tag>/results.json after every item; --resume skips finished (layer, id, stage) rows.
Compare with the dictionary reader (runs/e14*_L22) and the token lens on the same item ids."""
import argparse, importlib.util, json, math, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min, pseudo_logits
from sjlens.lens import jlens
from sjlens.lens.jlens import valid_mask
from sjlens.lens.forward import ContextBank, jvp_step
from sjlens.lens.score import hutchinson_sigma, shrink_sigma, nnomp, conformal_threshold, z_fw
from sjlens.eval.common import load_hf, background_activations, sample_activations
from sjlens.eval.phase_a import contexts, token_matrix32
from sjlens.beam.jbeam import JBeam, Cfg
from huggingface_hub import hf_hub_download

VERSION = "e18 v1.1 (agent D, 25 Sep 2026)"
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("e12", os.path.join(HERE, "e12_second_token.py")); e12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e12)


def variants(m_max):
    """name -> (kind, prior, m, floor, nu, support): the normaliser variants screened on one set of null readouts."""
    v = {"raw": ("raw", None, m_max, 1.0, None, 0.0)}
    for m in (8, 16, 32):
        if m <= m_max: v[f"mc{m}"] = ("mc", None, m, 1.0, None, 0.0)
    for c in (0.25, 0.5, 1.0): v[f"floor16_c{c:g}"] = ("floor", None, min(16, m_max), c, None, 0.0)
    for p in ("wu", "dnorm", "lens", "pooled"):
        v[f"proxy_{p}"] = ("proxy", p, min(16, m_max), 1.0, None, 0.0)
        v[f"eb16_{p}"] = ("eb", p, min(16, m_max), 1.0, None, 0.0)
    for m in (8, 32):
        if m <= m_max: v[f"eb{m}_pooled"] = ("eb", "pooled", m, 1.0, None, 0.0); v[f"eb{m}_lens"] = ("eb", "lens", m, 1.0, None, 0.0)
    v["eb16_pooled_nu16"] = ("eb", "pooled", min(16, m_max), 1.0, 16.0, 0.0); v["eb16_pooled_nu64"] = ("eb", "pooled", min(16, m_max), 1.0, 64.0, 0.0)
    v["mc16_supp0.5"] = ("mc", "pooled", min(16, m_max), 1.0, None, 0.5); v["eb16_pooled_supp0.5"] = ("eb", "pooled", min(16, m_max), 1.0, None, 0.5)
    v["mc16_supp0.8"] = ("mc", "pooled", min(16, m_max), 1.0, None, 0.8)
    return v


def sigma_of(spec_, nn_, priors):
    kind, p, m, floor, nu, supp = spec_
    if kind == "raw": return torch.ones_like(nn_[0]), None
    sig = shrink_sigma(nn_[:m], priors.get(p), kind, floor, nu)
    ok = (priors[p] >= priors[p].quantile(supp)) if supp > 0 else None
    return sig, ok


def z_of(spec_, num, nn_, priors):
    sig, ok = sigma_of(spec_, nn_, priors); z = num / sig
    return z.masked_fill(~ok, -float("inf")) if ok is not None else z


def rank_of(z, w):
    return int((z > z[w]).sum()) + 1 if torch.isfinite(z[w]) else int(z.numel())


def overlap(a, b, k):
    return len(set(torch.topk(a, k).indices.tolist()) & set(torch.topk(b, k).indices.tolist())) / k


def parse_cfgs(s, m_max):
    out = {}
    for part in filter(None, s.split(";")):
        name, _, kvs = part.partition(":"); kw = {}
        for kv in filter(None, kvs.split(",")):
            k, _, v = kv.partition("=")
            f = Cfg.__dataclass_fields__[k].type
            kw[k] = (v == "1" or v.lower() == "true") if f is bool else (None if v == "None" else f(v)) if f in (int, float, str) else v
        kw.setdefault("m_null", min(16, m_max)); out[name.strip()] = kw
    return out


def is_latin(s):
    return all(ord(c) <= 0x24F for c in s)


def junk_script(t, item_string):
    if not any(c.isalnum() for c in t): return True
    if is_latin(item_string): return any(ord(c) > 0x24F for c in t)
    return any(ord(c) > 0x24F and c not in item_string for c in t)


def offline_stop(chain, thresh, use_coh):
    """chain: log entries of one seed in order (expand 1). Returns (length, A, Q) of the phrase under this rule."""
    n, A, Q = 1, chain[0]["A"], chain[0]["Q"]
    for e in chain:
        if e["zstep"] > thresh and (e["coh"] or not use_coh): n += 1; A, Q = e["A"] + e["a"], e["Q"] + e["qinc"]
        else: break
    return n, A, Q


def beam_metrics(rows, cfgname, dec, rule=None):
    """recall / junk over item rows for one config; dec: token list -> text; rule = (thresh, use_coh) evaluates an offline stopping rule."""
    n = len(rows); r1 = r10 = p10 = s2 = 0; ext = []; junk_s = junk_l = 0; lens_ = []
    for r in rows:
        b = r["beam"][cfgname]; ph = []
        if rule is None:
            ph = [(p["toks"], p["z"]) for p in b["phrases"]]
        else:
            for seed, chain in b["chains"].items():
                ln, A, Q = offline_stop(chain, *rule); toks = [int(seed)] + [e["w"] for e in chain[: ln - 1]]
                ph.append((toks, A / max(math.sqrt(max(Q, 1e-12)), 1e-12)))
            ph.sort(key=lambda x: -x[1])
        ph = ph[:10]; pieces = r["pieces"]; s = r["string"]
        hit = [p for p, _ in ph if dec(p) == s]; r1 += bool(ph and dec(ph[0][0]) == s); r10 += bool(hit)
        p10 += any(p[: len(pieces)] == pieces for p, _ in ph)
        s2 += any(len(p) >= 2 and p[0] == pieces[0] and p[1] == pieces[1] for p, _ in ph)
        for p, _ in ph:
            lens_.append(len(p))
            for w in p[1:]:
                ext.append(w); junk_s += r["junk_script"].get(str(w), False); junk_l += r["junk_lens"].get(str(w), False)
    f = lambda x: x / n if n else None
    return {"n": n, "recall1": f(r1), "recall10": f(r10), "prefix10": f(p10), "s2_in_beam": f(s2), "n_ext": len(ext),
            "junk_script": junk_s / len(ext) if ext else None, "junk_lens": junk_l / len(ext) if ext else None, "mean_len": sum(lens_) / len(lens_) if lens_ else None}


def summarise_screen(rows, keys):
    out = {}
    for k in keys:
        rk = [r["rank_p2"][k] for r in rows if k in r["rank_p2"]]
        if rk: t = torch.tensor(rk).float(); out[k] = {"n": len(rk), "top1": float((t == 1).float().mean()), "top10": float((t <= 10).float().mean()), "median": float(t.median())}
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--items", default=os.path.join(os.path.dirname(HERE), "data", "h2a_items.json")); p.add_argument("--split", default=os.path.join(os.path.dirname(HERE), "data", "h2a_split.json"))
    p.add_argument("--half", default="tuning", help="tuning | heldout | all"); p.add_argument("--layers", default="18,22"); p.add_argument("--T", type=int, default=128)
    p.add_argument("--n-ctx", type=int, default=64); p.add_argument("--n-bg", type=int, default=256); p.add_argument("--m-null", type=int, default=32); p.add_argument("--pool-prefixes", type=int, default=64)
    p.add_argument("--dtype", default="bf16", help="bf16: bf16 weights + fp32 activations (G2: exact for Qwen3); fp32")
    p.add_argument("--stage", default="screen,beam"); p.add_argument("--injects", default="hc,hJ"); p.add_argument("--rhos", default="0.02,0.05,0.1"); p.add_argument("--jvp-items", type=int, default=40)
    p.add_argument("--n-rel", type=int, default=64); p.add_argument("--n-lens-vocab", type=int, default=512); p.add_argument("--max-items", type=int, default=0); p.add_argument("--per-category", type=int, default=0)
    p.add_argument("--min-p1-chars", type=int, default=0); p.add_argument("--kinds", default="answer"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beam-configs", default="mc16_hJ:normalizer=mc,m_null=16,inject=hJ;eb16_pooled_hJ:normalizer=eb,prior=pooled,m_null=16,inject=hJ")
    p.add_argument("--beam-items", type=int, default=0); p.add_argument("--seeds", type=int, default=10); p.add_argument("--beams", type=int, default=10); p.add_argument("--max-len", type=int, default=5)
    p.add_argument("--expand", type=int, default=1); p.add_argument("--stop", default="none"); p.add_argument("--coherence", type=int, default=0); p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--device", default="cuda"); p.add_argument("--contexts", default="runs/contexts/qwen3_T128_256.pt"); p.add_argument("--tag", default="e18_jbeam"); p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(HERE), "runs", a.tag); os.makedirs(out_dir, exist_ok=True); out_f = os.path.join(out_dir, "results.json")
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True); log(VERSION); log(" ".join(sys.argv))
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    tok, m = load_hf(a.model, dt, a.device); q = Qwen3Min(m, act_dtype=torch.float32 if dt == torch.bfloat16 else None)
    J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))
    ctx, bg = contexts(tok, a.n_ctx, a.n_bg, a.T, a.device, path=a.contexts if os.path.exists(a.contexts) else None)
    ids = torch.stack(ctx); mask = valid_mask(a.T, 16).to(a.device); C = len(ctx); half = C // 2
    items = e12.subset(e12.load_items(tok, a.items), a); split = json.load(open(a.split))
    keep = set(split["tuning"]) if a.half == "tuning" else set(split["heldout"]) if a.half == "heldout" else None
    if keep is not None: items = [it for it in items if it["id"] in keep]
    kept = []
    for it in items:
        pid = tok(it["prompt"], return_tensors="pt").input_ids.to(a.device); ok, gated, txt = e12.gate(q, tok, it, pid)
        if ok and gated: it["prompt_ids"] = pid; kept.append(it)
    log(f"{len(items)} {a.half} answer items, {len(kept)} greedy-correct (gated)")
    if a.max_items: kept = kept[: a.max_items]
    stages = a.stage.split(","); injects = a.injects.split(","); rhos = [float(x) for x in a.rhos.split(",")]; inj0 = injects[-1]; nk0 = f"{inj0}_r{rhos[0]:g}"
    VAR = variants(a.m_null); cfgs = parse_cfgs(a.beam_configs, a.m_null)
    results = {"version": VERSION, "args": vars(a), "half": a.half, "n_items": len(items), "n_kept": len(kept), "ids": [it["id"] for it in kept], "variants": {k: list(v) for k, v in VAR.items()}, "beam_configs": cfgs, "layers": {}}
    if a.resume and os.path.exists(out_f):
        old = json.load(open(out_f)); results["layers"] = old.get("layers", {}); log(f"resuming from {out_f}")
    for l in [int(x) for x in a.layers.split(",")]:
        t0 = time.time(); D = token_matrix32(q.w, J[l]); Hall = background_activations(q, bg, l, mask); mu = Hall.mean(0); Hc = Hall - mu
        g = torch.Generator().manual_seed(l); nulls = Hc[torch.randperm(Hc.shape[0], generator=g)[: a.m_null].to(Hc.device)]
        bank = ContextBank(q, ctx, 16)
        jb0 = JBeam(q, bank, D, l, nulls, Cfg(m_null=a.m_null, normalizer="mc", rho=a.rho), mu, Hbg=Hall)
        jbp = JBeam(q, bank, D, l, nulls, Cfg(m_null=a.m_null, normalizer="eb", prior="pooled", pool_prefixes=a.pool_prefixes, rho=a.rho), mu, Hbg=Hall, null_set=jb0.null_set)
        priors = {"wu": q.w.lm.float().norm(dim=1).clamp_min(1e-12), "dnorm": jb0.Dn, "lens": jb0.sig1_all, "pooled": jbp.prior}
        Lr = results["layers"].setdefault(str(l), {}); Lr["n_ctx"] = C; Lr["m_null"] = a.m_null; Lr["hn"] = jb0.hn; Lr["pool_tokens"] = [tok.decode([t]) for t in jbp.pool_tokens]
        log(f"L{l}: null caches + priors in {time.time() - t0:.0f}s; mean source norm {jb0.hn:.1f}; pooled prefixes {len(jbp.pool_tokens)}; "
            f"cos(log prior) pooled~lens {float(torch.corrcoef(torch.stack([priors['pooled'].log(), priors['lens'].log()]))[0, 1]):.3f} pooled~wu {float(torch.corrcoef(torch.stack([priors['pooled'].log(), priors['wu'].log()]))[0, 1]):.3f}")
        # lens-reachable vocabulary for junk_lens: union of the lens top-1000 over background activations
        Hv = sample_activations(Hall, a.n_lens_vocab, seed=l + 100); reach = set()
        for i in range(0, Hv.shape[0], 64): reach |= set(torch.topk(D @ Hv[i:i + 64].T, 1000, dim=0).indices.flatten().tolist())
        Lr["n_lens_reachable"] = len(reach)
        rows = Lr.setdefault("screen", []); done = {r["id"] for r in rows}
        u_of = lambda h: (lambda hc: {"hc": hc, "h": h, "hJ": nnomp(D, hc, 25)[1]})(h - mu)

        if "screen" in stages:
            # (d) reliability + conformal tau on background activations, prefix = the lens top-1 (bench seed rule)
            if "reliability" not in Lr:
                t1 = time.time(); Hr = sample_activations(Hall, a.n_rel, seed=l + 7); rel = {k: {inj: {"ov10": [], "top1": [], "maxz": []} for inj in injects} for k in VAR}
                for i in range(Hr.shape[0]):
                    h = Hr[i]; w1 = int(torch.argmax((D @ h) / jb0.Dn)); nn_pc = bank.step_set([w1], jb0.null_set, per_context=True)  # [m, C, V]
                    nnF, nnA, nnB = nn_pc.mean(1), nn_pc[:, :half].mean(1), nn_pc[:, half:].mean(1); del nn_pc
                    us = u_of(h)
                    for inj in injects:
                        u = us[inj]; uset = bank.cache_set(l, u.view(1, -1), [jb0._eps(u)]); pc = bank.step_set([w1], uset, per_context=True)[0]; del uset
                        nF, nA, nB = pc.mean(0), pc[:half].mean(0), pc[half:].mean(0)
                        for k, sp in VAR.items():
                            zF, zA, zB = z_of(sp, nF, nnF, priors), z_of(sp, nA, nnA, priors), z_of(sp, nB, nnB, priors)
                            rel[k][inj]["ov10"].append(overlap(zA, zB, 10)); rel[k][inj]["top1"].append(float(int(zA.argmax()) == int(zB.argmax()))); rel[k][inj]["maxz"].append(float(zF.max()))
                Lr["reliability"] = {k: {inj: {"ov10": sum(v["ov10"]) / len(v["ov10"]), "top1": sum(v["top1"]) / len(v["top1"]), "maxz_mean": sum(v["maxz"]) / len(v["maxz"]),
                                                "maxz_sd": float(torch.tensor(v["maxz"]).std()), "tau05": conformal_threshold(torch.tensor(v["maxz"]), 0.05), "tau20": conformal_threshold(torch.tensor(v["maxz"]), 0.2),
                                                "maxz": v["maxz"]} for inj, v in d_.items()} for k, d_ in rel.items()}
                Lr["reliability"]["_n"] = int(Hr.shape[0]); Lr["reliability"]["_seconds"] = time.time() - t1; Lr["zfw"] = z_fw(D.shape[0], 0.05)
                json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
                log(f"L{l} reliability ({Hr.shape[0]} acts, {half}+{C - half} ctx halves, {time.time() - t1:.0f}s; ov10 / tau05, {inj0}): " + " | ".join(f"{k} {Lr['reliability'][k][inj0]['ov10']:.2f}/{Lr['reliability'][k][inj0]['tau05']:.1f}" for k in ("raw", "mc16", "mc32", "eb16_pooled", "eb16_lens", "proxy_pooled", "proxy_lens") if k in VAR))
            # (b) teacher-forced second piece per item
            for n, it in enumerate(kept):
                if it["id"] in done: continue
                t1 = time.time(); p1, p2 = it["pieces"][0], it["pieces"][1]
                with torch.no_grad(): h = q.resid(it["prompt_ids"], l)[0, -1].float()
                us = u_of(h); Dh = D @ h; Dhc = D @ (h - mu)
                row = {"id": it["id"], "category": it["category"], "string": it["string"], "pieces": it["pieces"], "pieces_text": it["pieces_text"],
                       "rank_p1_lens": rank_of(Dh, p1), "rank_p1_bench": rank_of(Dh / jb0.Dn, p1), "rank_p1_z": rank_of(Dhc / jb0.sig1_all, p1), "rank_p2_lens": rank_of(Dh, p2), "rank_p2": {}, "top5": {}, "fd_err": {}}
                nn_ = jb0.null_readout([p1])
                for inj in injects:
                    u = us[inj]
                    nums = {}
                    for rho in rhos:
                        eps = rho * jb0.hn / float(u.norm()); uset = bank.cache_set(l, u.view(1, -1), [eps]); nums[f"{inj}_r{rho:g}"] = bank.step_set([p1], uset)[0]; del uset
                    if n < a.jvp_items: nums[f"{inj}_jvp"] = torch.stack([jvp_step(q, ids[c0:c0 + 16], l, mask, u, [p1]) for c0 in range(0, C, 16)]).mean(0)
                    for nk, num in nums.items():
                        if nk.endswith("_jvp"):
                            for rho in rhos: row["fd_err"][f"{inj}_r{rho:g}"] = float((nums[f"{inj}_r{rho:g}"] - num).norm() / num.norm())
                        for k, sp in VAR.items():
                            z = z_of(sp, num, nn_, priors); row["rank_p2"][f"{k}|{nk}"] = rank_of(z, p2)
                            if k in ("raw", "mc16", "eb16_pooled", "eb16_lens", "proxy_pooled") and nk == nk0: row["top5"][f"{k}|{nk}"] = [tok.decode([t]) for t in torch.topk(z, 5).indices.tolist()]
                rows.append(row); row["seconds"] = time.time() - t1
                rp = lambda k: row["rank_p2"].get(f"{k}|{nk0}", "-")
                log(f"L{l} screen {n + 1}/{len(kept)} {it['string']!r}: p1 lens {row['rank_p1_lens']} bench {row['rank_p1_bench']} z {row['rank_p1_z']}; p2 rank raw {rp('raw')} mc16 {rp('mc16')} "
                    f"eb16_pooled {rp('eb16_pooled')} eb16_lens {rp('eb16_lens')} ({nk0}); top5 eb16_pooled {row['top5'].get('eb16_pooled|' + nk0)}; {row['seconds']:.1f}s")
                if (n + 1) % 10 == 0: json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
            keys = sorted({k for r in rows for k in r["rank_p2"]}); Lr["screen_summary"] = summarise_screen(rows, keys)
            t = lambda k: torch.tensor([r[k] for r in rows]).float()
            Lr["seed_summary"] = {k: {"top1": float((t(k) == 1).float().mean()), "top10": float((t(k) <= 10).float().mean()), "median": float(t(k).median())} for k in ("rank_p1_lens", "rank_p1_bench", "rank_p1_z", "rank_p2_lens")}
            fe = {k: [r["fd_err"][k] for r in rows if k in r["fd_err"]] for k in sorted({k for r in rows for k in r["fd_err"]})}; Lr["fd_err"] = {k: sum(v) / len(v) for k, v in fe.items() if v}
            json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
            ss = Lr["screen_summary"]
            log(f"L{l} screen summary ({len(rows)} items): seeds p1 top-10 lens {Lr['seed_summary']['rank_p1_lens']['top10']:.2f} bench {Lr['seed_summary']['rank_p1_bench']['top10']:.2f} z {Lr['seed_summary']['rank_p1_z']['top10']:.2f}; "
                f"p2 in D h (token lens) top-1 {Lr['seed_summary']['rank_p2_lens']['top1']:.2f}; FD err {Lr['fd_err']}")
            for nk in [f"{inj}_r{rho:g}" for inj in injects for rho in rhos] + [f"{inj}_jvp" for inj in injects]:
                log(f"L{l}  p2 top-1/top-10 [{nk}]: " + " ".join(f"{k} {ss[f'{k}|{nk}']['top1']:.2f}/{ss[f'{k}|{nk}']['top10']:.2f}" for k in VAR if f"{k}|{nk}" in ss))

        if "beam" in stages:
            brows = Lr.setdefault("beam_rows", []); bdone = {r["id"] for r in brows}; bitems = kept[: a.beam_items] if a.beam_items else kept
            jbs = {}
            for name, kw in cfgs.items():
                cfg = Cfg(**{**dict(seeds=a.seeds, beams=a.beams, max_len=a.max_len, expand=a.expand, stop=a.stop, coherence=bool(a.coherence), rho=a.rho), **kw})
                jbs[name] = JBeam(q, bank, D, l, nulls, cfg, mu, Hbg=Hall, null_set=jb0.null_set, prior=priors.get(cfg.prior) if (cfg.normalizer in ("proxy", "eb") or cfg.support > 0) else None)
                if cfg.stop == "tau" and cfg.tau is None:  # conformal tau of the matching screen variant at q = 0.05
                    key = {"mc": f"mc{cfg.m_null}", "eb": f"eb{cfg.m_null}_{cfg.prior}", "proxy": f"proxy_{cfg.prior}", "raw": "raw", "floor": f"floor16_c{cfg.floor:g}"}[cfg.normalizer]
                    cfg.tau = Lr["reliability"][key][cfg.inject]["tau05"]; log(f"L{l} {name}: tau05 = {cfg.tau:.2f} from screen variant {key}")
            for n, it in enumerate(bitems):
                if it["id"] in bdone: continue
                t1 = time.time(); memo = {}
                with torch.no_grad(): h = q.resid(it["prompt_ids"], l)[0, -1].float()
                row = {"id": it["id"], "category": it["category"], "string": it["string"], "pieces": it["pieces"], "beam": {}, "junk_script": {}, "junk_lens": {}}
                for name, jb in jbs.items():
                    lg = []; out, _ = jb.read(h, lg, memo)
                    chains = {}
                    if a.expand == 1:
                        for e in lg: chains.setdefault(str(e["prefix"][0]), []).append(e)
                        for c in chains.values(): c.sort(key=lambda e: len(e["prefix"]))
                    row["beam"][name] = {"phrases": [{"toks": t, "text": tok.decode(t), "z": z, "steps": [(w, round(zz, 3)) for w, zz in st]} for t, z, zs, st in out],
                                         "chains": {s_: [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k != "top"} for e in c] for s_, c in chains.items()}}
                    for t, *_ in out:
                        for w in t[1:]:
                            row["junk_script"][str(w)] = junk_script(tok.decode([w]), it["string"]); row["junk_lens"][str(w)] = w not in reach
                    for c in chains.values():
                        for e in c: row["junk_script"][str(e["w"])] = junk_script(tok.decode([e["w"]]), it["string"]); row["junk_lens"][str(e["w"])] = e["w"] not in reach
                del memo; torch.cuda.empty_cache() if a.device == "cuda" else None
                brows.append(row); row["seconds"] = time.time() - t1
                log(f"L{l} beam {n + 1}/{len(bitems)} {it['string']!r}: " + " | ".join(f"{name}: {[p['text'] for p in row['beam'][name]['phrases'][:3]]}" for name in jbs) + f"; {row['seconds']:.1f}s")
                if (n + 1) % 5 == 0: json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
            Lr["beam_summary"] = {}
            for name, jb in jbs.items():
                cfg = jb.cfg; summ = {"cfg": {k: getattr(cfg, k) for k in Cfg.__dataclass_fields__}, "as_run": beam_metrics(brows, name, tok.decode)}
                if a.expand == 1 and a.stop == "none" and not a.coherence and "reliability" in Lr:
                    key = {"mc": f"mc{cfg.m_null}", "eb": f"eb{cfg.m_null}_{cfg.prior}", "proxy": f"proxy_{cfg.prior}", "raw": "raw", "floor": f"floor16_c{cfg.floor:g}"}.get(cfg.normalizer)
                    rl = Lr["reliability"].get(key, {}).get(cfg.inject, {})
                    rules = {"zfw": Lr["zfw"], "tau05": rl.get("tau05"), "tau20": rl.get("tau20"), "none": -float("inf")}
                    summ["stop_rules"] = {f"{rn}{'_coh' if coh else ''}": beam_metrics(brows, name, tok.decode, (th, coh)) for rn, th in rules.items() if th is not None for coh in (True, False)}
                    summ["stop_rules"]["_thresholds"] = {k: v for k, v in rules.items() if v is not None}
                Lr["beam_summary"][name] = summ
                fm = lambda x, p=2: "-" if x is None else f"{x:.{p}f}"
                log(f"L{l} beam {name}: as run {summ['as_run']}" + (f"; stop rules: " + " | ".join(f"{rn}: r1 {fm(v['recall1'])} r10 {fm(v['recall10'])} pre10 {fm(v['prefix10'])} junk {fm(v['junk_script'])}/{fm(v['junk_lens'])} len {fm(v['mean_len'], 1)}" for rn, v in summ["stop_rules"].items() if rn != "_thresholds") if "stop_rules" in summ else ""))
            json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
        Lr["seconds"] = time.time() - t0; json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False); log(f"L{l} done in {Lr['seconds']:.0f}s")
        del jb0, jbp, bank; jbs = None; torch.cuda.empty_cache() if a.device == "cuda" else None


if __name__ == "__main__":
    main()
