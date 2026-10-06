"""Phase A of the proposal on a trained model: E1 (parity), E3 (plumbing gate), E4 (FD vs exact JVP), E7 (gap
profile on sampled output dims), E8 (H1a reduction). Device-agnostic; scripts/phase_a_local.py runs it on a CPU or
a local GPU or a cloud GPU. `save(results)` is called after every experiment so a killed run
leaves partial results behind.
"""
import time
import torch


def token_matrix32(w, J):
    """D_l = W_U J_l in float32 (lens/jlens.py::token_matrix uses float64, which is 6 GB of W_U on a 14B)."""
    return w.lm.float() @ J.float().to(w.lm.device)


def default_cfg(**kw):
    cfg = dict(skip=16, n_act=50, rhos_e3=[0.05, 0.1, 0.2], rhos_e4=[0.01, 0.05, 0.1, 0.2, 0.3, 1.0],
               n_dims=64, dim_batch=16, n_ctx_gap=4, n_tokens=300, n_ctx_red=16, top=100,
               do=["E1", "E3", "E4", "E7", "E8"])
    cfg.update(kw)
    return cfg


def contexts(tok, n_ctx, n_bg, T, device, path=None):
    """n_ctx generic contexts and n_bg background contexts of T tokens. With `path`, load them from a file written by
    scripts/make_contexts.py (for compute nodes without internet) instead of streaming wikitext."""
    if path:
        o = torch.load(path); assert o["T"] == T, f"contexts file has T={o['T']}, asked for {T}"
        assert len(o["ctx"]) >= n_ctx and len(o["bg"]) >= n_bg, f"contexts file holds {len(o['ctx'])} ctx / {len(o['bg'])} bg"
        return [c.to(device) for c in o["ctx"][:n_ctx]], [c.to(device) for c in o["bg"][:n_bg]]
    from .common import wikitext_contexts
    ctx = [c.to(device) for c in wikitext_contexts(tok, n_ctx, T, seed=0)]
    bg = [c.to(device) for c in wikitext_contexts(tok, n_bg, T, seed=1)]
    return ctx, bg


def run(q, q_bf, m, tok, ctx, bg, layers, J, cfg, log=print, save=lambda r: None):
    """q: Qwen3Min on the fp32 model m; q_bf: optional bf16 copy; J: {layer: [d, d]} lens Jacobian."""
    from ..lens.jlens import valid_mask
    from ..lens.forward import ContextBank
    from .parity import check_parity
    from .fwd_bwd import side_result, fd_curve
    from .gap_profile import gap_profile_sampled, gap_stats, reduction
    from .common import background_activations, sample_activations

    dev = q.w.emb.device; T = ctx[0].numel(); mask = valid_mask(T, cfg["skip"]); nT = int(mask.sum())
    out = {"layers": layers, "n_ctx": len(ctx), "n_bg": len(bg), "T": T, "nT": nT, "cfg": cfg, "d": q.d, "n_layers": q.w.n_layers}
    do = set(cfg["do"])

    if "E1" in do:
        t0 = time.time(); n = min(4, len(ctx))
        ids = torch.stack(ctx[:n]); inj_mask = mask.view(1, -1).expand(n, -1).to(dev)
        delta = (torch.randn(q.d, generator=torch.Generator().manual_seed(0)) * 0.1).to(dev, q.w.emb.dtype)
        out["E1_parity"] = check_parity(m, q, ids, layers, n_ext=8, inject_layer=layers[0], delta=delta, mask=inj_mask)
        out["E1_parity"]["seconds"] = time.time() - t0
        log("E1 parity: " + ", ".join(f"{k} rel {v['rel']:.1e}" for k, v in out["E1_parity"].items() if k != "seconds"))
        save(out)

    bank = ContextBank(q, ctx, cfg["skip"])
    bank_bf = ContextBank(q_bf, ctx, cfg["skip"]) if q_bf is not None else None
    g = torch.Generator().manual_seed(1)
    for l in layers:
        r = out.setdefault(f"L{l}", {}); t_layer = time.time()
        D = token_matrix32(q.w, J[l])
        H = sample_activations(background_activations(q, bg, l, mask), max(cfg["n_act"], 5), seed=l)
        r["resid_norm"] = float(H.norm(dim=-1).mean()); r["n_bg_act"] = int(H.shape[0])

        if "E3" in do:
            t0 = time.time(); r["E3"] = {}
            def e3_row(res):
                sp = res["top"]
                return {"spearman_top_mean": float(sp.mean()), "spearman_top_min": float(sp.min()), "spearman_top_p10": float(sp.quantile(0.1)),
                        "spearman_top1k_mean": float(res["top1k"].mean()), "spearman_all_mean": float(res["all"].mean()),
                        "fd_rel_err_mean": float(res["err"].mean()), "n_act": int(sp.numel())}
            for rho in cfg["rhos_e3"]:
                res = side_result(q, bank, D, H[: cfg["n_act"]], l, rho=rho, top=cfg["top"]); row = e3_row(res); r["E3"][str(rho)] = row
                log(f"L{l} E3 rho={rho}: Spearman top-{cfg['top']} mean {row['spearman_top_mean']:.3f} p10 {row['spearman_top_p10']:.3f} (gate >= 0.95) | top-1k {row['spearman_top1k_mean']:.3f} | all {row['spearman_all_mean']:.3f} | rel err {row['fd_rel_err_mean']:.2e}")
            if bank_bf is not None:
                res = side_result(q_bf, bank_bf, D, H[: cfg["n_act"]], l, rho=0.1, top=cfg["top"]); row = e3_row(res); r["E3"]["bf16_rho0.1"] = row
                log(f"L{l} E3 bf16 rho=0.1: Spearman top mean {row['spearman_top_mean']:.3f} | all {row['spearman_all_mean']:.3f} | rel err {row['fd_rel_err_mean']:.2e}")
            r["E3"]["seconds"] = time.time() - t0; save(out)

        if "E4" in do:
            t0 = time.time(); h = H[0]; prefix = [int(torch.topk(D @ h.to(dev), 1).indices)]
            curve32, hn = fd_curve(q, q, ctx, ctx, l, h, prefix, cfg["rhos_e4"], cfg["skip"])
            r["E4"] = {"prefix": prefix, "prefix_text": tok.decode(prefix), "source_norm": hn, "h_norm": float(h.norm()), "fp32": {str(k): v for k, v in curve32.items()}}
            log(f"L{l} E4 fp32 (prefix {prefix} {r['E4']['prefix_text']!r}): " + ", ".join(f"rho={k:g} {v:.2e}" for k, v in curve32.items()))
            if q_bf is not None:
                curve16, _ = fd_curve(q_bf, q, ctx, ctx, l, h, prefix, cfg["rhos_e4"], cfg["skip"])
                r["E4"]["bf16"] = {str(k): v for k, v in curve16.items()}
                log(f"L{l} E4 bf16: " + ", ".join(f"rho={k:g} {v:.2e}" for k, v in curve16.items()))
            r["E4"]["seconds"] = time.time() - t0; save(out)

        if "E7" in do:
            t0 = time.time(); dims = torch.randperm(q.d, generator=g)[: cfg["n_dims"]].tolist()
            G, Je = gap_profile_sampled(q, ctx[: cfg["n_ctx_gap"]], l, dims, cfg["skip"], dim_batch=cfg["dim_batch"])
            st = gap_stats(G, nT)
            r["E7"] = {"rho_frob": st["rho_frob"], "rho_exact": st["rho_exact"], "S_future": st["S_future"], "share_gap1": st["share_gap1"], "share_gaps_1_5": st["share_gaps_1_5"], "share_gaps_ge20": st["share_gaps_ge20"],
                       "norms": {str(k): v for k, v in st["norms"].items()}, "dims": dims, "n_ctx": min(cfg["n_ctx_gap"], len(ctx)), "seconds": time.time() - t0}
            ns = st["norms"]; fs = lambda x: "-" if x is None else f"{x:.3f}"
            log(f"L{l} E7: gap-0 fraction rho_frob {st['rho_frob']:.3f} (exact {fs(st['rho_exact'])}); ||G_g||_F g=0..5: " + ", ".join(f"{ns[k]:.3g}" for k in sorted(ns)[:6]) + f"; g={max(ns)}: {ns[max(ns)]:.3g}"
                f"; future sum S {st['S_future']:.3g}: gap 1 {fs(st['share_gap1'])}, gaps 1-5 {fs(st['share_gaps_1_5'])}, gaps >= 20 {fs(st['share_gaps_ge20'])}")
            save(out)

        if "E8" in do:
            t0 = time.time(); n_top = min(5, H.shape[0])
            top = torch.cat([torch.topk(D @ H[i].to(dev), 100).indices.cpu() for i in range(n_top)])
            rand = torch.randint(0, D.shape[0], (cfg["n_tokens"],), generator=g)
            tokens = torch.unique(torch.cat([top, rand]))[: cfg["n_tokens"]].tolist()
            red = reduction(q, D, l, ctx[: cfg["n_ctx_red"]], tokens, cfg["skip"], acts=H[: cfg["n_act"]], top=cfg["top"])
            cos = red["cos"]
            r["E8"] = {"n_tokens": len(tokens), "n_ctx": min(cfg["n_ctx_red"], len(ctx)), "cos_median": float(cos.median()), "cos_p10": float(cos.quantile(0.1)), "cos_p90": float(cos.quantile(0.9)),
                       "spearman_top_mean": float(red["spearman_top"].mean()), "spearman_top_min": float(red["spearman_top"].min()),
                       "tokens": tokens, "cos": cos.tolist(), "examples": [(tok.decode([w]), round(float(c), 3)) for w, c in zip(tokens, cos)][:60], "seconds": time.time() - t0}
            log(f"L{l} E8 (H1a): median cos(d(w), v_z(w)) {r['E8']['cos_median']:.3f} [p10 {r['E8']['cos_p10']:.3f}, p90 {r['E8']['cos_p90']:.3f}] over {len(tokens)} tokens (pass >= 0.9); readout Spearman top-{cfg['top']} {r['E8']['spearman_top_mean']:.3f}")
            save(out)
        r["seconds"] = time.time() - t_layer
    return out
