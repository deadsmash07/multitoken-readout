"""G2 / E4b: which numerical mode for the perturbed passes?

Reference: exact JVP through the fp32 model. Variants, for the lens target (jvp_lens) and a one-token prefix
(jvp_exact): exact JVP with bf16 weights and bf16 activations; exact JVP with bf16 weights and fp32 activations
(mixed: weights cast per op); fp32 central FD at rho 0.02 and 0.05; mixed FD at 0.05; bf16 FD at 0.1. Reports the
relative error and the top-100 / all-token Spearman of the readout against the reference. Decision: the cheapest
mode within 2% relative error. Memory: fp32 + bf16 copies of the model."""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min, pseudo_logits
from sjlens.lens import jlens
from sjlens.lens.jlens import valid_mask
from sjlens.lens.forward import ContextBank, jvp_exact, jvp_lens
from sjlens.eval.common import load_hf, spearman, background_activations, sample_activations
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--layers", default="11"); p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=8); p.add_argument("--n-act", type=int, default=4)
    p.add_argument("--device", default="cpu"); p.add_argument("--contexts", default="contexts/qwen3_T128.pt"); p.add_argument("--tag", default="e4b_numerics"); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.tag); os.makedirs(out_dir, exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    tok, m32 = load_hf(a.model, torch.float32, a.device); q32 = Qwen3Min(m32)
    _, m16 = load_hf(a.model, torch.bfloat16, a.device); q16 = Qwen3Min(m16); qmix = Qwen3Min(m16, act_dtype=torch.float32)
    J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))
    ctx, bg = contexts(tok, a.n_ctx, 8, a.T, a.device, path=a.contexts if os.path.exists(a.contexts) else None)
    ids = torch.stack(ctx); mask = valid_mask(a.T, 16).to(a.device); W = q32.w.lm
    results = {"model": a.model, "T": a.T, "n_ctx": len(ctx), "layers": {}}
    def stats(x, ref):
        x, ref = pseudo_logits(q32.w, x.float()), pseudo_logits(q32.w, ref.float()); top = torch.topk(ref, 100).indices
        return {"rel_err": float((x - ref).norm() / ref.norm()), "sp_top100": spearman(x[top], ref[top]), "sp_all": spearman(x, ref)}
    def fd(q, layer, h, prefix, rho):
        bank = ContextBank(q, ctx, 16)
        with torch.no_grad(): hn = q.resid(bank.ids, layer)[:, mask].float().norm(dim=-1).mean().item()
        eps = rho * hn / h.norm().item(); u = h.to(q.act_dtype or q.w.emb.dtype)
        cp, cm = bank.perturbed(layer, u, eps)
        if prefix: return bank.direction_estimate(list(prefix), cp, cm, eps)[0]
        # lens target: sum over masked targets of the final residual, from the perturbed prefills
        with torch.no_grad():
            hp, _ = q.prefill(bank.ids, inject=(layer, mask.view(1, -1).expand(bank.C, -1), eps * u)); hm, _ = q.prefill(bank.ids, inject=(layer, mask.view(1, -1).expand(bank.C, -1), -eps * u))
        return (hp.float() - hm.float())[:, mask].sum(1).mean(0) / (2 * eps * bank.nT)
    for l in [int(x) for x in a.layers.split(",")]:
        t0 = time.time(); D = token_matrix32(q32.w, J[l]); H = sample_activations(background_activations(q32, bg, l, mask), a.n_act, seed=l); rows = []
        for i in range(a.n_act):
            h = H[i]; prefix = [int(torch.topk(D @ h, 1).indices)]; row = {"prefix": tok.decode(prefix)}
            for tname, pf in (("lens", []), ("prefix1", prefix)):
                ref = (jvp_lens(q32, ids, l, mask, h) if not pf else jvp_exact(q32, ids, l, mask, h, prefix=pf))
                var = {}
                var["jvp_bf16"] = (jvp_lens(q16, ids, l, mask, h.to(torch.bfloat16)) if not pf else jvp_exact(q16, ids, l, mask, h.to(torch.bfloat16), prefix=pf))
                var["jvp_mixed"] = (jvp_lens(qmix, ids, l, mask, h) if not pf else jvp_exact(qmix, ids, l, mask, h, prefix=pf))
                var["fd_fp32_rho0.02"] = fd(q32, l, h, pf, 0.02); var["fd_fp32_rho0.05"] = fd(q32, l, h, pf, 0.05)
                var["fd_mixed_rho0.05"] = fd(qmix, l, h, pf, 0.05); var["fd_bf16_rho0.1"] = fd(q16, l, h, pf, 0.1)
                row[tname] = {k: stats(v, ref) for k, v in var.items()}
                log(f"L{l} act {i} {tname:8s}: " + " | ".join(f"{k} err {v['rel_err']:.3f} top100 {v['sp_top100']:.2f}" for k, v in row[tname].items()))
            rows.append(row)
        summ = {t: {k: {kk: sum(r[t][k][kk] for r in rows) / len(rows) for kk in ("rel_err", "sp_top100", "sp_all")} for k in rows[0][t]} for t in ("lens", "prefix1")}
        summ["rows"] = rows; summ["seconds"] = time.time() - t0; results["layers"][str(l)] = summ
        json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
        for t in ("lens", "prefix1"): log(f"L{l} mean {t:8s}: " + " | ".join(f"{k} err {v['rel_err']:.3f} top100 {v['sp_top100']:.2f}" for k, v in summ[t].items()))


if __name__ == "__main__":
    main()
