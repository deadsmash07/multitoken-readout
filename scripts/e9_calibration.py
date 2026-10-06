"""E9 / H11: is z_bg standard normal on background activations, and does the conformal threshold hold its rate?

Per layer: background activations from disjoint context sets (statistics / calibration / test). z_bg of every
normalised lens row d(w) on the test activations is compared with N(0, 1) by a Kolmogorov-Smirnov statistic; the
max-z statistic over the vocabulary is calibrated by split conformal on the calibration set and its false-positive
rate measured on the test set for q = 0.05 and 0.01, next to the Bonferroni z_fw(V, q)."""
import argparse, json, math, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min
from sjlens.lens import jlens
from sjlens.lens.jlens import valid_mask
from sjlens.lens.score import z_bg, conformal_threshold, z_fw
from sjlens.eval.common import load_hf, background_activations
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download


def ks_normal(z):
    """Kolmogorov-Smirnov distance between the empirical CDF of z and the standard normal CDF."""
    z = z.flatten().double().cpu().sort().values; n = z.numel()
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2))); emp_hi = torch.arange(1, n + 1, dtype=torch.float64) / n; emp_lo = emp_hi - 1 / n
    return float(torch.maximum((emp_hi - cdf).abs(), (emp_lo - cdf).abs()).max())


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--layers", default="11"); p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=48)
    p.add_argument("--n-tokens", type=int, default=2000, help="lens rows used for the max-z statistic (all if 0)")
    p.add_argument("--device", default="cpu"); p.add_argument("--contexts", default="contexts/qwen3_T128.pt"); p.add_argument("--tag", default="e9_calibration"); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", a.tag); os.makedirs(out_dir, exist_ok=True)
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    tok, m = load_hf(a.model, torch.float32, a.device); q = Qwen3Min(m)
    J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))
    ctx, bg = contexts(tok, a.n_ctx, a.n_ctx, a.T, a.device, path=a.contexts if os.path.exists(a.contexts) else None)
    allc = ctx + bg; third = len(allc) // 3; mask = valid_mask(a.T, 16).to(a.device)
    results = {"model": a.model, "T": a.T, "n_ctx_per_split": third, "layers": {}}
    g = torch.Generator().manual_seed(0)
    for l in [int(x) for x in a.layers.split(",")]:
        t0 = time.time(); D = token_matrix32(q.w, J[l]); V = D.shape[0]
        rows = torch.randperm(V, generator=g)[: a.n_tokens] if a.n_tokens else torch.arange(V)
        Dn = D[rows.to(D.device)]; Dn = Dn / Dn.norm(dim=1, keepdim=True).clamp_min(1e-12)
        Hs = background_activations(q, allc[:third], l, mask); Hcal = background_activations(q, allc[third:2 * third], l, mask); Htest = background_activations(q, allc[2 * third:3 * third], l, mask)
        mu = Hs.mean(0); Hc = Hs - mu
        Z = torch.stack([z_bg(Dn, h, mu, Hc) for h in Htest])  # [n_test, n_rows]
        ks = ks_normal(Z); zstd = float(Z.std()); zmean = float(Z.mean())
        maxz_cal = torch.stack([z_bg(Dn, h, mu, Hc).max() for h in Hcal]); maxz_test = Z.max(1).values
        r = {"n_stats": int(Hs.shape[0]), "n_cal": int(Hcal.shape[0]), "n_test": int(Htest.shape[0]), "n_rows": int(Dn.shape[0]),
             "z_mean": zmean, "z_std": zstd, "ks_vs_normal": ks, "ks_95pct_critical": 1.36 / math.sqrt(Z.numel()), "conformal": {}}
        for qq in (0.05, 0.01):
            tau = conformal_threshold(maxz_cal, qq); fpr = float((maxz_test > tau).float().mean())
            zb = z_fw(int(Dn.shape[0]), qq)
            r["conformal"][str(qq)] = {"tau": tau, "fpr": fpr, "fpr_2se": 2 * math.sqrt(qq * (1 - qq) / len(maxz_test)), "z_fw_bonferroni": zb, "fpr_at_bonferroni": float((maxz_test > zb).float().mean())}
        r["maxz_test_mean"] = float(maxz_test.mean()); r["seconds"] = time.time() - t0; results["layers"][str(l)] = r
        log(f"L{l}: z_bg on {Htest.shape[0]} fresh background x {Dn.shape[0]} rows: mean {zmean:+.3f} std {zstd:.3f} KS {ks:.4f} (95% critical {r['ks_95pct_critical']:.4f}); "
            + "; ".join(f"q={qq}: tau {v['tau']:.2f} FPR {v['fpr']:.3f} (+-{v['fpr_2se']:.3f}); Bonferroni {v['z_fw_bonferroni']:.2f} would give FPR {v['fpr_at_bonferroni']:.3f}" for qq, v in r["conformal"].items()))
        json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
