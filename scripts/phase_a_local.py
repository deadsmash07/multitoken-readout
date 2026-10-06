"""Phase A (E1, E3, E4, E7, E8) on a trained Qwen3 with a released or freshly fitted J-lens, on CPU or a local GPU.

    python scripts/phase_a_local.py --model Qwen/Qwen3-1.7B \
        --lens-file qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt \
        --layers 8,14,20 --device cpu --T 64 --n-ctx 8 --n-bg 16 --n-act 20 --tag cpu_1p7b

Writes runs/<tag>/results.json after every experiment and runs/<tag>/log.txt. --fit fits the lens with
eval/fit_fast.py instead of downloading one (the only option for models without a released lens).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
    p.add_argument("--fit", action="store_true", help="fit the lens on the contexts instead of downloading one")
    p.add_argument("--layers", default="8,14,20")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16", action="store_true", help="also load a bf16 copy for the bf16 E3/E4 arms")
    p.add_argument("--T", type=int, default=128); p.add_argument("--n-ctx", type=int, default=16); p.add_argument("--n-bg", type=int, default=32)
    p.add_argument("--n-act", type=int, default=50); p.add_argument("--n-dims", type=int, default=64); p.add_argument("--dim-batch", type=int, default=16)
    p.add_argument("--n-ctx-gap", type=int, default=4); p.add_argument("--n-tokens", type=int, default=300); p.add_argument("--n-ctx-red", type=int, default=16)
    p.add_argument("--rhos-e3", default="0.05,0.1,0.2"); p.add_argument("--rhos-e4", default="0.01,0.05,0.1,0.2,0.3,1.0")
    p.add_argument("--do", default="E1,E3,E4,E7,E8"); p.add_argument("--tag", default=None); p.add_argument("--threads", type=int, default=0)
    p.add_argument("--contexts", default=None, help="contexts .pt from scripts/make_contexts.py (no internet needed at run time)")
    p.add_argument("--dtype", default="fp32", help="fp32 | bf16 | bf16-mixed (bf16 weights, fp32 activations; = fp32 for Qwen3, half the memory)")
    a = p.parse_args()
    if a.threads: torch.set_num_threads(a.threads)

    from sjlens.model.qwen3_min import Qwen3Min
    from sjlens.lens import jlens
    from sjlens.eval.common import load_hf, load_q
    from sjlens.eval import phase_a, fit_fast

    tag = a.tag or f"{a.model.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", tag); os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "log.txt"), "a")
    def log(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    def save(res):
        with open(os.path.join(out_dir, "results.json"), "w") as f: json.dump(res, f, indent=1)

    layers = [int(x) for x in a.layers.split(",")]
    log(f"model {a.model} on {a.device}; layers {layers}; T={a.T} n_ctx={a.n_ctx} n_bg={a.n_bg}")
    tok, m, q = load_q(a.model, a.dtype, a.device)
    q_bf = Qwen3Min(load_hf(a.model, dtype=torch.bfloat16, device=a.device)[1]) if a.bf16 else None
    ctx, bg = phase_a.contexts(tok, a.n_ctx, a.n_bg, a.T, a.device, path=a.contexts)
    log(f"d={q.d}, {q.w.n_layers} blocks; {len(ctx)} contexts, {len(bg)} background")

    if a.fit:
        t0 = time.time(); J = fit_fast.fit(q, ctx, layers, skip_first=16, dim_batch=a.dim_batch)
        log(f"fitted J-lens at {layers} in {time.time() - t0:.0f}s"); jlens.save(os.path.join(out_dir, "jlens.pt"), J, {"model": a.model, "n_prompts": len(ctx)})
        lens_name = f"fitted on {len(ctx)} contexts"
    else:
        from huggingface_hub import hf_hub_download
        J, meta = jlens.load(hf_hub_download(a.lens_repo, a.lens_file)); lens_name = f"{a.lens_repo}/{a.lens_file}"
        log(f"lens {lens_name}: layers {sorted(J)[0]}..{sorted(J)[-1]}, meta {meta}")
        J = {l: J[l] for l in layers}

    cfg = phase_a.default_cfg(model=a.model, dtype=a.dtype, lens=lens_name, n_act=a.n_act, n_dims=a.n_dims, dim_batch=a.dim_batch, n_ctx_gap=a.n_ctx_gap,
                              n_tokens=a.n_tokens, n_ctx_red=a.n_ctx_red, rhos_e3=[float(x) for x in a.rhos_e3.split(",")],
                              rhos_e4=[float(x) for x in a.rhos_e4.split(",")], do=a.do.split(","), device=a.device, threads=torch.get_num_threads())
    res = phase_a.run(q, q_bf, m, tok, ctx, bg, layers, J, cfg, log=log, save=save)
    save(res); log(f"done; wrote {out_dir}/results.json")


if __name__ == "__main__":
    main()
