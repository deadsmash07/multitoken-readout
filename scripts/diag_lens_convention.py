"""Which convention does the released lens use? For activations h at our layer l (residual after block l), compare
the finite-difference readout (two perturbed passes, targets summed, W_U applied; raw hL and RMS-normed hL variants)
against D h for candidate Jacobians: lens layers l-1, l, l+1, each as stored and transposed. One D at a time (1.2 GB
each on a 2048-wide model). Reports Spearman over the vocabulary, over the candidate's top-100 and top-1000, and the
relative error."""
import argparse, gc, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import Qwen3Min, pseudo_logits, rms
from sjlens.lens import jlens
from sjlens.lens.forward import ContextBank
from sjlens.eval.common import load_hf, spearman, background_activations
from sjlens.eval.phase_a import contexts, token_matrix32
from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--lens-file", default="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
p.add_argument("--layer", type=int, default=11); p.add_argument("--T", type=int, default=64); p.add_argument("--n-ctx", type=int, default=4)
p.add_argument("--n-act", type=int, default=2); p.add_argument("--rho", type=float, default=0.1); p.add_argument("--contexts", default="contexts/qwen3_T128.pt"); a = p.parse_args()

tok, m = load_hf(a.model, torch.float32, "cpu"); q = Qwen3Min(m); l = a.layer
J, meta = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file))
ctx, bg = contexts(tok, a.n_ctx, 4, a.T, "cpu", path=a.contexts if os.path.exists(a.contexts) else None); bank = ContextBank(q, ctx, 16)
mask = bank.mask; H = background_activations(q, bg, l, mask)
with torch.no_grad(): hn = q.resid(bank.ids, l)[:, mask].norm(dim=-1).mean().item()
print(f"layer {l}: source residual norm {hn:.2f}; {bank.C} contexts x {bank.nT} sources; lens layers {sorted(J)[0]}..{sorted(J)[-1]}", flush=True)
m_ = mask.view(1, -1).expand(bank.C, -1)
fds = []
for i in range(a.n_act):
    h = H[i]; eps = a.rho * hn / h.norm().item()
    with torch.no_grad():
        hp, _ = q.prefill(bank.ids, inject=(l, m_, eps * h)); hm, _ = q.prefill(bank.ids, inject=(l, m_, -eps * h))
        fd_raw = pseudo_logits(q.w, (hp - hm)[:, mask].sum(1).mean(0) / (2 * eps * bank.nT))
        fd_nrm = (pseudo_logits(q.w, rms(hp, q.w.norm, q.w.eps)[:, mask]) - pseudo_logits(q.w, rms(hm, q.w.norm, q.w.eps)[:, mask])).sum(1).mean(0) / (2 * eps * bank.nT)
    fds.append((h, fd_raw, fd_nrm)); del hp, hm
    print(f"activation {i}: |h| = {h.norm():.2f}; FD top-5 raw: {[tok.decode([t]) for t in torch.topk(fd_raw, 5).indices.tolist()]}", flush=True)
rows = []
for ll in (l - 1, l, l + 1):
    if ll not in J: continue
    for tr in (False, True):
        D = token_matrix32(q.w, J[ll].T if tr else J[ll]); name = f"J[{ll}]" + (".T" if tr else "")
        for i, (h, fd_raw, fd_nrm) in enumerate(fds):
            ex = D @ h; top = torch.topk(ex, 100).indices; top1k = torch.topk(ex, 1000).indices
            for tname, fd in (("raw", fd_raw), ("normed", fd_nrm)):
                rows.append((i, name, tname, spearman(fd, ex), spearman(fd[top], ex[top]), spearman(fd[top1k], ex[top1k]), float((fd - ex).norm() / ex.norm()), [tok.decode([t]) for t in top[:5].tolist()]))
        del D; gc.collect()
print(f"\n{'act':>3s} {'candidate':10s} {'target':6s} {'sp_all':>7s} {'sp_top100':>9s} {'sp_top1k':>8s} {'rel_err':>7s}  top-5 by candidate")
for i, name, tname, s_all, s100, s1k, err, top5 in sorted(rows, key=lambda r: (r[0], r[2], -r[3])):
    print(f"{i:3d} {name:10s} {tname:6s} {s_all:7.3f} {s100:9.3f} {s1k:8.3f} {err:7.3f}  {top5}")
