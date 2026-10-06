"""J-lens fit with batched cotangents.

Identical estimator to lens/jlens.py::fit (cotangent one-hot on every valid target, gradient at each valid
source, mean over sources and prompts) but `dim_batch` output dimensions share one vmapped backward pass
(`is_grads_batched=True`), so a d=1024 model needs d/dim_batch backward launches per (prompt, layer) instead
of d. tests/test_eval_tiny.py pins it to jlens.fit at 1e-12.
"""
import torch
from ..model.qwen3_min import forward
from ..lens.jlens import valid_mask


def fit(q, prompts, layers, skip_first=16, dim_batch=64, target_layer=None):
    w = q.w; L = target_layer if target_layer is not None else w.n_layers - 1; d = q.d; dev = w.emb.device; ad = getattr(q, "act_dtype", None)
    J = {l: torch.zeros(d, d, dtype=torch.float64, device=dev) for l in layers}
    for ids in prompts:
        ids = ids.view(1, -1); T = ids.shape[1]; mask = valid_mask(T, skip_first).to(dev)
        for l in layers:
            src = torch.zeros(1, T, d, dtype=ad or w.emb.dtype, device=dev, requires_grad=True)
            hL, _ = forward(w, ids, inject=(l, None, src), stop_layer=L, act_dtype=ad)
            tgt = hL[0][mask].sum(0)  # [d]: sum over valid targets t' of hL[t', j]
            for j0 in range(0, d, dim_batch):
                js = torch.arange(j0, min(j0 + dim_batch, d), device=dev)
                eye = torch.zeros(len(js), d, dtype=hL.dtype, device=dev)
                eye[torch.arange(len(js), device=dev), js] = 1
                g, = torch.autograd.grad(tgt, src, grad_outputs=eye, retain_graph=True, is_grads_batched=True)
                J[l][js] += g[:, 0][:, mask].double().mean(1)  # [nb, T, d] -> mean over sources
    return {l: (J[l] / len(prompts)).cpu() for l in layers}
