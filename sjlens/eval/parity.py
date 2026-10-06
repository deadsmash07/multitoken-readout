"""E1: parity of qwen3_min with transformers.Qwen3ForCausalLM.

Compares logits, the residual after chosen blocks (captured with forward hooks, so the check does not depend
on how a given transformers version indexes `hidden_states`), extension of a K/V cache by the last n_ext tokens,
and an additive injection after one block. Each entry is reported as a max absolute difference and as that
difference relative to the max magnitude of the HF reference tensor. Both implementations build their RoPE
tables in float32 (HF does so by design), so parity is bounded near 1e-8 relative even in float64; the
proposal's E1 target on a trained fp32 model is that order.
"""
import torch
from ..model.qwen3_min import forward, logits as min_logits


def _out_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def hf_layer_outputs(m, ids, layers):
    """(logits, {layer: residual after that decoder block}) from the HF model."""
    caps, handles = {}, []
    for l in layers:
        def mk(l):
            def hook(_mod, _inp, out):
                caps[l] = _out_tensor(out).detach()
            return hook
        handles.append(m.model.layers[l].register_forward_hook(mk(l)))
    with torch.no_grad():
        lg = m(input_ids=ids, use_cache=False).logits.detach()
    for h in handles:
        h.remove()
    return lg, caps


def hf_logits_injected(m, ids, layer, mask, delta):
    """HF logits with `delta` added at masked positions to the output of decoder block `layer`."""
    def hook(_mod, _inp, out):
        t = _out_tensor(out)
        t2 = t + delta.view(1, 1, -1).to(t.dtype) * mask.to(t.dtype)[..., None]
        return (t2,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else t2
    h = m.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        lg = m(input_ids=ids, use_cache=False).logits.detach()
    h.remove()
    return lg


def _diff(a, ref):
    a, ref = a.float(), ref.float()
    d = (a - ref).abs().max().item()
    return {"abs": d, "rel": d / max(ref.abs().max().item(), 1e-30)}


def check_parity(m, q, ids, layers, n_ext=8, inject_layer=None, delta=None, mask=None):
    """ids [B, T]. Returns {name: {"abs", "rel"}}. With inject_layer, delta [d] and mask [B, T] also checks injection."""
    res = {}
    hf_lg, caps = hf_layer_outputs(m, ids, layers)
    with torch.no_grad():
        hL, _ = forward(q.w, ids, act_dtype=getattr(q, 'act_dtype', None))
        res["logits"] = _diff(min_logits(q.w, hL), hf_lg)
        for l in layers:
            res[f"resid_L{l}"] = _diff(q.resid(ids, l), caps[l])
        T = ids.shape[1]
        _, cache = q.prefill(ids[:, : T - n_ext])
        ext = q.extend(ids[:, T - n_ext :], cache)
        res[f"extension_{n_ext}"] = _diff(ext, hL[:, T - n_ext :])
        if inject_layer is not None:
            hf_inj = hf_logits_injected(m, ids, inject_layer, mask, delta)
            hI, _ = forward(q.w, ids, inject=(inject_layer, mask, delta), act_dtype=getattr(q, 'act_dtype', None))
            res[f"injection_L{inject_layer}"] = _diff(min_logits(q.w, hI), hf_inj)
    return res
