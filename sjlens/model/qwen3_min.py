"""Minimal, functorch-friendly Qwen3 forward with residual injection and K/V caches.

Layer l (bench convention) = residual after decoder block l. inject=(l, mask[B,T] or None, delta[D] | [B,T,D])
adds delta to that residual at masked positions. prefill returns (hL, cache); extend(ids, cache) runs new tokens
against a cache and returns their last-block residuals. Attention is eager so torch.func.jvp works.
"""
from dataclasses import dataclass
import math, torch, torch.nn.functional as F


@dataclass
class W:
    emb: torch.Tensor
    ln1: list; ln2: list; q: list; k: list; v: list; o: list; qn: list; kn: list; g: list; u: list; d: list
    norm: torch.Tensor
    lm: torch.Tensor
    eps: float; nh: int; nkv: int; hd: int; theta: float

    @staticmethod
    def from_hf(m):
        c, L = m.config, m.model.layers
        g = lambda f: [f(l) for l in L]
        return W(m.model.embed_tokens.weight, g(lambda l: l.input_layernorm.weight), g(lambda l: l.post_attention_layernorm.weight),
                 g(lambda l: l.self_attn.q_proj.weight), g(lambda l: l.self_attn.k_proj.weight), g(lambda l: l.self_attn.v_proj.weight),
                 g(lambda l: l.self_attn.o_proj.weight), g(lambda l: l.self_attn.q_norm.weight), g(lambda l: l.self_attn.k_norm.weight),
                 g(lambda l: l.mlp.gate_proj.weight), g(lambda l: l.mlp.up_proj.weight), g(lambda l: l.mlp.down_proj.weight),
                 m.model.norm.weight, m.lm_head.weight, c.rms_norm_eps, c.num_attention_heads, c.num_key_value_heads,
                 getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
                 c.rope_parameters["rope_theta"] if getattr(c, "rope_parameters", None) else c.rope_theta)  # transformers 5.x (rope_parameters) and 4.x (rope_theta)

    @property
    def n_layers(s): return len(s.q)


def rms(x, w, eps):
    dt = x.dtype; lo = dt in (torch.bfloat16, torch.float16)
    x = x.float() if lo else x; wf = w.float() if lo else w
    return (wf * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))).to(dt)


def rope(hd, pos, theta, dtype, device):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = pos.float()[:, None] * inv[None]
    e = torch.cat([f, f], -1)
    return e.cos().to(dtype), e.sin().to(dtype)


def rot(x, cos, sin):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def attn(w, i, x, cos, sin, past):
    B, T, _ = x.shape
    dt = x.dtype
    q = rms(F.linear(x, w.q[i].to(dt)).view(B, T, w.nh, w.hd), w.qn[i], w.eps).transpose(1, 2)
    k = rms(F.linear(x, w.k[i].to(dt)).view(B, T, w.nkv, w.hd), w.kn[i], w.eps).transpose(1, 2)
    v = F.linear(x, w.v[i].to(dt)).view(B, T, w.nkv, w.hd).transpose(1, 2)
    q, k = rot(q, cos, sin), rot(k, cos, sin)
    if past is not None:
        k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
    S = k.shape[2]; r = w.nh // w.nkv
    kk, vv = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
    a = (q @ kk.transpose(-1, -2)) * (w.hd ** -0.5)
    m = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)
    a = a.masked_fill(~m, float("-inf"))
    a = a.float().softmax(-1).to(x.dtype) if x.dtype in (torch.bfloat16, torch.float16) else a.softmax(-1)
    y = (a @ vv).transpose(1, 2).reshape(B, T, w.nh * w.hd)
    return F.linear(y, w.o[i].to(x.dtype)), (k, v)


def block(w, i, x, cos, sin, past):
    a, kv = attn(w, i, rms(x, w.ln1[i], w.eps), cos, sin, past)
    x = x + a
    h = rms(x, w.ln2[i], w.eps)
    dt = h.dtype
    return x + F.linear(F.silu(F.linear(h, w.g[i].to(dt))) * F.linear(h, w.u[i].to(dt)), w.d[i].to(dt)), kv


def _inject(x, inject, B, T):
    l, mask, delta = inject
    d = delta if delta.dim() == 3 else delta.view(1, 1, -1).expand(B, T, -1)
    if mask is not None: d = d * mask.to(device=d.device, dtype=d.dtype)[..., None]
    return x + d


def _overwrite(x, inject_set, B, T):
    """inject_set = (layers, mask [B, T] bool, value [d] | [B, T, d]): the residual at the masked positions is SET to
    `value` (not added to). Used after every layer in `layers` (and at the embedding when -1 is in `layers`), so the
    patched position carries `value` at every one of those depths and no representation of the carrier's own token
    survives there (the placeholder-contamination arm of prereg A21)."""
    _, mask, value = inject_set
    v = value if value.dim() == 3 else value.view(1, 1, -1).expand(B, T, -1)
    return torch.where(mask.to(device=x.device)[..., None], v.to(device=x.device, dtype=x.dtype), x)


def forward(w, ids, inject=None, cache=None, stop_layer=None, act_dtype=None, resume=None, return_cache=True, inject_set=None):
    """returns (hL or h_{stop_layer}, new_cache). cache: list of (k, v) per layer or None.
    act_dtype: run the residual stream in this dtype (weights are cast per op; e.g. bf16 weights, fp32 activations).
    resume=(l, x): x [B, T, d] is the residual after block l before any injection there; blocks 0..l are skipped
    (their cache entries come back as None, and `cache` may hold None there too). return_cache=False drops the
    new K/V (an extension read only at its last position needs none).
    inject_set=(layers, mask, value): OVERWRITE the masked positions with `value` after every block in `layers`
    (and at the embedding output when -1 is in `layers`); independent of `inject` (A21 multi-layer patch)."""
    B, T = ids.shape
    off = 0 if cache is None else next(c for c in cache if c is not None)[0].shape[2]
    pos = torch.arange(off, off + T, device=ids.device)
    set_layers = set(int(l) for l in inject_set[0]) if inject_set is not None else set()
    if resume is None:
        x = F.embedding(ids, w.emb); i0 = 0
        if act_dtype is not None: x = x.to(act_dtype)
        if -1 in set_layers: x = _overwrite(x, inject_set, B, T)
    else:
        x = resume[1] if act_dtype is None else resume[1].to(act_dtype); i0 = resume[0] + 1
        if inject is not None and inject[0] == resume[0]: x = _inject(x, inject, B, T)
        if resume[0] in set_layers: x = _overwrite(x, inject_set, B, T)
        if stop_layer is not None and stop_layer == resume[0]: return x, [None] * i0
    cos, sin = rope(w.hd, pos, w.theta, x.dtype, x.device)
    new = [None] * i0
    for i in range(i0, w.n_layers):
        x, kv = block(w, i, x, cos, sin, None if cache is None else cache[i])
        new.append(kv if return_cache else None)
        if inject is not None and inject[0] == i: x = _inject(x, inject, B, T)
        if i in set_layers: x = _overwrite(x, inject_set, B, T)
        if stop_layer is not None and i == stop_layer:
            return x, new
    return x, new


def logits(w, hL):
    return F.linear(rms(hL, w.norm, w.eps), w.lm.to(hL.dtype))


def pseudo_logits(w, hL):
    return F.linear(hL, w.lm.to(hL.dtype))


class Qwen3Min:
    def __init__(s, hf_model, act_dtype=None):
        s.w = W.from_hf(hf_model)
        s.d = s.w.emb.shape[1]
        s.act_dtype = act_dtype  # None: the weights' dtype

    def prefill(s, ids, inject=None):
        return forward(s.w, ids, inject, act_dtype=s.act_dtype)

    def extend(s, ids, cache):
        return forward(s.w, ids, None, cache, act_dtype=s.act_dtype)[0]

    def resid(s, ids, layer):
        return forward(s.w, ids, None, None, stop_layer=layer, act_dtype=s.act_dtype)[0]
