"""Two interchangeable model backends for the prompt-blind readers (P2 / P3 / P6): residual read, residual patch,
last-position log-probs and patched greedy decoding.

MinBackend      wraps sjlens.model.qwen3_min.Qwen3Min - the wave-2 code path (Qwen3 dense only; fp32 / bf16-mixed;
                functorch-safe). Its last_logp does exactly the ops of scripts/x1_gap1_patch.py::delta_logp and its
                greedy exactly those of scripts/x3_patchscope.py::patched_generate, so P3 / P6 on Qwen3-1.7B / 14B
                reuse the X1 / X3 machinery bit-for-bit (tests/test_p2p3_tiny.py pins both).
HFHooksBackend  wraps ANY Hugging Face causal LM whose decoder blocks are a module list (Qwen3, Qwen3.6-27B =
                Qwen3_5ForConditionalGeneration with hybrid linear / full attention, ...): the residual after block l
                is read and patched with forward hooks on the block (the bench's own convention, produce/backend.py:
                "a benchmark layer L is the output of decoder block model.layers[L]"), carriers are batched along the
                batch axis, and generation runs through the model's own KV / recurrent-state cache with the patch
                applied during the prefill only (the patched state is then in the cache, as in patched_generate).
                Activations run in the model's dtype (bf16 on the 27B; no fp32-activation "mixed" mode exists through
                HF hooks) and the head is applied in fp32: logits = lm_head(final_norm(h_L)) with the model's own modules.
                Never loaded with the 27B here; tests/test_p2p3_tiny.py checks it equals MinBackend on a tiny Qwen3
                for resid / final / last_logp with a patch / greedy with a patch. See docs/P6_BACKEND_NOTE.md.

Interface (both):
  n_layers, d, V, device, tok
  resid(ids [B,T], layer)                       -> [B, T, d] residual after block `layer`
  final(ids, inject=None)                       -> [B, T, d] final residual (pre-norm, after the last block)
  last_logp(ids, inject=None)                   -> [B, V] float64 log-softmax at the last position (fp32 head)
  logits(h [..., d])                            -> [..., V] lm_head(norm(h)) (fp32)
  unembed()                                     -> [V, d] float32 W_U
  greedy(ids [1,T], n, inject=None)             -> list of n token ids, greedy, patch baked into the prefill cache
  inject = (layer, mask [B,T] bool or None, delta [d] or [B,T,d]): delta is ADDED to the residual after block `layer`
           at the masked positions (replace = delta = h - x_clean, as X1 / X3 do).
"""
import torch, torch.nn.functional as F


def _f32(z):
    return z.float() if z.dtype in (torch.bfloat16, torch.float16) else z


class MinBackend:
    def __init__(s, tok, q):
        from .qwen3_min import forward, logits as true_logits
        s.tok, s.q, s._forward, s._logits = tok, q, forward, true_logits
        s.n_layers, s.d, s.V, s.device = q.w.n_layers, q.d, q.w.lm.shape[0], q.w.emb.device
        s.act_dtype = q.act_dtype; s.name = "min"

    def _cast(s, inject):
        if inject is None: return None
        l, m, d = inject
        return (l, m, d.to(s.act_dtype or s.q.w.emb.dtype))

    def resid(s, ids, layer):
        with torch.no_grad(): return s.q.resid(ids, layer)

    def final(s, ids, inject=None):
        with torch.no_grad(): return s._forward(s.q.w, ids, s._cast(inject), act_dtype=s.act_dtype)[0]

    def last_logp(s, ids, inject=None):
        with torch.no_grad():
            hL, _ = s._forward(s.q.w, ids, s._cast(inject), act_dtype=s.act_dtype)
            return F.log_softmax(_f32(s._logits(s.q.w, hL[:, -1])), -1).double()

    def logits(s, h):
        with torch.no_grad(): return _f32(s._logits(s.q.w, h))

    def unembed(s):
        return s.q.w.lm.float()

    def greedy(s, ids, n, inject=None):
        with torch.no_grad():
            hL, cache = s._forward(s.q.w, ids, s._cast(inject), act_dtype=s.act_dtype); out = []
            for _ in range(n):
                t = int(s._logits(s.q.w, hL[0, -1]).argmax()); out.append(t)
                hL, cache = s._forward(s.q.w, torch.tensor([[t]], device=ids.device), None, cache, act_dtype=s.act_dtype)
        return out


class _Stop(Exception):
    pass


class HFHooksBackend:
    BLOCK_PATHS = ("model.layers", "model.language_model.layers", "base_model.model.model.layers")
    NORM_PATHS = ("model.norm", "model.language_model.norm", "base_model.model.model.norm")

    def __init__(s, model, tok):
        s.model, s.tok = model.eval(), tok
        s.blocks = s._find(s.BLOCK_PATHS, "decoder blocks"); s.norm = s._find(s.NORM_PATHS, "final norm")
        s.head = model.get_output_embeddings(); s.n_layers = len(s.blocks)
        s.d = s.head.weight.shape[1]; s.V = s.head.weight.shape[0]; s.device = s.head.weight.device; s.name = "hf"

    def _find(s, paths, what):
        for path in paths:
            obj = s.model
            try:
                for part in path.split("."): obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise AttributeError(f"{what} not found on this model (tried {paths})")

    @staticmethod
    def _out_h(out):
        return out[0] if isinstance(out, tuple) else out

    @staticmethod
    def _set_h(out, h):
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def _hooks(s, inject, capture, stop_layer):
        """forward hooks: add the patch after block `inject[0]`, store block outputs for `capture`, and raise _Stop
        after `stop_layer` (the blocks above it are never run)."""
        store, handles = {}, []
        li = inject[0] if inject is not None else None
        want = set(capture or ())
        for l in sorted(want | ({li} if li is not None else set()) | ({stop_layer} if stop_layer is not None else set())):
            def hook(mod, inp, out, l=l):
                h = s._out_h(out)
                if l == li:
                    _, mask, delta = inject
                    B, T = h.shape[:2]
                    d = delta if delta.dim() == 3 else delta.view(1, 1, -1).expand(B, T, -1)
                    d = d.to(device=h.device, dtype=h.dtype)
                    if mask is not None: d = d * mask.to(device=h.device, dtype=h.dtype)[..., None]
                    h = h + d; out = s._set_h(out, h)
                if l in want: store[l] = h
                if l == stop_layer: raise _Stop()
                return out
            handles.append(s.blocks[l].register_forward_hook(hook))
        return store, handles

    def _run(s, ids, inject=None, capture=(), stop_layer=None, past=None, use_cache=False):
        store, handles = s._hooks(inject, capture, stop_layer); out = None
        try:
            with torch.no_grad():
                kw = {"input_ids": ids, "use_cache": use_cache}
                if past is not None: kw["past_key_values"] = past
                try:
                    out = s.model(**kw, logits_to_keep=1)  # the head is applied by us; keep the model's own to one row
                except TypeError:
                    out = s.model(**kw)
        except _Stop:
            out = None
        finally:
            for h in handles: h.remove()
        return store, out

    def resid(s, ids, layer):
        store, _ = s._run(ids, capture=[layer], stop_layer=layer); return store[layer]

    def final(s, ids, inject=None):
        store, _ = s._run(ids, inject=inject, capture=[s.n_layers - 1]); return store[s.n_layers - 1]

    def logits(s, h):
        with torch.no_grad():
            x = s.norm(h.to(s.norm.weight.dtype)).float()
            return F.linear(x, s.head.weight.float(), None if s.head.bias is None else s.head.bias.float())

    def last_logp(s, ids, inject=None):
        hL = s.final(ids, inject)[:, -1]
        return F.log_softmax(s.logits(hL), -1).double()

    def unembed(s):
        return s.head.weight.float()

    def greedy(s, ids, n, inject=None):
        L = s.n_layers - 1; out = []
        store, o = s._run(ids, inject=inject, capture=[L], use_cache=True); past = o.past_key_values
        hL = store[L][:, -1]
        for _ in range(n):
            t = int(s.logits(hL[0]).argmax()); out.append(t)
            store, o = s._run(torch.tensor([[t]], device=ids.device), capture=[L], past=past, use_cache=True); past = o.past_key_values
            hL = store[L][:, -1]
        return out


def make_backend(kind, tok, m, q=None):
    """kind 'min' (Qwen3Min over the HF model m; q optional, built if absent) or 'hf' (hooks on m)."""
    if kind == "min":
        if q is None:
            from .qwen3_min import Qwen3Min
            q = Qwen3Min(m)
        return MinBackend(tok, q)
    if kind == "hf": return HFHooksBackend(m, tok)
    raise ValueError(kind)
