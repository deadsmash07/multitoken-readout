"""wsbench producer method: register in src/wsbench/produce/methods.py::METHODS as 'sjlens'.

read(h, layer) sees only the activation and the layer. Everything else (lens, contexts, caches, nulls,
background stats) is loaded in bind(); the prompt never reaches this class.

Contexts: a file written by scripts/make_contexts.py ({"ctx": [...], "bg": [...], "T": ...}) or a plain list of
id tensors. Background activations come from `background_path` ({layer: [N, d]}) when given, else from the
`bg` split of the contexts file at bind time (one prefill per background record and layer). The J-beam config
defaults to the one chosen in docs/JBEAM_TUNING.md (beam/jbeam.py Cfg()).
"""
import torch
from .model.qwen3_min import Qwen3Min
from .lens.jlens import load, valid_mask
from .lens.forward import ContextBank
from .beam.jbeam import JBeam, Cfg

try:
    from wsbench.produce.methods import Readout
except Exception:
    from dataclasses import dataclass
    @dataclass
    class Readout:
        tokens: list = None; scores: list = None; samples: list = None


class SJLens:
    name = "sjlens"
    layers = None

    def __init__(s, lens_path, contexts_path, background_path=None, cfg=None, skip_first=16, n_ctx=None, n_bg=None, act_dtype=None):
        s.lens_path, s.contexts_path, s.background_path, s.skip = lens_path, contexts_path, background_path, skip_first
        s.cfg = cfg if cfg is not None else Cfg(); s.n_ctx, s.n_bg, s.act_dtype = n_ctx, n_bg, act_dtype
        s._beams = {}

    def bind(s, backend):
        s.backend = backend
        s.q = Qwen3Min(backend.model, act_dtype=s.act_dtype)
        s.J, _ = load(s.lens_path)
        o = torch.load(s.contexts_path, weights_only=False); dev = s.q.w.emb.device
        ctx, bg = (o["ctx"], o.get("bg", [])) if isinstance(o, dict) else (list(o), [])
        if s.n_ctx: ctx = ctx[: s.n_ctx]
        if s.n_bg: bg = bg[: s.n_bg]
        s.bank = ContextBank(s.q, [c.to(dev) for c in ctx], s.skip)
        s.bg_ctx = [c.to(dev) for c in bg]
        s.bg = {int(l): H for l, H in torch.load(s.background_path, weights_only=False).items()} if s.background_path else {}
        s.tok = backend.tokenizer

    def _background(s, layer):
        if layer not in s.bg:
            assert s.bg_ctx, "no background: give background_path or a contexts file with a 'bg' split"
            mask = valid_mask(s.bg_ctx[0].numel(), s.skip)
            with torch.no_grad(): s.bg[layer] = torch.cat([s.q.resid(c.view(1, -1), layer)[0, mask] for c in s.bg_ctx]).float()
        return s.bg[layer]

    def _beam(s, layer):
        if layer not in s._beams:
            w = s.q.w; D = w.lm.float() @ s.J[layer].float().to(w.lm.device)  # D_l = W_U J_l in float32
            H = s._background(layer).to(D.device); mu = H.mean(0); Hc = H - mu
            g = torch.Generator().manual_seed(layer)
            idx = torch.randperm(Hc.shape[0], generator=g)[: s.cfg.m_null].to(Hc.device)
            s._beams[layer] = JBeam(s.q, s.bank, D, layer, Hc[idx], s.cfg, mu, Hbg=H)
        return s._beams[layer]

    def read(s, h, layer):
        out, _ = s._beam(layer).read(torch.as_tensor(h))
        samples = [s.tok.decode(t).strip() for t, *_ in out]
        return Readout(tokens=[t for t, *_ in out], scores=[z for _, z, *_ in out], samples=samples)
