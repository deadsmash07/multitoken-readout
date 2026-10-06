"""E3 (plumbing gate) and E4 (finite differences vs exact JVP).

side_result: identity (4) of tests/toy_identities.py on a real lens. Two perturbed passes (+/- eps h at every
source) summed over the valid targets give |T| J h; W_U of that is the lens readout D h. Pseudo-logits are linear
in the residual, so the sum over targets is taken on the residuals before the unembedding. Spearman over the
top-`top` tokens of D h, one number per activation. The proposal's gate is Spearman >= 0.95.

fd_curve: relative error of the central-difference estimate of J(p) h (model q_fd in whatever dtype J-beam will
use) against the exact JVP through q_ref (fp32 or fp64), as a function of rho = ||eps h|| / mean ||h_t||.
"""
import torch
from ..model.qwen3_min import pseudo_logits
from ..lens.forward import ContextBank, jvp_exact
from .common import spearman


def source_norm(q, bank, layer):
    with torch.no_grad():
        return q.resid(bank.ids, layer)[:, bank.mask].float().norm(dim=-1).mean().item()


def side_result(q, bank, D, H, layer, rho=0.1, top=100):
    """H [n, d] activations; D [V, d] on the model device. Returns (spearman [n], fd-vs-exact rel. err [n])."""
    hn = source_norm(q, bank, layer); dt = q.w.emb.dtype
    m_ = bank.mask.view(1, -1).expand(bank.C, -1)
    sps, sps_all, sps_1k, errs = [], [], [], []
    for h in H:
        h = h.to(D.device); eps = rho * hn / h.norm().item()
        with torch.no_grad():
            hp, _ = q.prefill(bank.ids, inject=(layer, m_, eps * h.to(dt)))
            hm, _ = q.prefill(bank.ids, inject=(layer, m_, -eps * h.to(dt)))
            dsum = (hp.float() - hm.float())[:, bank.mask].sum(1).mean(0) / (2 * eps * bank.nT)  # [d] = J h
            fd = pseudo_logits(q.w, dsum.to(q.w.lm.dtype)).float()
            exact = (D @ h.to(D.dtype)).float()
        idx = torch.topk(exact, top).indices; idx1k = torch.topk(exact, min(1000, exact.numel())).indices
        sps.append(spearman(fd[idx], exact[idx])); sps_1k.append(spearman(fd[idx1k], exact[idx1k])); sps_all.append(spearman(fd, exact))
        errs.append(float((fd - exact).norm() / exact.norm()))
    return {"top": torch.tensor(sps), "top1k": torch.tensor(sps_1k), "all": torch.tensor(sps_all), "err": torch.tensor(errs)}


def fd_curve(q_fd, q_ref, contexts_fd, contexts_ref, layer, h, prefix, rhos, skip_first=16):
    """returns ({rho: rel err}, mean source norm). contexts_* are the same ids on each model's device."""
    bank = ContextBank(q_fd, contexts_fd, skip_first)
    hn = source_norm(q_fd, bank, layer)
    ref_ids = torch.stack([c.view(-1) for c in contexts_ref])
    ex = jvp_exact(q_ref, ref_ids, layer, bank.mask.to(ref_ids.device), h.to(q_ref.w.emb.dtype).to(ref_ids.device), prefix=list(prefix)).double().cpu()
    out = {}
    for rho in rhos:
        eps = rho * hn / h.norm().item()
        cp, cm = bank.perturbed(layer, h.to(q_fd.w.emb.dtype).to(bank.ids.device), eps)
        uhat, _ = bank.direction_estimate(list(prefix), cp, cm, eps)
        out[rho] = float((uhat.double().cpu() - ex).norm() / ex.norm())
    return out, hn
