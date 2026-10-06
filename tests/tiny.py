import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def tiny(seed=0, dtype=torch.float64, **kw):
    torch.manual_seed(seed)
    cfg = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
               num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=False,
               attention_bias=False, rms_norm_eps=1e-6, use_sliding_window=False)
    cfg.update(kw)
    m = Qwen3ForCausalLM(Qwen3Config(**cfg)).to(dtype).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m
