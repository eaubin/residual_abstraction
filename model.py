"""
model.py — a minimal GPT whose residual stream we can read out exactly.

CONTEXT (see README.md): the *residual stream* — the d_model-dimensional
vector at each position that blocks read from and write to — is the candidate
abstract domain of this experiment. We deliberately hand-roll attention
instead of using nn.TransformerEncoder so that the residual stream is an
explicit, inspectable tensor, and we expose it at the same readout point Shai
et al. 2024 probed: after the final block, *before* the final LayerNorm
(ln_f). Everything downstream (analysis.py, refine.py) consumes only this
cached tensor plus exact ground truth from processes.py.

Sizing note: these processes have 3 hidden states, so the minimal sufficient
statistic (the belief simplex) is 2-dimensional. d_model=64 is therefore
hugely over-complete on purpose: the question is whether the *discovered*
abstraction collapses to the known 2-D geometry, not whether the model can fit.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab: int
    seq_len: int = 32
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_mlp: int = 256


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.h, self.dk = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        shp = (B, L, self.h, self.dk)
        q, k, v = (t.view(shp).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)


class Block(nn.Module):
    """Pre-LN block: each sublayer *adds into* the residual stream."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp), nn.GELU(),
            nn.Linear(cfg.d_mlp, cfg.d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

    def forward(self, idx, return_resid: bool = False):
        B, L = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(L, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        resid = x                      # <- the residual stream we study
        logits = self.head(self.ln_f(x))
        return (logits, resid) if return_resid else logits


def pick_device() -> str:
    """macOS-friendly: Apple-silicon MPS if present, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
