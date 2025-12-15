"""
Utility module that exposes a configurable sparse-attention implementation.

The original repo ships a TensorFlow-style implementation that does not run
with modern PyTorch because it relies on non-existent `torch.transpose`
semantics and assumes causal decoding.  The TinyViT upgrade in this project
needs a drop-in PyTorch module that can be imported both from regular Python
code and inside notebooks, so we rebuild the implementation here with a
focus on clarity and robustness.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn


def _band_mask(seq_len: int, bandwidth: Optional[int], device: torch.device) -> torch.Tensor:
    """Return a bidirectional band mask centred on the main diagonal."""
    if bandwidth is None or bandwidth <= 0 or bandwidth >= seq_len:
        return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)

    idx = torch.arange(seq_len, device=device)
    dist = (idx[:, None] - idx[None, :]).abs()
    return (dist <= bandwidth).to(torch.bool)


def _strided_mask(seq_len: int, stride: int, device: torch.device) -> torch.Tensor:
    """Return a mask that keeps tokens whose offsets differ by multiples of `stride`."""
    if stride <= 1:
        return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    idx = torch.arange(seq_len, device=device)
    offsets = (idx[:, None] - idx[None, :]).abs()
    return (offsets % stride == 0).to(torch.bool)


class SparseAttention(nn.Module):
    """
    Multi-head sparse attention layer with pluggable sparsity patterns.

    The module expects queries, keys and values in (B, H, N, D) format and
    supports three sparsity configurations:

    - ``all``: default dense attention (used to preserve TinyViT's behaviour).
    - ``local``: keeps tokens within ``local_attn_ctx`` distance from one another.
    - ``strided``: keeps positions whose index difference is a multiple of ``local_attn_ctx``.

    Parameters
    ----------
    num_heads: int
        Number of attention heads.
    attn_mode: str
        One of ``all``, ``local`` or ``strided``.
    local_attn_ctx: Optional[int]
        Context/window size that defines the sparsity bandwidth or stride.
    blocksize: int
        Placeholder kept for backward compatibility; TinyViT does not rely on it
        but downstream users might.
    causal: bool
        When True, restricts attention to the lower triangle in addition to the
        sparsity mask.  Windowed TinyViT attention is bidirectional so the
        default is False.
    """

    _valid_modes = {"all", "local", "strided"}

    def __init__(
        self,
        num_heads: int,
        attn_mode: str = "all",
        local_attn_ctx: Optional[int] = None,
        blocksize: int = 32,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if attn_mode not in self._valid_modes:
            raise ValueError(f"Unsupported attn_mode '{attn_mode}'. "
                             f"Expected one of {sorted(self._valid_modes)}.")
        self.num_heads = num_heads
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        self.blocksize = blocksize
        self.causal = causal

        # Cache previously-built masks per (mode, ctx, seq_len) tuple to avoid work.
        self._mask_cache: Dict[Tuple[str, Optional[int], int, bool, torch.device], torch.Tensor] = {}

    def _get_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        cache_key = (self.attn_mode, self.local_attn_ctx, seq_len, self.causal, device)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        if self.attn_mode == "all" and not self.causal:
            mask = None
        elif self.attn_mode == "all" and self.causal:
            mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
        elif self.attn_mode == "local":
            mask = _band_mask(seq_len, self.local_attn_ctx, device)
            if self.causal:
                mask = mask & torch.ones_like(mask, dtype=torch.bool).tril()
        elif self.attn_mode == "strided":
            stride = self.local_attn_ctx or 1
            mask = _strided_mask(seq_len, stride, device)
            if self.causal:
                mask = mask & torch.ones_like(mask, dtype=torch.bool).tril()
        else:
            raise RuntimeError("Unexpected attention mode")

        if mask is not None:
            # Broadcast later as (1, 1, N, N)
            mask = mask.unsqueeze(0).unsqueeze(0)
            self._mask_cache[cache_key] = mask
        else:
            self._mask_cache[cache_key] = None

        return self._mask_cache[cache_key]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError("SparseAttention expects tensors in (B, H, N, D) format.")
        if q.shape != k.shape:
            raise ValueError("q and k must share the same shape.")
        if v.shape[:3] != q.shape[:3]:
            raise ValueError(
                "q, k and v must share the same batch/head/sequence dimensions."
            )
        if q.size(1) != self.num_heads:
            raise ValueError(
                f"Received {q.size(1)} heads but module was initialised with "
                f"{self.num_heads} heads."
            )

        B, H, N, D = q.shape
        scale = 1.0 / math.sqrt(max(D, 1))

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_bias is not None:
            # Allow passing either (H, N, N) or (1, H, N, N)
            if attn_bias.dim() == 3:
                bias = attn_bias.unsqueeze(0)
            elif attn_bias.dim() == 4:
                bias = attn_bias
            else:
                raise ValueError("attn_bias must have shape (H, N, N) or (1, H, N, N).")
            scores = scores + bias

        mask = self._get_mask(N, q.device)
        if mask is not None:
            fill_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, fill_value)

        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return context


__all__ = ["SparseAttention"]
