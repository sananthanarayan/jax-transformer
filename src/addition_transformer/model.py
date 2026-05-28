"""Decoder-only transformer in Flax NNX.

Architectural notes:

* Pre-norm (RMSNorm before attention and MLP), GeLU MLPs, learned positional
  embeddings, untied output head.
* Causal mask is built inside ``CausalSelfAttention`` — there's no separate
  attention-mask argument to keep the call signature trivial.
* Default config is sized to ~10.6M parameters; see :class:`TransformerConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx

from .vocab import VOCAB_SIZE

PosEncoding = Literal["learned", "none", "rope", "abacus"]


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    max_len: int = 32
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.0
    pos_encoding: PosEncoding = "learned"
    max_digit_pos: int = 16  # Abacus: positions 1..max_digit_pos plus 0 for non-digit


def _rope_table(T: int, d_head: int) -> tuple[jax.Array, jax.Array]:
    """Standard rotary table. Returns (cos, sin) of shape (T, d_head // 2)."""
    half = d_head // 2
    freq_idx = jnp.arange(half, dtype=jnp.float32)
    theta = 10000.0 ** (-2.0 * freq_idx / d_head)
    pos = jnp.arange(T, dtype=jnp.float32)
    angles = pos[:, None] * theta[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def _apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate consecutive pairs of feature dims. ``x`` is (B, H, T, D)."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    rot_even = x_even * cos_b - x_odd * sin_b
    rot_odd = x_even * sin_b + x_odd * cos_b
    out = jnp.stack([rot_even, rot_odd], axis=-1)
    return out.reshape(x.shape)


class CausalSelfAttention(nnx.Module):
    def __init__(self, cfg: TransformerConfig, *, rngs: nnx.Rngs):
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.use_rope = cfg.pos_encoding == "rope"
        self.qkv = nnx.Linear(cfg.d_model, 3 * cfg.d_model, use_bias=False, rngs=rngs)
        self.out = nnx.Linear(cfg.d_model, cfg.d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        # (B, T, 3, H, D) -> (3, B, H, T, D)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope:
            cos, sin = _rope_table(T, self.d_head)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        scale = 1.0 / jnp.sqrt(self.d_head).astype(x.dtype)
        att = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        att = jnp.where(mask, att, jnp.finfo(att.dtype).min)
        att = jax.nn.softmax(att, axis=-1)
        out = jnp.einsum("bhts,bhsd->bhtd", att, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, T, C)
        return self.out(out)


class MLP(nnx.Module):
    def __init__(self, cfg: TransformerConfig, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(cfg.d_model, cfg.d_ff, rngs=rngs)
        self.fc2 = nnx.Linear(cfg.d_ff, cfg.d_model, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(jax.nn.gelu(self.fc1(x)))


class Block(nnx.Module):
    def __init__(self, cfg: TransformerConfig, *, rngs: nnx.Rngs):
        self.ln1 = nnx.RMSNorm(cfg.d_model, rngs=rngs)
        self.attn = CausalSelfAttention(cfg, rngs=rngs)
        self.ln2 = nnx.RMSNorm(cfg.d_model, rngs=rngs)
        self.mlp = MLP(cfg, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nnx.Module):
    def __init__(self, cfg: TransformerConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.tok_emb = nnx.Embed(cfg.vocab_size, cfg.d_model, rngs=rngs)
        if cfg.pos_encoding == "learned":
            self.pos_emb = nnx.Embed(cfg.max_len, cfg.d_model, rngs=rngs)
        else:
            self.pos_emb = None
        if cfg.pos_encoding == "abacus":
            self.digit_pos_emb = nnx.Embed(cfg.max_digit_pos + 1, cfg.d_model, rngs=rngs)
        else:
            self.digit_pos_emb = None
        self.blocks = nnx.List([Block(cfg, rngs=rngs) for _ in range(cfg.n_layers)])
        self.ln_f = nnx.RMSNorm(cfg.d_model, rngs=rngs)
        self.head = nnx.Linear(cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        ids: jax.Array,
        digit_positions: jax.Array | None = None,
    ) -> jax.Array:
        """ids: (B, T) int32 -> logits: (B, T, vocab_size).

        ``digit_positions`` is required when ``cfg.pos_encoding == 'abacus'`` and
        is ignored otherwise. Each entry must be in ``[0, cfg.max_digit_pos]``,
        with 0 reserved for non-digit tokens.
        """
        _, T = ids.shape
        x = self.tok_emb(ids)
        if self.pos_emb is not None:
            pos = jnp.arange(T)
            x = x + self.pos_emb(pos)[None, :, :]
        if self.digit_pos_emb is not None:
            if digit_positions is None:
                raise ValueError("Abacus model requires digit_positions")
            x = x + self.digit_pos_emb(digit_positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def count_params(model: nnx.Module) -> int:
    _, params, *_ = nnx.split(model, nnx.Param, ...)
    leaves = jax.tree_util.tree_leaves(params)
    return sum(int(jnp.size(leaf)) for leaf in leaves)
