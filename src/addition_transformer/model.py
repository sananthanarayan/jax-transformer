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

import jax
import jax.numpy as jnp
from flax import nnx

from .vocab import VOCAB_SIZE


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    max_len: int = 20
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.0


class CausalSelfAttention(nnx.Module):
    def __init__(self, cfg: TransformerConfig, *, rngs: nnx.Rngs):
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nnx.Linear(cfg.d_model, 3 * cfg.d_model, use_bias=False, rngs=rngs)
        self.out = nnx.Linear(cfg.d_model, cfg.d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        # (B, T, 3, H, D) -> (3, B, H, T, D)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

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
        self.pos_emb = nnx.Embed(cfg.max_len, cfg.d_model, rngs=rngs)
        self.blocks = nnx.List([Block(cfg, rngs=rngs) for _ in range(cfg.n_layers)])
        self.ln_f = nnx.RMSNorm(cfg.d_model, rngs=rngs)
        self.head = nnx.Linear(cfg.d_model, cfg.vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, ids: jax.Array) -> jax.Array:
        """ids: (B, T) int32 -> logits: (B, T, vocab_size)."""
        B, T = ids.shape
        pos = jnp.arange(T)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def count_params(model: nnx.Module) -> int:
    _, params, *_ = nnx.split(model, nnx.Param, ...)
    leaves = jax.tree_util.tree_leaves(params)
    return sum(int(jnp.size(leaf)) for leaf in leaves)
