#!/usr/bin/env python3
"""A GPT whose architecture can change mid-training, plus honest FLOP accounting.

Ported from reverse_distillation/autoresearch_grow/train.py. Two deliberate
changes from the inherited version, both documented in NOTES.md:

1. grow_width no longer zeros new dimensions. The inherited operator zeroed the
   whole new model and copied old weights into the top-left, leaving every new
   width dim at exactly 0 -- new attention heads got Q,K=0, uniform softmax, and
   zero gradient. That is the softmax barrier, reproduced inside the growth
   direction that was supposed to escape it. A mid-training model has nothing
   precious to preserve, so new weights are initialized at real magnitude and
   gradients flow. `--new-init zero` restores the old behaviour for ablation.
2. LayerNorm gain/bias on new dims init to (1, 0) rather than (0, 0). Zeroing
   them left the new dims contributing nothing while still shifting LayerNorm's
   statistics -- neither preserving the function nor using the capacity.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

INIT_STD = 0.02


class Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.qkv = nn.Linear(hidden_dim, 3 * num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.out_proj(out)


class Block(nn.Module):
    def __init__(self, hidden_dim, num_heads, head_dim, intermediate_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = Attention(hidden_dim, num_heads, head_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.down(F.gelu(self.up(x)))


class GrowableGPT(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, head_dim, max_seq_len=512):
        super().__init__()
        self.vocab_size, self.hidden_dim, self.head_dim = vocab_size, hidden_dim, head_dim
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.blocks = nn.ModuleList(
            [Block(hidden_dim, num_heads, head_dim, hidden_dim * 4) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=INIT_STD)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, labels=None):
        T = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        for b in self.blocks:
            x = b(x)
        logits = self.lm_head(self.ln_f(x))
        if labels is None:
            return logits, None
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab_size), labels[:, 1:].reshape(-1)
        )
        return logits, loss

    def arch(self):
        b = self.blocks[0]
        return f"{len(self.blocks)}L {b.attn.num_heads}H {self.hidden_dim}D"


def count_params(model):
    """Trainable params; shared (tied) tensors counted once."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flops_per_token(model, seq_len):
    """Forward+backward FLOPs per token.

    6*N covers every matmul weight, including the tied embed/lm_head projection
    (a real matmul; the input-side lookup is free but shares the same tensor, so
    it is not double counted). 12*L*T*d is the attention score/value product,
    which has no parameters and so is invisible to 6*N.
    """
    n_layers = len(model.blocks)
    d = model.hidden_dim
    return 6 * count_params(model) + 12 * n_layers * seq_len * d


def _grown(old, shape, std=INIT_STD, fill="scaled"):
    """Write `old` into the top-left of a fresh tensor of the new shape."""
    if fill == "zero":
        t = torch.zeros(shape, device=old.device, dtype=old.dtype)
    else:
        t = torch.randn(shape, device=old.device, dtype=old.dtype) * std
    t[tuple(slice(0, s) for s in old.shape)] = old
    return t


def grow_width(model, new_hidden_dim, device, new_init="scaled"):
    """Widen every layer. head_dim is held constant, so heads are added."""
    old_dim = model.hidden_dim
    head_dim = model.head_dim
    new_heads = new_hidden_dim // head_dim
    new_hidden_dim = new_heads * head_dim
    if new_hidden_dim <= old_dim:
        return model

    new = GrowableGPT(
        model.vocab_size, new_hidden_dim, len(model.blocks), new_heads, head_dim, model.max_seq_len
    ).to(device)

    def cp(dst_param, src, fill=new_init):
        dst_param.data = _grown(src.data, dst_param.shape, fill=fill)

    cp(new.embed.weight, model.embed.weight)
    cp(new.pos_embed.weight, model.pos_embed.weight)

    for nb, ob in zip(new.blocks, model.blocks):
        # qkv rows are [3, heads, head_dim]-major: copy each of q,k,v separately
        # so old heads land in the right rows rather than being smeared.
        old_h = ob.attn.num_heads
        nb.attn.qkv.weight.data = (
            torch.randn_like(nb.attn.qkv.weight) * INIT_STD
            if new_init != "zero"
            else torch.zeros_like(nb.attn.qkv.weight)
        )
        ow = ob.attn.qkv.weight.data.view(3, old_h, head_dim, old_dim)
        nw = nb.attn.qkv.weight.data.view(3, new_heads, head_dim, new_hidden_dim)
        nw[:, :old_h, :, :old_dim] = ow
        cp(nb.attn.out_proj.weight, ob.attn.out_proj.weight)
        cp(nb.up.weight, ob.up.weight)
        cp(nb.down.weight, ob.down.weight)
        for a, b in ((nb.ln1, ob.ln1), (nb.ln2, ob.ln2)):
            a.weight.data[:old_dim] = b.weight.data
            a.bias.data[:old_dim] = b.bias.data
    new.ln_f.weight.data[:old_dim] = model.ln_f.weight.data
    new.ln_f.bias.data[:old_dim] = model.ln_f.bias.data
    return new


def grow_depth(model, device, new_init="scaled"):
    """Append a block. Output projections zero-init so it starts as identity.

    Zeroing the *output* side is safe: the residual path still delivers gradient
    to the block's internal weights on the very next step. Zeroing an input-side
    projection is what kills heads permanently.
    """
    block = copy.deepcopy(model.blocks[-1])
    nn.init.zeros_(block.attn.out_proj.weight)
    nn.init.zeros_(block.down.weight)
    model.blocks.append(block.to(device))
    return model


def apply_growth(model, action, device, new_init="scaled"):
    if action["action"] == "width":
        return grow_width(model, action["target_hidden"], device, new_init)
    if action["action"] == "depth":
        return grow_depth(model, device, new_init)
    raise ValueError(f"unknown growth action: {action}")
