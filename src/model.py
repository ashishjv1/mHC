"""
GPT language model supporting both vanilla residual and mHC modes.

Architecture: pre-norm transformer with rotary-free learned positional
embeddings (nanoGPT style). Uses Flash Attention via PyTorch's SDPA.

Design decision: 24 layers × 1024 hidden × 16 heads = ~356M params.
The user suggested 12–16 layers with 1024 hidden, but that only reaches
~200–255M. Bumping to 24 layers hits the 350–500M target while keeping
head_dim=64, which is efficient for Flash Attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from src.hyper_connections import HyperConnection


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)
        self.use_mhc = config.use_mhc
        if self.use_mhc:
            self.hc_attn = HyperConnection(config.n_streams)
            self.hc_ffn = HyperConnection(config.n_streams)

    def forward(self, x):
        if self.use_mhc:
            S, h = self.hc_attn.route_in(x)
            h = self.attn(self.ln_1(h))
            x = self.hc_attn.route_out(S, h)
            S, h = self.hc_ffn.route_in(x)
            h = self.mlp(self.ln_2(h))
            x = self.hc_ffn.route_out(S, h)
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, bias=config.bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.context_len
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        if self.config.use_mhc:
            # Expand to n streams: place embedding in stream 0, zeros elsewhere
            S = torch.zeros(
                B, T, self.config.n_streams, self.config.d_model,
                device=x.device, dtype=x.dtype,
            )
            S[..., 0, :] = x
            x = S

        for block in self.blocks:
            x = block(x)

        if self.config.use_mhc:
            x = x[..., 0, :]  # collapse: take stream 0

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def get_param_groups(self):
        """Split parameters for Muon (2D weight matrices) vs AdamW (rest)."""
        muon_params = []
        adamw_decay = []
        adamw_nodecay = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            is_embedding = "tok_emb" in name or "pos_emb" in name
            is_hc_matrix = "hc_attn.A" in name or "hc_attn.B" in name or \
                           "hc_ffn.A" in name or "hc_ffn.B" in name
            is_norm = "ln_" in name
            is_bias = name.endswith(".bias")

            if is_embedding or is_hc_matrix or is_norm or is_bias or p.ndim < 2:
                if is_norm or is_bias or p.ndim < 2:
                    adamw_nodecay.append(p)
                else:
                    adamw_decay.append(p)
            else:
                muon_params.append(p)

        return muon_params, adamw_decay, adamw_nodecay
