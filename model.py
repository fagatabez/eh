# model.py — your AI brain, built from scratch
import torch
import torch.nn as nn
import math

class Config:
    # ── Bias / personality settings ──────────────────────────
    name        = "MyAI"
    personality = "helpful, honest, curious"
    # ─────────────────────────────────────────────────────────

    vocab_size  = 5000    # how many words it knows
    seq_len     = 128     # how many words it reads at once
    embed_dim   = 512     # how "rich" each word representation is || jak długo będzie trenonwany more dłóżej mniej krócej
    num_heads   = 8       # attention heads (how many things it focuses on)
    num_layers  = 8       # how deep the thinking goes
    dropout     = 0.1

class SelfAttention(nn.Module):
    """The core mechanic — lets each word look at every other word"""
    def __init__(self, cfg):
        super().__init__()
        self.heads   = cfg.num_heads
        self.d       = cfg.embed_dim // cfg.num_heads
        self.qkv     = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim)
        self.out     = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.drop    = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.d).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale   = math.sqrt(self.d)
        scores  = (q @ k.transpose(-2,-1)) / scale
        mask    = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        scores  = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = self.drop(weights)
        out     = (weights @ v).transpose(1,2).reshape(B, T, C)
        return self.out(out)

class TransformerBlock(nn.Module):
    """One layer of thinking"""
    def __init__(self, cfg):
        super().__init__()
        self.attn  = SelfAttention(cfg)
        self.ff    = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout)
        )
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class MyAI(nn.Module):
    """Your complete AI — no external models"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.seq_len,    cfg.embed_dim)
        self.blocks  = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm    = nn.LayerNorm(cfg.embed_dim)
        self.head    = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.drop    = nn.Dropout(cfg.dropout)

    def forward(self, tokens):
        B, T    = tokens.shape
        pos     = torch.arange(T, device=tokens.device)
        x       = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))
        x       = self.blocks(x)
        x       = self.norm(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())