import math
import torch
import torch.nn as nn


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DiTBlock(nn.Module):
    def __init__(self, dim, cond_dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_kv_proj = nn.Linear(cond_dim, dim * 2)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, cond_tokens):
        x_ln = self.norm1(x)
        attn_out, _ = self.self_attn(x_ln, x_ln, x_ln, need_weights=False)
        x = x + attn_out

        x_ln2 = self.norm_cross(x)
        kv = self.cross_kv_proj(cond_tokens)
        k, v = kv.chunk(2, dim=-1)
        cross_out, _ = self.cross_attn(x_ln2, k, v, need_weights=False)
        x = x + cross_out

        x_ln3 = self.norm2(x)
        x = x + self.mlp(x_ln3)
        return x


class RectifiedFlowBackbone(nn.Module):
    def __init__(self, latent_dim=64, cond_dim=128, depth=6, num_heads=4, time_embed_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(latent_dim, cond_dim, num_heads) for _ in range(depth)
        ])

    def forward(self, x_t, t, cond_tokens):
        B, C, H, W = x_t.shape
        tokens = x_t.flatten(2).transpose(1, 2)
        t_embed = timestep_embedding(t, self.time_embed_dim)
        t_embed = self.time_mlp(t_embed)
        tokens = tokens + t_embed[:, None, :]
        for block in self.blocks:
            tokens = block(tokens, cond_tokens)
        v_pred = tokens.transpose(1, 2).view(B, C, H, W)
        return v_pred