import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.coord_utils import window_partition, window_reverse


class SpectralChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 4)),
            nn.GELU(),
            nn.Linear(max(dim // reduction, 4), dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out


class HATBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift=False, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.spectral_gate = SpectralChannelAttention(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x_ln = x.permute(0, 2, 3, 1)
        x_ln = self.norm1(x_ln)

        if self.shift_size > 0:
            x_ln = torch.roll(x_ln, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x_ln, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x_attn = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x_attn = torch.roll(x_attn, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x_attn = x_attn.permute(0, 3, 1, 2)
        x = shortcut + x_attn

        x_mlp_in = x.permute(0, 2, 3, 1)
        x_mlp_in = self.norm2(x_mlp_in)
        x_mlp_out = self.mlp(x_mlp_in).permute(0, 3, 1, 2)
        x = x + x_mlp_out

        x = self.spectral_gate(x)
        return x


class HATEncoder(nn.Module):
    def __init__(self, in_ch=4, embed_dim=128, cond_dim=128, depth=4, num_heads=4, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.embed = nn.Conv2d(in_ch, embed_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            HATBlock(embed_dim, num_heads, window_size, shift=(i % 2 == 1))
            for i in range(depth)
        ])
        self.cond_proj = nn.Conv2d(embed_dim, cond_dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        feat = self.embed(x_padded)
        for block in self.blocks:
            feat = block(feat)
        feat = feat[:, :, :H, :W]

        cond_map = self.cond_proj(feat)
        cond_tokens = cond_map.flatten(2).transpose(1, 2)
        return cond_tokens, feat
