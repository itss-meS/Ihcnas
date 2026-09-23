import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.coord_utils import make_coord


class LIIFDecoder(nn.Module):
    def __init__(self, latent_dim=64, out_ch=4, hidden_dim=128, num_layers=4):
        super().__init__()
        in_dim = latent_dim + 2 + 2
        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            d = hidden_dim
        layers.append(nn.Linear(d, out_ch))
        self.imnet = nn.Sequential(*layers)

    def forward(self, latent, target_shape):
        B, C, H_lat, W_lat = latent.shape
        H_out, W_out = target_shape
        device = latent.device

        coord = make_coord((H_out, W_out), flatten=True).to(device)
        coord = coord.unsqueeze(0).expand(B, -1, -1)

        cell = torch.ones_like(coord)
        cell[:, :, 0] = cell[:, :, 0] * (2.0 / H_out)
        cell[:, :, 1] = cell[:, :, 1] * (2.0 / W_out)

        grid = coord.flip(-1).unsqueeze(1)
        sampled_feat = F.grid_sample(
            latent, grid, mode='bilinear', align_corners=False, padding_mode='border'
        )
        sampled_feat = sampled_feat.squeeze(2).transpose(1, 2)

        lat_coord = make_coord((H_lat, W_lat), flatten=True).to(device)
        lat_coord = lat_coord.unsqueeze(0).expand(B, -1, -1)

        idx_h = ((coord[:, :, 0] + 1) / 2 * H_lat).long().clamp(0, H_lat - 1)
        idx_w = ((coord[:, :, 1] + 1) / 2 * W_lat).long().clamp(0, W_lat - 1)
        nearest_idx = idx_h * W_lat + idx_w
        nearest_coord = torch.gather(
            lat_coord, 1, nearest_idx.unsqueeze(-1).expand(-1, -1, 2)
        )

        rel_coord = coord - nearest_coord
        rel_coord = torch.stack(
            [rel_coord[:, :, 0] * H_lat, rel_coord[:, :, 1] * W_lat], dim=-1
        )

        inp = torch.cat([sampled_feat, rel_coord, cell], dim=-1)
        out = self.imnet(inp)
        out = out.transpose(1, 2).view(B, -1, H_out, W_out)
        return out
