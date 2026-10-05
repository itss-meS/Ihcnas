import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.hat_encoder import HATEncoder
from src.model.rectified_flow import RectifiedFlowBackbone
from src.model.liif_decoder import LIIFDecoder


def charbonnier_loss(pred, target, eps=1e-6):
    diff = pred - target
    loss = torch.sqrt(diff * diff + eps)
    return loss.mean()


class GTLatentEncoder(nn.Module):
    def __init__(self, in_ch=4, latent_dim=64, scale_factor=4):
        super().__init__()
        num_down = max(int(round(math.log2(scale_factor))), 0)
        layers = [nn.Conv2d(in_ch, latent_dim, kernel_size=3, padding=1), nn.GELU()]
        for _ in range(num_down):
            layers.append(nn.Conv2d(latent_dim, latent_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SatSRFlow(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, latent_dim=64, cond_dim=128, hidden_dim=128,
                 embed_dim=128, flow_depth=6, hat_depth=4, window_size=8, scale_factor=4):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.latent_dim = latent_dim
        self.scale_factor = scale_factor

        self.hat_encoder = HATEncoder(in_ch, embed_dim, cond_dim, hat_depth, num_heads=4, window_size=window_size)
        self.feat_to_latent = nn.Conv2d(embed_dim, latent_dim, kernel_size=1)
        self.gt_latent_encoder = GTLatentEncoder(in_ch, latent_dim, scale_factor)
        self.flow_backbone = RectifiedFlowBackbone(latent_dim, cond_dim, flow_depth)
        self.liif_decoder = LIIFDecoder(latent_dim, out_ch, hidden_dim)

    def bicubic_base(self, lr_img, target_shape):
        return F.interpolate(lr_img, size=target_shape, mode='bicubic', align_corners=False)

    def forward_stage1(self, lr_img, hr_gt):
        H_out, W_out = hr_gt.shape[-2:]
        cond_tokens, lr_feat = self.hat_encoder(lr_img)
        latent = self.feat_to_latent(lr_feat)
        liif_residual = self.liif_decoder(latent, (H_out, W_out))
        base = self.bicubic_base(lr_img, (H_out, W_out))
        pred_hr = torch.clamp(base + liif_residual, 0.0, 1.0)
        loss = charbonnier_loss(pred_hr, hr_gt)
        loss_dict = {'loss_recon': loss.detach(), 'loss_total': loss.detach()}
        return loss, loss_dict

    def forward_train(self, lr_img, hr_gt):
        B, _, H_in, W_in = lr_img.shape
        H_out, W_out = hr_gt.shape[-2:]

        cond_tokens, lr_feat = self.hat_encoder(lr_img)

        x1 = self.gt_latent_encoder(hr_gt)
        x0 = torch.randn_like(x1)

        t = torch.sigmoid(torch.randn(B, device=lr_img.device))
        t_expand = t.view(B, 1, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x1
        v_target = x1 - x0

        v_pred = self.flow_backbone(x_t, t, cond_tokens)
        loss_flow = F.mse_loss(v_pred, v_target)

        refined_latent = x_t + (1 - t_expand) * v_pred
        liif_residual = self.liif_decoder(refined_latent, (H_out, W_out))
        base = self.bicubic_base(lr_img, (H_out, W_out))
        pred_hr = torch.clamp(base + liif_residual, 0.0, 1.0)

        loss_recon = charbonnier_loss(pred_hr, hr_gt)
        total_loss = loss_flow + 0.5 * loss_recon

        loss_dict = {
            'loss_flow': loss_flow.detach(),
            'loss_recon': loss_recon.detach(),
            'loss_total': total_loss.detach(),
        }
        return total_loss, loss_dict

    @torch.no_grad()
    def infer(self, lr_img, target_shape, num_steps=5, ensemble_size=3):
        was_training = self.training
        self.eval()

        B, _, H_in, W_in = lr_img.shape
        H_out, W_out = target_shape

        cond_tokens, lr_feat = self.hat_encoder(lr_img)
        base = self.bicubic_base(lr_img, (H_out, W_out))

        outputs = []
        for _ in range(ensemble_size):
            x = torch.randn(B, self.latent_dim, H_in, W_in, device=lr_img.device)
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t_val = step * dt
                t = torch.full((B,), t_val, device=lr_img.device)
                v_pred = self.flow_backbone(x, t, cond_tokens)
                x = x + v_pred * dt

            liif_residual = self.liif_decoder(x, (H_out, W_out))
            pred_hr = torch.clamp(base + liif_residual, 0.0, 1.0)
            outputs.append(pred_hr)

        stacked = torch.stack(outputs, dim=0)
        mean_sr = stacked.mean(dim=0)
        if ensemble_size > 1:
            uncertainty_map = stacked.var(dim=0).mean(dim=1, keepdim=True)
        else:
            uncertainty_map = torch.zeros(B, 1, H_out, W_out, device=lr_img.device)

        if was_training:
            self.train()
        return mean_sr, uncertainty_map
