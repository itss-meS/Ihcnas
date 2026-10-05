import numpy as np
import torch
import torch.nn.functional as F


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10((max_val ** 2) / mse)


def _gaussian_window(window_size, sigma, channels, device):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(pred, target, window_size=11, max_val=1.0):
    B, C, H, W = pred.shape
    device = pred.device
    window = _gaussian_window(window_size, 1.5, C, device)
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=C)
    mu2 = F.conv2d(target, window, padding=pad, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu1_mu2

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def sam(pred, target, eps=1e-8):
    B, C, H, W = pred.shape
    pred_flat = pred.reshape(B, C, -1)
    target_flat = target.reshape(B, C, -1)
    dot = (pred_flat * target_flat).sum(dim=1)
    pred_norm = pred_flat.norm(dim=1)
    target_norm = target_flat.norm(dim=1)
    cos_angle = dot / (pred_norm * target_norm + eps)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angles = torch.acos(cos_angle)
    angles_deg = angles * 180.0 / np.pi
    return angles_deg.mean()


def ergas(pred, target, scale_factor=4, eps=1e-8):
    rmse_per_band = torch.sqrt(torch.mean((pred - target) ** 2, dim=(0, 2, 3)))
    mean_per_band = torch.mean(target, dim=(0, 2, 3))
    ratio = (rmse_per_band / (mean_per_band + eps)) ** 2
    value = (100.0 / scale_factor) * torch.sqrt(torch.mean(ratio))
    return value


def ndvi(bands_4, eps=1e-6):
    red = bands_4[:, 2:3, :, :]
    nir = bands_4[:, 3:4, :, :]
    return (nir - red) / (nir + red + eps)


def ndvi_mae(pred_4band, target_4band):
    pred_ndvi = ndvi(pred_4band)
    target_ndvi = ndvi(target_4band)
    return torch.mean(torch.abs(pred_ndvi - target_ndvi))


def compute_all_metrics(pred, target, scale_factor=4):
    return {
        'psnr': psnr(pred, target).item(),
        'ssim': ssim(pred, target).item(),
        'sam': sam(pred, target).item(),
        'ergas': ergas(pred, target, scale_factor).item(),
        'ndvi_mae': ndvi_mae(pred, target).item(),
    }
