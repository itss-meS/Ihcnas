import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.satsrflow import SatSRFlow


def test_forward_train_shapes():
    model = SatSRFlow(latent_dim=32, cond_dim=64, hidden_dim=64, flow_depth=2, hat_depth=2)
    lr = torch.rand(2, 4, 64, 64)
    gt = torch.rand(2, 4, 256, 256)
    loss, loss_dict = model.forward_train(lr, gt)
    assert loss.dim() == 0
    assert 'loss_flow' in loss_dict
    assert 'loss_recon' in loss_dict


def test_infer_shapes():
    model = SatSRFlow(latent_dim=32, cond_dim=64, hidden_dim=64, flow_depth=2, hat_depth=2)
    lr = torch.rand(1, 4, 64, 64)
    mean_sr, uncertainty = model.infer(lr, (256, 256), num_steps=2, ensemble_size=2)
    assert mean_sr.shape == (1, 4, 256, 256)
    assert uncertainty.shape == (1, 1, 256, 256)


def test_stage1_shapes():
    model = SatSRFlow(latent_dim=32, cond_dim=64, hidden_dim=64, flow_depth=2, hat_depth=2)
    lr = torch.rand(2, 4, 64, 64)
    gt = torch.rand(2, 4, 256, 256)
    loss, loss_dict = model.forward_stage1(lr, gt)
    assert loss.dim() == 0
    assert 'loss_recon' in loss_dict


def test_arbitrary_scale_shapes():
    model = SatSRFlow(latent_dim=32, cond_dim=64, hidden_dim=64, flow_depth=2, hat_depth=2)
    lr = torch.rand(1, 4, 64, 64)
    mean_sr, uncertainty = model.infer(lr, (237, 237), num_steps=2, ensemble_size=2)
    assert mean_sr.shape == (1, 4, 237, 237)
    assert uncertainty.shape == (1, 1, 237, 237)


if __name__ == '__main__':
    test_forward_train_shapes()
    test_infer_shapes()
    test_stage1_shapes()
    test_arbitrary_scale_shapes()
    print('all shape tests passed')
