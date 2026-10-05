import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.satsrflow import SatSRFlow


def test_overfit_single_batch():
    torch.manual_seed(0)
    model = SatSRFlow(latent_dim=32, cond_dim=64, hidden_dim=64, flow_depth=2, hat_depth=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    lr = torch.rand(2, 4, 64, 64)
    gt = torch.rand(2, 4, 256, 256)

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss, _ = model.forward_stage1(lr, gt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


if __name__ == '__main__':
    test_overfit_single_batch()
    print('overfit test passed')
