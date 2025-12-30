import torch

from targetformer_crime.losses.mil import mil_ranking_loss, smoothness_loss, sparsity_loss_abnormal


def test_smoothness_zero_for_constant():
    s = torch.ones(2, 8)
    assert smoothness_loss(s).item() == 0.0


def test_sparsity_mean():
    s = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    assert abs(sparsity_loss_abnormal(s).item() - 0.5) < 1e-6


def test_ranking_loss_margin():
    # abn max=0.2, nor max=0.1 => loss=max(0,1-0.2+0.1)=0.9
    abn = torch.tensor([[0.1, 0.2, 0.05]])
    nor = torch.tensor([[0.1, 0.0, 0.05]])
    loss = mil_ranking_loss(abn, nor, margin=1.0)
    assert abs(loss.item() - 0.9) < 1e-6

