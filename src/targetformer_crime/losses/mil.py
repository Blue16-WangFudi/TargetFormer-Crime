from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class MilLossConfig:
    margin: float = 1.0
    lambda_smooth: float = 0.1
    lambda_sparse: float = 0.001
    lambda_proto_div: float = 0.0


def mil_ranking_loss(abn_scores: torch.Tensor, nor_scores: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    # Scores are per-segment probabilities in [0,1], shape (B,N) or (N,).
    abn_max = abn_scores.max(dim=-1).values
    nor_max = nor_scores.max(dim=-1).values
    loss = torch.relu(torch.tensor(margin, device=abn_scores.device, dtype=abn_scores.dtype) - abn_max + nor_max)
    return loss.mean()


def smoothness_loss(scores: torch.Tensor) -> torch.Tensor:
    # scores: (B,N)
    if scores.size(-1) < 2:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    diffs = scores[..., 1:] - scores[..., :-1]
    return (diffs * diffs).mean()


def sparsity_loss_abnormal(scores_abn: torch.Tensor) -> torch.Tensor:
    # Encourage sparse peaks in abnormal video: sum_i s_i (normalized).
    return scores_abn.mean()


def compute_total_loss(
    abn_scores: torch.Tensor,
    nor_scores: torch.Tensor,
    cfg: MilLossConfig,
    proto_div_loss: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    rank = mil_ranking_loss(abn_scores, nor_scores, margin=cfg.margin)
    smooth = 0.5 * (smoothness_loss(abn_scores) + smoothness_loss(nor_scores))
    sparse = sparsity_loss_abnormal(abn_scores)
    proto_div = proto_div_loss if proto_div_loss is not None else torch.zeros_like(rank)

    total = rank + cfg.lambda_smooth * smooth + cfg.lambda_sparse * sparse + cfg.lambda_proto_div * proto_div
    return {"total": total, "rank": rank, "smooth": smooth, "sparse": sparse, "proto_div": proto_div}

