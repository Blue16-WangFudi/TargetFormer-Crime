from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from targetformer_crime.models.targetformer import TargetFormerOutput, masked_softmax


class GlobalMLP(nn.Module):
    def __init__(self, input_dim: int, num_segments: int, max_k: int, d_model: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_segments = int(num_segments)
        self.max_k = int(max_k)
        self.d_model = int(d_model)

        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.token_scorer = nn.Linear(self.d_model, 1)
        self.score_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, 1),
        )

    def forward(self, tokens: torch.Tensor, masks: torch.Tensor) -> TargetFormerOutput:
        b, n, k, d = tokens.shape
        if n != self.num_segments:
            raise ValueError(f"num_segments mismatch: model={self.num_segments} got={n}")
        if k > self.max_k:
            raise ValueError(f"k too large: model_max_k={self.max_k} got={k}")
        if d != self.input_dim:
            raise ValueError(f"input_dim mismatch: model={self.input_dim} got={d}")

        h = self.input_proj(tokens)
        token_logits = self.token_scorer(h).squeeze(-1)
        token_weights = masked_softmax(token_logits, mask=masks, dim=-1)
        seg_emb = torch.sum(h * token_weights.unsqueeze(-1), dim=2)
        seg_logits = self.score_head(seg_emb).squeeze(-1)
        seg_scores = torch.sigmoid(seg_logits)
        return TargetFormerOutput(
            segment_scores=seg_scores,
            segment_logits=seg_logits,
            segment_embeddings=seg_emb,
            token_weights=token_weights,
            proto_weights=None,
        )

    def prototype_diversity_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)


class YoloGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_segments: int,
        max_k: int,
        d_model: int = 256,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_segments = int(num_segments)
        self.max_k = int(max_k)
        self.d_model = int(d_model)

        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.token_scorer = nn.Linear(self.d_model, 1)
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, tokens: torch.Tensor, masks: torch.Tensor) -> TargetFormerOutput:
        b, n, k, d = tokens.shape
        if n != self.num_segments:
            raise ValueError(f"num_segments mismatch: model={self.num_segments} got={n}")
        if k > self.max_k:
            raise ValueError(f"k too large: model_max_k={self.max_k} got={k}")
        if d != self.input_dim:
            raise ValueError(f"input_dim mismatch: model={self.input_dim} got={d}")

        h = self.input_proj(tokens)  # (B,N,K,D)
        token_logits = self.token_scorer(h).squeeze(-1)
        token_weights = masked_softmax(token_logits, mask=masks, dim=-1)
        seg_emb = torch.sum(h * token_weights.unsqueeze(-1), dim=2)  # (B,N,D)

        out, _ = self.gru(seg_emb)  # (B,N,H)
        seg_logits = self.score_head(out).squeeze(-1)
        seg_scores = torch.sigmoid(seg_logits)
        return TargetFormerOutput(
            segment_scores=seg_scores,
            segment_logits=seg_logits,
            segment_embeddings=out,
            token_weights=token_weights,
            proto_weights=None,
        )

    def prototype_diversity_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)

