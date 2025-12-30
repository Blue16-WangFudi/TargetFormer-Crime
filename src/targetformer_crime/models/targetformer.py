from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    # mask: 1 for valid, 0 for pad
    mask = mask.to(dtype=torch.bool)
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~mask, neg_inf)
    probs = torch.softmax(logits, dim=dim)
    probs = probs.masked_fill(~mask, 0.0)
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return probs / denom


@dataclass(frozen=True)
class TargetFormerOutput:
    segment_scores: torch.Tensor  # (B, N)
    segment_logits: torch.Tensor  # (B, N)
    segment_embeddings: torch.Tensor  # (B, N, D)
    token_weights: torch.Tensor  # (B, N, K)
    proto_weights: Optional[torch.Tensor]  # (B, N, P) or None


class PrototypeBank(nn.Module):
    def __init__(self, d_model: int, num_prototypes: int, tau: float = 0.2) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model) * 0.02)
        self.tau = float(tau)

    def forward(self, seg_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        # seg_emb: (B, N, D)
        p = torch.nn.functional.normalize(self.prototypes, dim=-1)
        z = torch.nn.functional.normalize(seg_emb, dim=-1)
        logits = torch.matmul(z, p.t()) / max(self.tau, 1e-6)  # (B,N,P)
        w = torch.softmax(logits, dim=-1)
        pattern = torch.matmul(w, self.prototypes)  # (B,N,D)
        return {"pattern": pattern, "weights": w, "logits": logits}

    def diversity_loss(self) -> torch.Tensor:
        # Encourage orthogonal prototypes (cosine similarity close to I).
        p = torch.nn.functional.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(p, p.t())  # (P,P)
        eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
        return torch.mean((sim - eye) ** 2)


class TargetFormer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_segments: int,
        max_k: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_prototypes: bool = True,
        num_prototypes: int = 32,
        proto_tau: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_segments = int(num_segments)
        self.max_k = int(max_k)
        self.d_model = int(d_model)

        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.seg_pos = nn.Embedding(self.num_segments, self.d_model)
        self.tgt_pos = nn.Embedding(self.max_k, self.d_model)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.token_scorer = nn.Linear(self.d_model, 1)
        self.score_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, 1),
        )

        self.use_prototypes = bool(use_prototypes)
        self.prototypes = PrototypeBank(self.d_model, num_prototypes=num_prototypes, tau=proto_tau) if self.use_prototypes else None

    def forward(self, tokens: torch.Tensor, masks: torch.Tensor) -> TargetFormerOutput:
        # tokens: (B,N,K,Din), masks: (B,N,K) with 1 valid.
        b, n, k, d = tokens.shape
        if n != self.num_segments:
            raise ValueError(f"num_segments mismatch: model={self.num_segments} got={n}")
        if k > self.max_k:
            raise ValueError(f"k too large: model_max_k={self.max_k} got={k}")
        if d != self.input_dim:
            raise ValueError(f"input_dim mismatch: model={self.input_dim} got={d}")

        x = self.input_proj(tokens)  # (B,N,K,D)
        seg_idx = torch.arange(n, device=x.device).view(1, n, 1)
        tgt_idx = torch.arange(k, device=x.device).view(1, 1, k)
        x = x + self.seg_pos(seg_idx) + self.tgt_pos(tgt_idx)
        x = self.drop(x)

        s = x.view(b, n * k, self.d_model)
        key_padding_mask = ~(masks.view(b, n * k).to(dtype=torch.bool))
        s = self.encoder(s, src_key_padding_mask=key_padding_mask)  # (B,S,D)
        h = s.view(b, n, k, self.d_model)

        token_logits = self.token_scorer(h).squeeze(-1)  # (B,N,K)
        token_weights = masked_softmax(token_logits, mask=masks, dim=-1)
        seg_emb = torch.sum(h * token_weights.unsqueeze(-1), dim=2)  # (B,N,D)

        proto_weights = None
        if self.prototypes is not None:
            proto = self.prototypes(seg_emb)
            seg_emb = seg_emb + proto["pattern"]
            proto_weights = proto["weights"]

        seg_logits = self.score_head(seg_emb).squeeze(-1)  # (B,N)
        seg_scores = torch.sigmoid(seg_logits)

        return TargetFormerOutput(
            segment_scores=seg_scores,
            segment_logits=seg_logits,
            segment_embeddings=seg_emb,
            token_weights=token_weights,
            proto_weights=proto_weights,
        )

    def prototype_diversity_loss(self) -> torch.Tensor:
        if self.prototypes is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.prototypes.diversity_loss()
