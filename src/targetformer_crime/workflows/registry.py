from __future__ import annotations

from typing import Any, Dict, Tuple

from torch import nn

from targetformer_crime.models.baselines import GlobalMLP, YoloGRU
from targetformer_crime.models.targetformer import TargetFormer


def build_model(
    kind: str,
    input_dim: int,
    num_segments: int,
    max_k: int,
    model_cfg: Dict[str, Any],
) -> nn.Module:
    kind = str(kind).lower()
    if kind in {"targetformer", "yolo_no_track"}:
        return TargetFormer(
            input_dim=input_dim,
            num_segments=num_segments,
            max_k=max_k,
            d_model=int(model_cfg.get("d_model", 256)),
            nhead=int(model_cfg.get("nhead", 8)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_prototypes=bool(model_cfg.get("use_prototypes", True)),
            num_prototypes=int(model_cfg.get("num_prototypes", 32)),
            proto_tau=float(model_cfg.get("proto_tau", 0.2)),
        )
    if kind == "global_mlp":
        return GlobalMLP(
            input_dim=input_dim,
            num_segments=num_segments,
            max_k=max_k,
            d_model=int(model_cfg.get("d_model", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if kind in {"yolo_gru", "gru", "lstm"}:
        return YoloGRU(
            input_dim=input_dim,
            num_segments=num_segments,
            max_k=max_k,
            d_model=int(model_cfg.get("d_model", 256)),
            hidden_size=int(model_cfg.get("hidden_size", model_cfg.get("d_model", 256))),
            num_layers=int(model_cfg.get("num_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"unknown model kind: {kind}")

