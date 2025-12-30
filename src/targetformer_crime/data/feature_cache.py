from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CacheItem:
    uid: str
    split: str
    path: Path
    label: int
    category: str


def list_cache_items(feature_cache_dir: Path, split: str) -> List[CacheItem]:
    feature_cache_dir = Path(feature_cache_dir)
    split_dir = feature_cache_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"feature cache not found: {split_dir}")

    items: List[CacheItem] = []
    for p in sorted(split_dir.glob("*.npz")):
        uid = p.stem
        with np.load(p, allow_pickle=True) as d:
            label = int(np.array(d["label"]).item()) if "label" in d else -1
            category = str(np.array(d.get("category", "Unknown")).item())
        items.append(CacheItem(uid=uid, split=split, path=p, label=label, category=category))
    return items


def load_npz_item(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as d:
        tokens = d["tokens"]
        masks = d["masks"]
        attn = d["attn_weights"] if "attn_weights" in d else None
        track_ids = d["track_ids"] if "track_ids" in d else None
        label = int(np.array(d["label"]).item()) if "label" in d else -1        
        category = str(np.array(d.get("category", "Unknown")).item())
        meta = {}
        if "meta_json" in d:
            meta = json.loads(str(np.array(d["meta_json"]).item()))
        return {
            "tokens": tokens,
            "masks": masks,
            "attn_weights": attn,
            "track_ids": track_ids,
            "label": label,
            "category": category,
            "meta": meta,
        }


def feature_layout_from_cache_item(path: Path) -> Dict[str, Tuple[int, int]]:
    item = load_npz_item(path)
    layout = (item.get("meta") or {}).get("feature_layout") or {}
    out: Dict[str, Tuple[int, int]] = {}
    for k in ["geometry", "motion", "appearance"]:
        if k in layout and isinstance(layout[k], list) and len(layout[k]) == 2:
            out[k] = (int(layout[k][0]), int(layout[k][1]))
    return out
