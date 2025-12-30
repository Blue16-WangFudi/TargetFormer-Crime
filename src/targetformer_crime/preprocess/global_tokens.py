from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from targetformer_crime.preprocess.yolo_tokens import _ResNet18Encoder, _to_device


def _full_frame_cxcywh_norm(*, w: int, h: int) -> np.ndarray:
    if w <= 1 or h <= 1:
        return np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32)
    x1, y1, x2, y2 = 0.0, 0.0, float(w - 1), float(h - 1)
    cx = (x1 + x2) / 2.0 / float(w)
    cy = (y1 + y2) / 2.0 / float(h)
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    return np.array([cx, cy, bw, bh], dtype=np.float32)


@dataclass
class GlobalTokenPreprocessor:
    fps: int
    num_segments: int
    max_k: int
    device: str
    half: bool
    cache_dtype: str
    max_frames_per_video: Optional[int] = None

    geom_dim: int = 4
    motion_dim: int = 8
    appearance_dim: int = 512

    def __post_init__(self) -> None:
        self.device = _to_device(self.device)
        self.half = bool(self.half and self.device.startswith("cuda"))
        self._encoder = _ResNet18Encoder(device=self.device)
        self._frames_index_cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    @property
    def raw_dim(self) -> int:
        return self.geom_dim + self.motion_dim + self.appearance_dim

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "GlobalTokenPreprocessor":
        return cls(
            fps=int(cfg.get("fps", 10)),
            num_segments=int(cfg.get("num_segments", 32)),
            max_k=int(cfg.get("max_k", 1)),
            device=str(cfg.get("device", "cuda")),
            half=bool(cfg.get("half", True)),
            cache_dtype=str(cfg.get("cache_dtype", "float16")),
            max_frames_per_video=int(cfg["max_frames_per_video"])
            if cfg.get("max_frames_per_video") is not None
            else None,
        )

    def process_frames(
        self,
        *,
        frames_dir: Path,
        prefix: str,
        ext: str = ".png",
        num_frames: Optional[int] = None,
        assumed_fps: float = 30.0,
        label: int = -1,
        category: str = "Unknown",
        uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        uid = uid or prefix
        frames_dir = Path(frames_dir)
        ext = ext if ext.startswith(".") else f".{ext}"
        ext_l = ext.lower()

        indices = self._get_frame_indices(frames_dir=frames_dir, prefix=prefix, ext_l=ext_l)
        if indices.size == 0:
            raise ValueError(f"empty frame sequence: {frames_dir} {prefix}{ext_l}")

        idx_min = int(indices[0])
        idx_max = int(indices[-1])
        num_available = int(indices.size)
        if num_frames is not None and int(num_frames) != num_available:
            num_frames = num_available
        else:
            num_frames = num_available

        assumed_fps = float(assumed_fps) if float(assumed_fps) > 0 else 30.0
        duration_s = float(idx_max - idx_min) / float(assumed_fps) if idx_max > idx_min else 0.0
        actual_fps = (float(num_available - 1) / duration_s) if (duration_s > 0 and num_available > 1) else 0.0

        stride = 1
        if self.fps and actual_fps > 0:
            stride = max(1, int(round(actual_fps / float(self.fps))))
        sampled_indices = indices[::stride]
        if self.max_frames_per_video:
            sampled_indices = sampled_indices[: int(self.max_frames_per_video)]

        frame_paths = [(int(i), frames_dir / f"{prefix}_{int(i)}{ext_l}") for i in sampled_indices]
        frame_paths = [(idx, p) for idx, p in frame_paths if p.exists()]
        if not frame_paths:
            raise ValueError(f"no sampled frames exist for {frames_dir}/{prefix}*{ext_l}")

        idx_to_path = {idx: p for idx, p in frame_paths}
        available = sorted(idx_to_path.keys())

        repr_paths: List[Path] = []
        for seg in range(self.num_segments):
            target = int(round(idx_min + (seg + 0.5) * (idx_max - idx_min) / max(1, self.num_segments)))
            nearest = min(available, key=lambda x: abs(x - target))
            repr_paths.append(idx_to_path[nearest])

        import cv2

        imgs_bgr: List[np.ndarray] = []
        sizes: List[Tuple[int, int]] = []
        for p in repr_paths:
            img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img_bgr is None:
                img_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
            h, w = img_bgr.shape[:2]
            imgs_bgr.append(img_bgr)
            sizes.append((w, h))

        feats = self._encoder.encode(crops_bgr=imgs_bgr, batch_size=64)
        if feats.shape[0] != self.num_segments:
            raise RuntimeError("unexpected embedding batch size")

        tokens = np.zeros((self.num_segments, self.max_k, self.raw_dim), dtype=np.float32)
        masks = np.zeros((self.num_segments, self.max_k), dtype=np.uint8)
        attn = np.zeros((self.num_segments, self.max_k), dtype=np.float32)
        track_ids = -np.ones((self.num_segments, self.max_k), dtype=np.int32)

        for seg_id in range(self.num_segments):
            w, h = sizes[seg_id]
            tokens[seg_id, 0, 0 : self.geom_dim] = _full_frame_cxcywh_norm(w=w, h=h)
            tokens[seg_id, 0, self.geom_dim : self.geom_dim + self.motion_dim] = 0.0
            tokens[seg_id, 0, self.geom_dim + self.motion_dim :] = feats[seg_id].astype(np.float32, copy=False)
            masks[seg_id, 0] = 1
            attn[seg_id, 0] = 0.0

        meta = {
            "uid": uid,
            "storage": "frames",
            "frames_dir": str(frames_dir),
            "frames_prefix": prefix,
            "frame_ext": ext_l,
            "assumed_fps": float(assumed_fps),
            "target_fps": int(self.fps),
            "stride": int(stride),
            "num_frames": int(num_frames),
            "num_sampled_frames": int(len(frame_paths)),
            "frame_idx_min": int(idx_min),
            "frame_idx_max": int(idx_max),
            "duration_seconds": float(duration_s),
            "estimated_source_fps": float(actual_fps),
            "num_segments": int(self.num_segments),
            "max_k": int(self.max_k),
            "raw_dim": int(self.raw_dim),
            "preprocess_kind": "global",
            "feature_layout": {
                "geometry": [0, self.geom_dim],
                "motion": [self.geom_dim, self.geom_dim + self.motion_dim],
                "appearance": [self.geom_dim + self.motion_dim, self.raw_dim],
            },
        }

        return {
            "tokens": tokens.astype(self.cache_dtype),
            "masks": masks.astype(np.uint8),
            "attn_weights": attn.astype(np.float32),
            "track_ids": track_ids.astype(np.int32),
            "label": np.array(label, dtype=np.int64),
            "meta_json": np.array(json.dumps(meta, ensure_ascii=False), dtype=object),
            "category": np.array(category, dtype=object),
        }

    def _get_frame_indices(self, *, frames_dir: Path, prefix: str, ext_l: str) -> np.ndarray:
        index = self._get_frames_index(frames_dir=frames_dir, ext_l=ext_l)
        arr = index.get(prefix)
        if arr is None:
            return np.array([], dtype=np.int32)
        return arr

    def _get_frames_index(self, *, frames_dir: Path, ext_l: str) -> Dict[str, np.ndarray]:
        key = (str(frames_dir.resolve()), ext_l)
        cached = self._frames_index_cache.get(key)
        if cached is not None:
            return cached

        mapping: Dict[str, List[int]] = {}
        with os.scandir(frames_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                name = entry.name
                if not name.lower().endswith(ext_l):
                    continue
                stem = Path(name).stem
                if "_" not in stem:
                    continue
                pref, idx_str = stem.rsplit("_", 1)
                if not idx_str.isdigit():
                    continue
                mapping.setdefault(pref, []).append(int(idx_str))

        out: Dict[str, np.ndarray] = {}
        for k, v in mapping.items():
            arr = np.asarray(v, dtype=np.int32)
            arr.sort()
            out[k] = arr

        self._frames_index_cache[key] = out
        return out
