from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover
    YOLO = None  # type: ignore


def _to_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _clip_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.astype(float).tolist()
    x1 = max(0.0, min(x1, w - 1.0))
    y1 = max(0.0, min(y1, h - 1.0))
    x2 = max(0.0, min(x2, w - 1.0))
    y2 = max(0.0, min(y2, h - 1.0))
    if x2 <= x1:
        x2 = min(w - 1.0, x1 + 1.0)
    if y2 <= y1:
        y2 = min(h - 1.0, y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _xyxy_to_cxcywh_norm(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:       
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    cx = (x1 + x2) / 2.0 / float(w)
    cy = (y1 + y2) / 2.0 / float(h)
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    return np.array([cx, cy, bw, bh], dtype=np.float32)


def _bbox_iou(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a_xyxy.tolist()]
    bx1, by1, bx2, by2 = [float(x) for x in b_xyxy.tolist()]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _crop_bgr(img_bgr: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_xyxy(xyxy, w=w, h=h).astype(int)
    return img_bgr[y1:y2, x1:x2].copy()


def _segment_sample_indices(
    *,
    indices: np.ndarray,
    idx_min: int,
    idx_max: int,
    num_segments: int,
    frames_per_segment: int,
) -> np.ndarray:
    if indices.size == 0:
        return np.array([], dtype=np.int32)
    frames_per_segment = int(frames_per_segment)
    if frames_per_segment <= 0:
        return indices.astype(np.int32, copy=False)

    picked: List[int] = []
    span = float(idx_max - idx_min)
    for seg in range(int(num_segments)):
        seg_start = idx_min + (float(seg) / float(num_segments)) * span
        seg_end = idx_min + (float(seg + 1) / float(num_segments)) * span
        cand = indices[(indices >= seg_start) & (indices <= seg_end)]
        if cand.size == 0:
            target = int(round(idx_min + (seg + 0.5) * span / max(1.0, float(num_segments))))
            nearest = int(indices[int(np.argmin(np.abs(indices - target)))])
            picked.append(nearest)
            continue
        if cand.size <= frames_per_segment:
            picked.extend([int(x) for x in cand.tolist()])
            continue

        pos = np.linspace(0, cand.size - 1, frames_per_segment)
        sel = cand[np.round(pos).astype(int)]
        picked.extend([int(x) for x in sel.tolist()])

    uniq = np.asarray(sorted(set(picked)), dtype=np.int32)
    return uniq


class _ResNet18Encoder:
    def __init__(self, device: str) -> None:
        # Torchvision is preferred (pretrained ResNet18); fall back to a tiny CNN
        # when torch/torchvision binaries are mismatched in the runtime.
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        try:
            from torchvision.models import ResNet18_Weights, resnet18  # type: ignore

            weights = None
            try:
                weights = ResNet18_Weights.DEFAULT
            except Exception:
                weights = None

            model = resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            model.eval()
            self.model = model.to(device)
            self.out_dim = 512
            self.kind = "resnet18"
            return
        except Exception:
            pass

        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 512),
        )
        model.eval()
        self.model = model.to(device)
        self.out_dim = 512
        self.kind = "tinycnn512"

    @torch.no_grad()
    def encode(self, crops_bgr: List[np.ndarray], batch_size: int = 64) -> np.ndarray:
        import cv2

        if not crops_bgr:
            return np.zeros((0, 512), dtype=np.float32)

        feats: List[np.ndarray] = []
        for i in range(0, len(crops_bgr), batch_size):
            batch = crops_bgr[i : i + batch_size]
            imgs = []
            for img_bgr in batch:
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
                t = torch.from_numpy(img).to(self.device).float() / 255.0
                t = t.permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
                imgs.append(t)
            x = torch.cat(imgs, dim=0)
            x = (x - self.mean) / self.std
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    y = self.model(x)
            else:
                y = self.model(x)
            feats.append(y.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(feats, axis=0)


@dataclass
class _TrackAccum:
    times: List[float]
    centers_norm: List[Tuple[float, float]]
    bboxes_xyxy: List[np.ndarray]
    confs: List[float]
    best_conf: float
    best_frame_path: Optional[Path]
    best_bbox_xyxy: Optional[np.ndarray]


@dataclass
class YoloTokenPreprocessor:
    fps: int
    num_segments: int
    max_k: int
    use_tracking: bool
    yolo_model: str
    yolo_imgsz: int
    yolo_conf: float
    yolo_iou: float
    yolo_classes: Optional[List[int]]
    tracker: str
    device: str
    half: bool
    cache_dtype: str
    max_frames_per_video: Optional[int] = None
    tracking_frames_per_segment: Optional[int] = None
    fallback_full_frame: bool = True

    geom_dim: int = 4
    motion_dim: int = 8
    appearance_dim: int = 512

    def __post_init__(self) -> None:
        if YOLO is None:  # pragma: no cover
            raise RuntimeError("ultralytics is not available; install ultralytics first.")
        self.device = _to_device(self.device)
        self.half = bool(self.half and self.device.startswith("cuda"))
        self._yolo = YOLO(self.yolo_model)
        self._encoder = _ResNet18Encoder(device=self.device)
        self._frames_index_cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "YoloTokenPreprocessor":
        return cls(
            fps=int(cfg.get("fps", 10)),
            num_segments=int(cfg.get("num_segments", 32)),
            max_k=int(cfg.get("max_k", 20)),
            use_tracking=bool(cfg.get("use_tracking", True)),
            yolo_model=str(cfg.get("yolo_model", "yolov8n.pt")),
            yolo_imgsz=int(cfg.get("yolo_imgsz", 640)),
            yolo_conf=float(cfg.get("yolo_conf", 0.25)),
            yolo_iou=float(cfg.get("yolo_iou", 0.7)),
            yolo_classes=list(cfg.get("yolo_classes", [0])) if cfg.get("yolo_classes") is not None else None,
            tracker=str(cfg.get("tracker", "bytetrack.yaml")),
            device=str(cfg.get("device", "cuda")),
            half=bool(cfg.get("half", True)),
            cache_dtype=str(cfg.get("cache_dtype", "float16")),
            max_frames_per_video=cfg.get("max_frames_per_video"),
            tracking_frames_per_segment=(
                int(cfg["tracking_frames_per_segment"])
                if cfg.get("tracking_frames_per_segment") is not None
                else None
            ),
            fallback_full_frame=bool(cfg.get("fallback_full_frame", True)),
        )

    @property
    def raw_dim(self) -> int:
        return self.geom_dim + self.motion_dim + self.appearance_dim

    def process_frames(
        self,
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

        # Downsample indices to approximately match target fps (never upsample).
        stride = 1
        if self.fps and actual_fps > 0:
            stride = max(1, int(round(actual_fps / float(self.fps))))
        segment_sampling = bool(
            self.use_tracking
            and self.tracking_frames_per_segment is not None
            and int(self.tracking_frames_per_segment) > 0
        )
        if segment_sampling:
            sampled_indices = _segment_sample_indices(
                indices=indices,
                idx_min=idx_min,
                idx_max=idx_max,
                num_segments=self.num_segments,
                frames_per_segment=int(self.tracking_frames_per_segment or 0),
            )
        else:
            sampled_indices = indices[::stride]
        if self.max_frames_per_video:
            sampled_indices = sampled_indices[: int(self.max_frames_per_video)]

        frame_paths = [(int(i), frames_dir / f"{prefix}_{int(i)}{ext_l}") for i in sampled_indices]
        frame_paths = [(idx, p) for idx, p in frame_paths if p.exists()]
        if not frame_paths:
            raise ValueError(f"no sampled frames exist for {frames_dir}/{prefix}*{ext_l}")

        if self.use_tracking:
            tokens, masks, attn, track_ids = self._tokens_with_tracking(
                frame_paths, assumed_fps=assumed_fps, idx_min=idx_min, idx_max=idx_max
            )
        else:
            tokens, masks, attn, track_ids = self._tokens_no_tracking(frame_paths, idx_min=idx_min, idx_max=idx_max)

        meta = {
            "uid": uid,
            "storage": "frames",
            "frames_dir": str(frames_dir),
            "frames_prefix": prefix,
            "frame_ext": ext_l,
            "assumed_fps": float(assumed_fps),
            "target_fps": int(self.fps),
            "stride": int(stride),
            "tracking_frames_per_segment": int(self.tracking_frames_per_segment)
            if self.tracking_frames_per_segment is not None
            else None,
            "segment_sampling": bool(segment_sampling),
            "num_frames": int(num_frames),
            "num_sampled_frames": int(len(frame_paths)),
            "frame_idx_min": int(idx_min),
            "frame_idx_max": int(idx_max),
            "duration_seconds": float(duration_s),
            "estimated_source_fps": float(actual_fps),
            "num_segments": int(self.num_segments),
            "max_k": int(self.max_k),
            "raw_dim": int(self.raw_dim),
            "fallback_full_frame": bool(self.fallback_full_frame),
            "use_tracking": bool(self.use_tracking),
            "preprocess_kind": "yolo_track" if self.use_tracking else "yolo_no_track",
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

    def _get_frame_indices(self, frames_dir: Path, prefix: str, ext_l: str) -> np.ndarray:
        index = self._get_frames_index(frames_dir=frames_dir, ext_l=ext_l)
        arr = index.get(prefix)
        if arr is None:
            return np.array([], dtype=np.int32)
        return arr

    def _get_frames_index(self, frames_dir: Path, ext_l: str) -> Dict[str, np.ndarray]:
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
                prefix, idx_str = stem.rsplit("_", 1)
                if not idx_str.isdigit():
                    continue
                mapping.setdefault(prefix, []).append(int(idx_str))

        out: Dict[str, np.ndarray] = {}
        for k, v in mapping.items():
            arr = np.asarray(v, dtype=np.int32)
            arr.sort()
            out[k] = arr

        self._frames_index_cache[key] = out
        return out

    def _tokens_no_tracking(
        self,
        frame_paths: List[Tuple[int, Path]],
        idx_min: int,
        idx_max: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # One representative frame per segment.
        idx_to_path = {idx: p for idx, p in frame_paths}
        available = sorted(idx_to_path.keys())
        if not available:
            raise ValueError("no frames")

        repr_paths: List[Path] = []
        repr_indices: List[int] = []
        for seg in range(self.num_segments):
            target = int(round(idx_min + (seg + 0.5) * (idx_max - idx_min) / max(1, self.num_segments)))
            # nearest available sampled frame
            nearest = min(available, key=lambda x: abs(x - target))
            repr_indices.append(nearest)
            repr_paths.append(idx_to_path[nearest])

        results = self._yolo.predict(
            source=[str(p) for p in repr_paths],
            stream=True,
            imgsz=self.yolo_imgsz,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            classes=self.yolo_classes,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        tokens = np.zeros((self.num_segments, self.max_k, self.raw_dim), dtype=np.float32)
        masks = np.zeros((self.num_segments, self.max_k), dtype=np.uint8)
        attn = np.zeros((self.num_segments, self.max_k), dtype=np.float32)  # placeholder for viz
        track_ids = -np.ones((self.num_segments, self.max_k), dtype=np.int32)

        crop_tasks: List[Tuple[int, int, np.ndarray]] = []
        for seg_id, r in enumerate(results):
            img_bgr = getattr(r, "orig_img", None)
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                if self.fallback_full_frame:
                    bb = np.array([0.0, 0.0, float(w - 1), float(h - 1)], dtype=np.float32)
                    tokens[seg_id, 0, 0 : self.geom_dim] = _xyxy_to_cxcywh_norm(bb, w=w, h=h)
                    tokens[seg_id, 0, self.geom_dim : self.geom_dim + self.motion_dim] = 0.0
                    masks[seg_id, 0] = 1
                    attn[seg_id, 0] = 0.0
                    track_ids[seg_id, 0] = -1
                    crop_tasks.append((seg_id, 0, img_bgr))
                continue

            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            order = np.argsort(-conf)
            xyxy = xyxy[order]
            conf = conf[order]
            k = min(self.max_k, xyxy.shape[0])
            for j in range(k):
                bb = _clip_xyxy(xyxy[j], w=w, h=h)
                tokens[seg_id, j, 0 : self.geom_dim] = _xyxy_to_cxcywh_norm(bb, w=w, h=h)
                # motion zeros
                tokens[seg_id, j, self.geom_dim : self.geom_dim + self.motion_dim] = 0.0
                masks[seg_id, j] = 1
                attn[seg_id, j] = float(conf[j])
                track_ids[seg_id, j] = int(j)
                crop_tasks.append((seg_id, j, _crop_bgr(img_bgr, bb)))

        # appearance embeddings
        if crop_tasks:
            crops = [c for _, _, c in crop_tasks]
            feats = self._encoder.encode(crops_bgr=crops, batch_size=64)
            for (seg_id, j, _), emb in zip(crop_tasks, feats, strict=False):
                tokens[seg_id, j, self.geom_dim + self.motion_dim :] = emb

        return tokens, masks, attn, track_ids

    def _tokens_with_tracking(
        self,
        frame_paths: List[Tuple[int, Path]],
        assumed_fps: float,
        idx_min: int,
        idx_max: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Track across sampled frames using Ultralytics ByteTrack/BoT-SORT.
        if assumed_fps <= 0:
            raise ValueError("assumed_fps must be >0 for tracking mode")

        duration = float(idx_max - idx_min) / float(assumed_fps) if idx_max > idx_min else 0.0
        seg_len = duration / float(self.num_segments)

        seg_tracks: List[Dict[int, _TrackAccum]] = [dict() for _ in range(self.num_segments)]

        # Robust, deterministic IoU tracking (avoids Windows OpenMP/lap issues and occasional hangs).
        seg_last_bbox: List[Dict[int, np.ndarray]] = [dict() for _ in range(self.num_segments)]
        seg_next_id = [0 for _ in range(self.num_segments)]

        results = self._yolo.predict(
            source=[str(p) for _, p in frame_paths],
            stream=True,
            imgsz=self.yolo_imgsz,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            classes=self.yolo_classes,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        iou_thresh = 0.3
        for (frame_idx, frame_path), r in zip(frame_paths, results, strict=False):
            img_bgr = getattr(r, "orig_img", None)
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            t = float(frame_idx - idx_min) / float(assumed_fps)
            seg_id = int(min(self.num_segments - 1, max(0, math.floor(t / seg_len)))) if seg_len > 0 else 0

            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            order = np.argsort(-conf)
            xyxy = xyxy[order]
            conf = conf[order]

            used: set[int] = set()
            for bb, c in zip(xyxy, conf, strict=False):
                bb = _clip_xyxy(bb, w=w, h=h)

                best_tid: Optional[int] = None
                best_iou = 0.0
                for tid, last_bb in seg_last_bbox[seg_id].items():
                    if tid in used:
                        continue
                    iou = _bbox_iou(bb, last_bb)
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = int(tid)

                if best_tid is None or best_iou < iou_thresh:
                    tid = int(seg_next_id[seg_id])
                    seg_next_id[seg_id] += 1
                else:
                    tid = int(best_tid)
                    used.add(tid)

                seg_last_bbox[seg_id][tid] = bb
                cxcywh = _xyxy_to_cxcywh_norm(bb, w=w, h=h)
                acc = seg_tracks[seg_id].get(tid)
                if acc is None:
                    acc = _TrackAccum(
                        times=[],
                        centers_norm=[],
                        bboxes_xyxy=[],
                        confs=[],
                        best_conf=-1.0,
                        best_frame_path=None,
                        best_bbox_xyxy=None,
                    )
                    seg_tracks[seg_id][tid] = acc
                acc.times.append(t)
                acc.centers_norm.append((float(cxcywh[0]), float(cxcywh[1])))
                acc.bboxes_xyxy.append(bb)
                acc.confs.append(float(c))
                if float(c) > acc.best_conf:
                    acc.best_conf = float(c)
                    acc.best_frame_path = frame_path
                    acc.best_bbox_xyxy = bb

        tokens = np.zeros((self.num_segments, self.max_k, self.raw_dim), dtype=np.float32)
        masks = np.zeros((self.num_segments, self.max_k), dtype=np.uint8)
        attn = np.zeros((self.num_segments, self.max_k), dtype=np.float32)
        track_ids = -np.ones((self.num_segments, self.max_k), dtype=np.int32)

        crop_tasks: List[Tuple[int, int, Path, np.ndarray]] = []

        for seg_id in range(self.num_segments):
            tracks = list(seg_tracks[seg_id].items())
            if not tracks:
                if self.fallback_full_frame:
                    # Use the nearest frame to segment center as a pseudo target.
                    target_t = (seg_id + 0.5) * seg_len
                    nearest = min(
                        frame_paths, key=lambda x: abs((float(x[0] - idx_min) / float(assumed_fps)) - target_t)
                    )
                    img_path = nearest[1]
                    import cv2

                    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        continue
                    h, w = img_bgr.shape[:2]
                    bb = np.array([0.0, 0.0, float(w - 1), float(h - 1)], dtype=np.float32)
                    tokens[seg_id, 0, 0 : self.geom_dim] = _xyxy_to_cxcywh_norm(bb, w=w, h=h)
                    tokens[seg_id, 0, self.geom_dim : self.geom_dim + self.motion_dim] = 0.0
                    masks[seg_id, 0] = 1
                    attn[seg_id, 0] = 0.0
                    track_ids[seg_id, 0] = -1
                    crop_tasks.append((seg_id, 0, img_path, bb))
                continue
            # rank by length then mean confidence
            tracks.sort(key=lambda kv: (len(kv[1].times), float(np.mean(kv[1].confs))), reverse=True)
            k = min(self.max_k, len(tracks))
            for j in range(k):
                tid, acc = tracks[j]
                # geometry: mean bbox over frames in segment
                bb = np.stack(acc.bboxes_xyxy, axis=0).mean(axis=0)
                # infer resolution from best frame
                import cv2

                img_path = acc.best_frame_path
                if img_path is None or not img_path.exists():
                    continue
                img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                h, w = img_bgr.shape[:2]
                bb = _clip_xyxy(bb, w=w, h=h)
                tokens[seg_id, j, 0 : self.geom_dim] = _xyxy_to_cxcywh_norm(bb, w=w, h=h)
                tokens[seg_id, j, self.geom_dim : self.geom_dim + self.motion_dim] = _motion_stats(
                    times=acc.times, centers=acc.centers_norm
                )
                masks[seg_id, j] = 1
                attn[seg_id, j] = float(np.mean(acc.confs))
                track_ids[seg_id, j] = int(tid)

                crop_bb = acc.best_bbox_xyxy if acc.best_bbox_xyxy is not None else bb
                crop_tasks.append((seg_id, j, img_path, crop_bb))

        # appearance embeddings (load/crop)
        if crop_tasks:
            import cv2

            crops = []
            for _, _, img_path, bb in crop_tasks:
                img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    crops.append(np.zeros((224, 224, 3), dtype=np.uint8))
                    continue
                crops.append(_crop_bgr(img_bgr, bb))
            feats = self._encoder.encode(crops_bgr=crops, batch_size=64)
            for (seg_id, j, _, _), emb in zip(crop_tasks, feats, strict=False):
                tokens[seg_id, j, self.geom_dim + self.motion_dim :] = emb

        return tokens, masks, attn, track_ids


def _motion_stats(times: List[float], centers: List[Tuple[float, float]]) -> np.ndarray:
    # centers are normalized (cx, cy) in [0,1]; times in seconds.
    if len(times) < 2:
        return np.zeros((8,), dtype=np.float32)
    order = np.argsort(np.array(times))
    t = np.array([times[i] for i in order], dtype=np.float32)
    c = np.array([centers[i] for i in order], dtype=np.float32)  # T x 2

    dt = np.diff(t)
    dp = np.diff(c, axis=0)
    valid = dt > 1e-6
    if not np.any(valid):
        return np.zeros((8,), dtype=np.float32)
    dt = dt[valid]
    dp = dp[valid]

    v = dp / dt[:, None]  # (T-1) x 2
    speed = np.linalg.norm(v, axis=1)
    dv = np.diff(v, axis=0)
    dt2 = dt[1:] if len(dt) > 1 else np.array([], dtype=np.float32)
    acc = np.linalg.norm(dv / dt2[:, None], axis=1) if len(dt2) > 0 else np.array([0.0], dtype=np.float32)

    vx, vy = v[:, 0], v[:, 1]
    out = np.array(
        [
            float(vx.mean()),
            float(vy.mean()),
            float(vx.std()),
            float(vy.std()),
            float(speed.mean()),
            float(speed.std()),
            float(acc.mean()),
            float(acc.std()),
        ],
        dtype=np.float32,
    )
    return out
