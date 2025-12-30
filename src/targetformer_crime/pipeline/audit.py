from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from targetformer_crime.config import get_paths
from targetformer_crime.data import discover_ucf_crime
from targetformer_crime.utils import dump_json, ensure_dir
from targetformer_crime.viz.dataset_audit import plot_dataset_audit


def _probe_video(video_path: Path) -> Dict[str, Any]:
    # Prefer container-level parsing (fast, no full decode).
    try:
        import av  # type: ignore

        container = av.open(str(video_path))
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise RuntimeError("no video stream")

        fps = float(stream.average_rate) if stream.average_rate else None
        width = int(getattr(stream, "width", 0) or 0) or None
        height = int(getattr(stream, "height", 0) or 0) or None

        duration = None
        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            # PyAV uses microseconds for container.duration
            duration = float(container.duration) / 1_000_000.0

        num_frames = int(getattr(stream, "frames", 0) or 0) or None
        if num_frames is None and duration is not None and fps is not None:
            num_frames = int(round(duration * fps))

        try:
            container.close()
        except Exception:
            pass

        return {
            "duration": duration,
            "fps": fps,
            "num_frames": num_frames,
            "width": width,
            "height": height,
        }
    except Exception:
        pass

    # Fallback: OpenCV metadata (may be slower for some codecs).
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("opencv cannot open")
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or None
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or None
        cap.release()
        return {"duration": None, "fps": fps, "num_frames": None, "width": width, "height": height}
    except Exception:
        return {"duration": None, "fps": None, "num_frames": None, "width": None, "height": None}


def run_audit(cfg: Dict[str, Any], repo_root: Path) -> None:
    paths = get_paths(cfg, repo_root=repo_root)
    ds_cfg = cfg.get("dataset", {})
    audit_cfg = cfg.get("audit", {})

    out_dir = ensure_dir(repo_root / Path(audit_cfg.get("out_dir", "outputs/dataset_audit")))
    manifest_path = out_dir / audit_cfg.get("manifest_name", "manifest.csv")
    env_path = out_dir / "env.json"

    index = discover_ucf_crime(
        paths.datasets_root,
        train_dirname=ds_cfg.get("train_dirname", "Train"),
        test_dirname=ds_cfg.get("test_dirname", "Test"),
        normal_dirname=ds_cfg.get("normal_dirname", "NormalVideos"),
        video_exts=ds_cfg.get("video_exts"),
        image_exts=ds_cfg.get("image_exts"),
        max_videos=audit_cfg.get("max_videos"),
    )

    records = list(index.records)

    rows: List[Dict[str, Any]] = []
    assumed_fps = float(ds_cfg.get("assumed_fps", 30.0))
    for r in tqdm(records, desc="Audit videos"):
        if r.storage == "video":
            assert r.video_path is not None
            meta = _probe_video(r.video_path)
            rows.append(
                {
                    "split": r.split,
                    "storage": r.storage,
                    "uid": r.uid,
                    "video_path": str(r.video_path),
                    "label": "abnormal" if r.label == 1 else "normal",
                    "category": r.category,
                    **meta,
                }
            )
        else:
            meta = _probe_frames(
                frames_dir=r.frames_dir,
                prefix=r.frames_prefix,
                ext=r.frame_ext,
                num_frames=r.num_frames,
                frame_idx_min=r.frame_idx_min,
                frame_idx_max=r.frame_idx_max,
                sample_path=r.sample_path,
                assumed_fps=assumed_fps,
            )
            rows.append(
                {
                    "split": r.split,
                    "storage": r.storage,
                    "uid": r.uid,
                    "video_path": str((r.frames_dir or Path(".")) / (r.frames_prefix or r.uid)),
                    "frames_dir": str(r.frames_dir) if r.frames_dir else None,
                    "frames_prefix": r.frames_prefix,
                    "frame_ext": r.frame_ext,
                    "label": "abnormal" if r.label == 1 else "normal",
                    "category": r.category,
                    **meta,
                }
            )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    dump_json(env_path, _collect_env_info())

    # figures + underlying data
    plot_dataset_audit(
        manifest_csv=manifest_path,
        figures_dir=repo_root / "outputs" / "figures",
        figures_data_dir=repo_root / "outputs" / "figures_data",
        prefix="dataset_audit",
    )


def _probe_frames(
    frames_dir: Path | None,
    prefix: str | None,
    ext: str | None,
    num_frames: int | None,
    frame_idx_min: int | None,
    frame_idx_max: int | None,
    sample_path: Path | None,
    assumed_fps: float,
) -> Dict[str, Any]:
    if frames_dir is None or prefix is None:
        return {"duration": None, "fps": None, "num_frames": None, "width": None, "height": None}

    if ext is None:
        ext = ".png"

    if num_frames is None:
        num_frames = _count_frames_for_prefix(frames_dir=frames_dir, prefix=prefix, ext=ext)

    width = None
    height = None
    try:
        import cv2

        img_path = sample_path
        if img_path is None:
            cand = frames_dir / f"{prefix}_0{ext}"
            img_path = cand if cand.exists() else None
        if img_path is not None and img_path.exists():
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                height, width = img.shape[:2]
    except Exception:
        pass

    fps = assumed_fps if assumed_fps > 0 else None
    duration = None
    if fps is not None:
        if (
            frame_idx_min is not None
            and frame_idx_max is not None
            and int(frame_idx_max) > int(frame_idx_min)
        ):
            duration = float(int(frame_idx_max) - int(frame_idx_min)) / float(fps)
        elif num_frames is not None:
            duration = float(num_frames) / float(fps)

    estimated_source_fps = None
    if duration is not None and duration > 0 and num_frames is not None and num_frames > 1:
        estimated_source_fps = float(num_frames - 1) / float(duration)

    return {
        "duration": duration,
        "fps": fps,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "frame_idx_min": frame_idx_min,
        "frame_idx_max": frame_idx_max,
        "estimated_source_fps": estimated_source_fps,
    }


def _count_frames_for_prefix(frames_dir: Path, prefix: str, ext: str) -> int:
    import os

    n = 0
    prefix_head = f"{prefix}_"
    with os.scandir(frames_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            name = entry.name
            if not name.startswith(prefix_head):
                continue
            if not name.lower().endswith(ext.lower()):
                continue
            n += 1
    return n


def _collect_env_info() -> Dict[str, Any]:
    import platform
    import subprocess

    import torch

    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update(
            {
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu_name": torch.cuda.get_device_name(0),
            }
        )
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        info["nvidia_driver"] = out.decode("utf-8").strip()
    except Exception:
        info["nvidia_driver"] = None

    return info
