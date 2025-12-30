from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from targetformer_crime.config import get_paths
from targetformer_crime.data import discover_ucf_crime
from targetformer_crime.preprocess.global_tokens import GlobalTokenPreprocessor
from targetformer_crime.preprocess.yolo_tokens import YoloTokenPreprocessor
from targetformer_crime.utils import dump_json, ensure_dir


def run_preprocess(cfg: Dict[str, Any], repo_root: Path) -> None:
    paths = get_paths(cfg, repo_root=repo_root)
    ds_cfg = cfg.get("dataset", {})
    base_pp_cfg = cfg.get("preprocess", {}) or {}

    index = discover_ucf_crime(
        paths.datasets_root,
        train_dirname=ds_cfg.get("train_dirname", "Train"),
        test_dirname=ds_cfg.get("test_dirname", "Test"),
        normal_dirname=ds_cfg.get("normal_dirname", "NormalVideos"),
        video_exts=ds_cfg.get("video_exts"),
        image_exts=ds_cfg.get("image_exts"),
        max_videos=base_pp_cfg.get("max_videos"),
    )
    records = list(index.records)

    variants = cfg.get("preprocess_variants") or []
    if not variants:
        variants = [{"name": "default", **base_pp_cfg}]

    for var in variants:
        pp_cfg = _merge_dicts(base_pp_cfg, var)
        _run_single_preprocess(
            repo_root=repo_root,
            ds_cfg=ds_cfg,
            pp_cfg=pp_cfg,
            records=records,
        )


def _run_single_preprocess(
    *, repo_root: Path, ds_cfg: Dict[str, Any], pp_cfg: Dict[str, Any], records: List[Any]
) -> None:
    out_dir = ensure_dir(repo_root / Path(pp_cfg.get("out_dir", "outputs/precomputed")))
    meta_path = out_dir / "preprocess_meta.json"
    pp_kind = str(pp_cfg.get("kind") or pp_cfg.get("preprocess_kind") or "yolo").lower()
    if pp_kind in {"global", "global_tokens"}:
        pre = GlobalTokenPreprocessor.from_config(pp_cfg)
    else:
        pre = YoloTokenPreprocessor.from_config(pp_cfg)
    dump_json(meta_path, {"config": pp_cfg})

    assumed_fps = float(ds_cfg.get("assumed_fps", 30.0))
    name = pp_cfg.get("name") or pp_cfg.get("tag") or out_dir.name

    for r in tqdm(records, desc=f"Preprocess[{name}]"):
        split_dir = out_dir / r.split
        ensure_dir(split_dir)
        out_path = split_dir / f"{r.uid}.npz"
        if pp_cfg.get("resume", True) and out_path.exists():
            continue
        try:
            if r.storage == "frames":
                pack = pre.process_frames(
                    frames_dir=r.frames_dir,
                    prefix=r.frames_prefix or r.uid,
                    ext=r.frame_ext or ".png",
                    num_frames=r.num_frames,
                    assumed_fps=assumed_fps,
                    label=r.label,
                    category=r.category,
                    uid=r.uid,
                )
            else:
                raise NotImplementedError("video-file preprocessing not implemented in this workspace")
        except Exception as e:
            tqdm.write(f"[WARN] preprocess failed: {r.key()} ({e})")
            continue
        np.savez_compressed(out_path, **pack)


def _merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out
