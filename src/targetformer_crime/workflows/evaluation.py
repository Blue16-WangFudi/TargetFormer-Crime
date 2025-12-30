from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from targetformer_crime.data.feature_cache import CacheItem, list_cache_items, load_npz_item
from targetformer_crime.metrics.classification import average_precision, safe_roc_auc
from targetformer_crime.utils import dump_json, ensure_dir
from targetformer_crime.workflows.registry import build_model


def eval_latest_experiment(*, cfg: Dict[str, Any], repo_root: Path) -> None:
    outputs_root = Path((cfg.get("paths", {}) or {}).get("outputs_root", "outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (repo_root / outputs_root).resolve()

    exp_dir = _find_latest_exp(outputs_root)
    if exp_dir is None:
        raise FileNotFoundError(f"no experiments found under {outputs_root}")

    run_dirs = sorted([p for p in exp_dir.glob("*") if p.is_dir()])
    for run_dir in run_dirs:
        seed_dirs = sorted([p for p in run_dir.glob("seed_*") if p.is_dir()])
        for sd in seed_dirs:
            _eval_one_seed_dir(cfg=cfg, repo_root=repo_root, seed_dir=sd)


def _eval_one_seed_dir(*, cfg: Dict[str, Any], repo_root: Path, seed_dir: Path) -> None:
    config_path = seed_dir / "config.yaml"
    if not config_path.exists():
        return
    run_cfg = _load_yaml(config_path)

    ckpt_path = seed_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        ckpt_path = seed_dir / "checkpoint_last.pt"
    if not ckpt_path.exists():
        return

    feature_cache = Path(run_cfg["feature_cache"])
    test_items = list_cache_items(feature_cache, split=str(run_cfg.get("eval_split", "Test")))

    sample = load_npz_item((feature_cache / "Train").glob("*.npz").__iter__().__next__())
    raw_dim = int(sample["tokens"].shape[-1])
    num_segments = int(sample["tokens"].shape[0])
    max_k = int(sample["tokens"].shape[1])
    k = int(run_cfg.get("k", min(10, max_k)))

    ablation = run_cfg.get("ablation", {}) or {}
    drop_geometry = bool(ablation.get("drop_geometry", False))
    drop_motion = bool(ablation.get("drop_motion", False))
    drop_appearance = bool(ablation.get("drop_appearance", False))
    layout = (sample.get("meta") or {}).get("feature_layout") or {}
    geom_slice = tuple(layout.get("geometry", [0, 0]))
    mot_slice = tuple(layout.get("motion", [0, 0]))
    app_slice = tuple(layout.get("appearance", [0, 0]))

    model_cfg = run_cfg.get("model", {}) or {}
    model = build_model(kind=run_cfg["kind"], input_dim=raw_dim, num_segments=num_segments, max_k=max_k, model_cfg=model_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true: List[int] = []
    y_score: List[float] = []
    seg_scores: List[np.ndarray] = []
    uids: List[str] = []

    for it in test_items:
        pack = load_npz_item(it.path)
        tokens = pack["tokens"][:, :k, :].astype(np.float32, copy=True)
        masks = pack["masks"][:, :k].astype(np.uint8, copy=False)
        if drop_geometry and geom_slice[1] > geom_slice[0]:
            tokens[:, :, geom_slice[0] : geom_slice[1]] = 0.0
        if drop_motion and mot_slice[1] > mot_slice[0]:
            tokens[:, :, mot_slice[0] : mot_slice[1]] = 0.0
        if drop_appearance and app_slice[1] > app_slice[0]:
            tokens[:, :, app_slice[0] : app_slice[1]] = 0.0
        t = torch.from_numpy(tokens).unsqueeze(0).to(device)
        m = torch.from_numpy(masks).unsqueeze(0).to(device)
        out = model(t, m)
        s = out.segment_scores.squeeze(0).detach().cpu().numpy().astype(np.float32)
        seg_scores.append(s)
        y_true.append(int(it.label))
        y_score.append(float(s.max()))
        uids.append(it.uid)

    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_score, dtype=np.float32)
    seg = np.stack(seg_scores, axis=0) if seg_scores else np.zeros((0, num_segments), dtype=np.float32)
    auc_video = safe_roc_auc(yt, ys)
    ap_video = average_precision(yt, ys)

    yt_seg = np.repeat(yt[:, None], seg.shape[1], axis=1) if seg.size else np.zeros((0, num_segments), dtype=np.int64)
    auc_segment = safe_roc_auc(yt_seg.reshape(-1), seg.reshape(-1)) if seg.size else float("nan")
    ap_segment = average_precision(yt_seg.reshape(-1), seg.reshape(-1)) if seg.size else float("nan")

    out_npz = seed_dir / "predictions.npz"
    np.savez_compressed(
        out_npz,
        uids=np.asarray(uids, dtype=object),
        y_true=yt,
        y_score=ys,
        seg_scores=seg,
        y_true_segment=yt_seg,
    )

    metrics = {
        "auc_video": auc_video,
        "ap_video": ap_video,
        "auc_segment": auc_segment,
        "ap_segment": ap_segment,
        "num_test": int(len(test_items)),
    }
    dump_json(seed_dir / "metrics.json", metrics)


def _find_latest_exp(outputs_root: Path) -> Optional[Path]:
    cands = [p for p in outputs_root.glob("exp_*") if p.is_dir()]
    if not cands:
        return None

    # Prefer "full" experiments (multi-run) over smoke runs, and prefer finished
    # runs (results_summary.json exists) over incomplete ones.
    def _is_multi_run(exp_dir: Path) -> bool:
        cfg_path = exp_dir / "config.yaml"
        if not cfg_path.exists():
            return False
        try:
            cfg = _load_yaml(cfg_path)
        except Exception:
            return False
        runs = ((cfg.get("experiments") or {}).get("runs") or [])
        return bool(runs)

    multi = [p for p in cands if _is_multi_run(p)]
    pool = multi if multi else cands

    finished = [p for p in pool if (p / "results_summary.json").exists()]
    pool = finished if finished else pool

    pool = sorted(pool, key=lambda p: p.name, reverse=True)
    return pool[0] if pool else None


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
