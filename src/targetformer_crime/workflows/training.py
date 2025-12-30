from __future__ import annotations

import csv
import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from targetformer_crime.data.feature_cache import CacheItem, list_cache_items, load_npz_item
from targetformer_crime.losses.mil import MilLossConfig, compute_total_loss
from targetformer_crime.metrics.classification import average_precision, safe_roc_auc
from targetformer_crime.utils import dump_json, dump_yaml, ensure_dir, set_determinism
from targetformer_crime.workflows.registry import build_model


@dataclass(frozen=True)
class TrainResult:
    run_name: str
    kind: str
    seed: int
    best_epoch: int
    best_auc: float
    best_ap: float
    last_auc: float
    last_ap: float
    num_train: int
    num_test: int


class _NpzLRU:
    def __init__(self, max_items: int = 64) -> None:
        self.max_items = int(max_items)
        self._data: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []

    def get(self, path: Path) -> Dict[str, Any]:
        key = str(path)
        if key in self._data:
            # refresh
            self._order.remove(key)
            self._order.append(key)
            return self._data[key]

        item = load_npz_item(path)
        self._data[key] = item
        self._order.append(key)
        if len(self._order) > self.max_items:
            old = self._order.pop(0)
            self._data.pop(old, None)
        return item


def train_one_run(
    *,
    run_cfg: Dict[str, Any],
    out_dir: Path,
    repo_root: Path,
) -> TrainResult:
    out_dir = ensure_dir(out_dir)

    run_name = str(run_cfg["name"])
    kind = str(run_cfg["kind"])
    seed = int(run_cfg["seed"])
    set_determinism(seed, deterministic=bool(run_cfg.get("deterministic", True)))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # save config snapshot (per-run)
    dump_yaml(out_dir / "config.yaml", run_cfg)
    dump_json(out_dir / "run_meta.json", _collect_run_meta(repo_root=repo_root, seed=seed, device=device))

    feature_cache = Path(run_cfg["feature_cache"])
    train_items = list_cache_items(feature_cache, split="Train")
    test_items = list_cache_items(feature_cache, split=str(run_cfg.get("eval_split", "Test")))

    normals = [it for it in train_items if it.label == 0]
    abnormals = [it for it in train_items if it.label == 1]
    if not normals or not abnormals:
        raise RuntimeError(f"training requires both normal and abnormal. normals={len(normals)} abnormals={len(abnormals)}")

    # infer dims from first item
    sample = load_npz_item(train_items[0].path)
    raw_dim = int(sample["tokens"].shape[-1])
    num_segments = int(sample["tokens"].shape[0])
    max_k = int(sample["tokens"].shape[1])

    k = int(run_cfg.get("k", min(10, max_k)))
    if k > max_k:
        raise ValueError(f"k={k} exceeds max_k in cache={max_k}")

    model_cfg = run_cfg.get("model", {}) or {}
    model = build_model(kind=kind, input_dim=raw_dim, num_segments=num_segments, max_k=max_k, model_cfg=model_cfg)
    model.to(device)

    optim_cfg = run_cfg.get("optim", {}) or {}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 1e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(run_cfg.get("epochs", 10))
    grad_accum_steps = int(run_cfg.get("grad_accum_steps", 1))
    use_amp = bool(run_cfg.get("amp", True) and device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loss_cfg_d = run_cfg.get("loss", {}) or {}
    loss_cfg = MilLossConfig(
        margin=float(loss_cfg_d.get("margin", 1.0)),
        lambda_smooth=float(loss_cfg_d.get("lambda_smooth", 0.1)),
        lambda_sparse=float(loss_cfg_d.get("lambda_sparse", 0.001)),
        lambda_proto_div=float(loss_cfg_d.get("lambda_proto_div", 0.0)),
    )

    ablation = run_cfg.get("ablation", {}) or {}
    drop_geometry = bool(ablation.get("drop_geometry", False))
    drop_motion = bool(ablation.get("drop_motion", False))
    drop_appearance = bool(ablation.get("drop_appearance", False))

    # Layout can be used to drop features.
    feature_layout = (sample.get("meta") or {}).get("feature_layout") or {}
    geom_slice = tuple(feature_layout.get("geometry", [0, 0]))
    mot_slice = tuple(feature_layout.get("motion", [0, 0]))
    app_slice = tuple(feature_layout.get("appearance", [0, 0]))

    cache = _NpzLRU(max_items=int(run_cfg.get("io_cache_items", 64)))
    history_path = out_dir / "train_history.csv"
    _init_history_csv(history_path)

    best_auc = float("-inf")
    best_ap = float("-inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        steps_per_epoch = int(run_cfg.get("steps_per_epoch") or min(len(normals), len(abnormals)))
        steps_per_epoch = max(1, steps_per_epoch)

        running: Dict[str, float] = {"total": 0.0, "rank": 0.0, "smooth": 0.0, "sparse": 0.0, "proto_div": 0.0}
        pbar = tqdm(range(steps_per_epoch), desc=f"train {run_name} s{seed} e{epoch}", leave=False)
        for step in pbar:
            abn = random.choice(abnormals)
            nor = random.choice(normals)

            abn_pack = cache.get(abn.path)
            nor_pack = cache.get(nor.path)

            abn_tokens, abn_masks = _prepare_batch(
                abn_pack["tokens"],
                abn_pack["masks"],
                k=k,
                drop_geometry=drop_geometry,
                drop_motion=drop_motion,
                drop_appearance=drop_appearance,
                geom_slice=geom_slice,
                mot_slice=mot_slice,
                app_slice=app_slice,
                device=device,
            )
            nor_tokens, nor_masks = _prepare_batch(
                nor_pack["tokens"],
                nor_pack["masks"],
                k=k,
                drop_geometry=drop_geometry,
                drop_motion=drop_motion,
                drop_appearance=drop_appearance,
                geom_slice=geom_slice,
                mot_slice=mot_slice,
                app_slice=app_slice,
                device=device,
            )

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out_abn = model(abn_tokens, abn_masks)
                out_nor = model(nor_tokens, nor_masks)
                proto_div = model.prototype_diversity_loss() if hasattr(model, "prototype_diversity_loss") else None
                losses = compute_total_loss(out_abn.segment_scores, out_nor.segment_scores, cfg=loss_cfg, proto_div_loss=proto_div)
                loss = losses["total"] / float(grad_accum_steps)

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == steps_per_epoch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for k_loss in running.keys():
                running[k_loss] += float(losses[k_loss].detach().cpu().item())

            pbar.set_postfix({"loss": running["total"] / (step + 1)})

        for k_loss in list(running.keys()):
            running[k_loss] /= float(steps_per_epoch)

        eval_max = run_cfg.get("eval_max_videos")
        auc, ap, _ = evaluate_video_level(model, test_items, k=k, device=device, max_videos=eval_max)
        auc_cmp = auc if np.isfinite(auc) else float("-inf")

        _append_history_csv(
            history_path,
            row={
                "epoch": epoch,
                **{f"loss_{k_loss}": running[k_loss] for k_loss in running},
                "auc_video": auc,
                "ap_video": ap,
            },
        )

        is_best = auc_cmp > best_auc
        if is_best:
            best_auc, best_ap, best_epoch = auc_cmp, ap, epoch
            _save_checkpoint(out_dir / "checkpoint_best.pt", model=model, optimizer=optimizer, epoch=epoch, auc=auc, ap=ap)

        if epoch == epochs or epoch % int(run_cfg.get("save_every", 1)) == 0:
            _save_checkpoint(out_dir / "checkpoint_last.pt", model=model, optimizer=optimizer, epoch=epoch, auc=auc, ap=ap)

    last_auc, last_ap, _ = evaluate_video_level(model, test_items, k=k, device=device, max_videos=run_cfg.get("eval_max_videos"))
    return TrainResult(
        run_name=run_name,
        kind=kind,
        seed=seed,
        best_epoch=best_epoch,
        best_auc=float(best_auc),
        best_ap=float(best_ap),
        last_auc=float(last_auc),
        last_ap=float(last_ap),
        num_train=len(train_items),
        num_test=len(test_items),
    )


@torch.no_grad()
def evaluate_video_level(
    model: torch.nn.Module,
    items: List[CacheItem],
    *,
    k: int,
    device: str,
    max_videos: Optional[int] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    model.eval()
    y_true: List[int] = []
    y_score: List[float] = []
    uids: List[str] = []

    cache = _NpzLRU(max_items=32)
    if max_videos is not None:
        items = items[: int(max_videos)]

    for it in items:
        pack = cache.get(it.path)
        tokens, masks = _prepare_batch(pack["tokens"], pack["masks"], k=k, device=device)
        out = model(tokens, masks)
        score = float(out.segment_scores.max(dim=-1).values.squeeze(0).detach().cpu().item())
        y_true.append(int(it.label))
        y_score.append(score)
        uids.append(it.uid)

    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_score, dtype=np.float32)
    auc = safe_roc_auc(yt, ys)
    ap = average_precision(yt, ys)
    return auc, ap, {"uids": uids, "y_true": yt, "y_score": ys}


def _prepare_batch(
    tokens_np: np.ndarray,
    masks_np: np.ndarray,
    *,
    k: int,
    device: str,
    drop_geometry: bool = False,
    drop_motion: bool = False,
    drop_appearance: bool = False,
    geom_slice: Tuple[int, int] = (0, 0),
    mot_slice: Tuple[int, int] = (0, 0),
    app_slice: Tuple[int, int] = (0, 0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = tokens_np[:, :k, :].astype(np.float32, copy=True)
    masks = masks_np[:, :k].astype(np.uint8, copy=False)

    if drop_geometry and geom_slice[1] > geom_slice[0]:
        tokens[:, :, geom_slice[0] : geom_slice[1]] = 0.0
    if drop_motion and mot_slice[1] > mot_slice[0]:
        tokens[:, :, mot_slice[0] : mot_slice[1]] = 0.0
    if drop_appearance and app_slice[1] > app_slice[0]:
        tokens[:, :, app_slice[0] : app_slice[1]] = 0.0

    t = torch.from_numpy(tokens).unsqueeze(0).to(device)
    m = torch.from_numpy(masks).unsqueeze(0).to(device)
    return t, m


def _collect_run_meta(repo_root: Path, seed: int, device: str) -> Dict[str, Any]:
    import platform

    meta: Dict[str, Any] = {
        "seed": int(seed),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
    }
    try:
        meta["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        meta["git_commit"] = None

    meta["torch"] = getattr(torch, "__version__", None)
    meta["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        meta["cuda"] = torch.version.cuda
        meta["cudnn"] = torch.backends.cudnn.version()
        meta["gpu_name"] = torch.cuda.get_device_name(0)
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        meta["nvidia_driver"] = out.decode("utf-8").strip()
    except Exception:
        meta["nvidia_driver"] = None
    return meta


def _save_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, auc: float, ap: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "auc_video": float(auc),
            "ap_video": float(ap),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def _init_history_csv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "loss_total",
                "loss_rank",
                "loss_smooth",
                "loss_sparse",
                "loss_proto_div",
                "auc_video",
                "ap_video",
            ],
        )
        w.writeheader()


def _append_history_csv(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = [
        "epoch",
        "loss_total",
        "loss_rank",
        "loss_smooth",
        "loss_sparse",
        "loss_proto_div",
        "auc_video",
        "ap_video",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)
