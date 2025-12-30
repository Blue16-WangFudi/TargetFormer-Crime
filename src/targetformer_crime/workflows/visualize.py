from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from targetformer_crime.data.feature_cache import load_npz_item
from targetformer_crime.metrics.classification import pr_curve_data, roc_curve_data, safe_roc_auc
from targetformer_crime.utils import ensure_dir
from targetformer_crime.workflows.registry import build_model


def viz_latest_experiment(*, cfg: Dict[str, Any], repo_root: Path) -> None:
    outputs_root = Path((cfg.get("paths", {}) or {}).get("outputs_root", "outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (repo_root / outputs_root).resolve()

    viz_cfg = cfg.get("viz", {}) or {}
    figures_dir = ensure_dir(repo_root / Path(viz_cfg.get("figures_dir", "outputs/figures")))
    figures_data_dir = ensure_dir(repo_root / Path(viz_cfg.get("figures_data_dir", "outputs/figures_data")))

    exp_dir = _find_latest_exp(outputs_root)
    if exp_dir is None:
        raise FileNotFoundError(f"no experiments found under {outputs_root}")

    run_dirs = sorted([p for p in exp_dir.glob("*") if p.is_dir()])
    for run_dir in run_dirs:
        seed_dirs = sorted([p for p in run_dir.glob("seed_*") if p.is_dir()])   
        for sd in seed_dirs:
            _viz_one_seed_dir(sd, figures_dir=figures_dir, figures_data_dir=figures_data_dir)

    # Extra interpretability artifacts for the main model if available; otherwise fallback.
    if run_dirs:
        pick_run = next((p for p in run_dirs if p.name == "main_targetformer"), run_dirs[0])
        seed_dirs = sorted([p for p in pick_run.glob("seed_*") if p.is_dir()])
        if seed_dirs:
            seed_dir = seed_dirs[0]
            _viz_embeddings_and_attention(
                seed_dir=seed_dir,
                figures_dir=figures_dir,
                figures_data_dir=figures_data_dir,
            )
            _viz_qualitative_videos(
                seed_dir=seed_dir,
                out_dir=ensure_dir(outputs_root / "qualitative"),
                max_videos=5,
                max_frames=240,
                upscale=4,
            )


def _viz_one_seed_dir(seed_dir: Path, *, figures_dir: Path, figures_data_dir: Path) -> None:
    run_name = seed_dir.parent.name
    seed_name = seed_dir.name
    prefix = f"{run_name}_{seed_name}"

    hist = seed_dir / "train_history.csv"
    if hist.exists():
        df = pd.read_csv(hist)
        df.to_csv(figures_data_dir / f"{prefix}_train_history.csv", index=False, encoding="utf-8-sig")

        # Loss curves
        plt.figure(figsize=(8, 4))
        for col in [c for c in df.columns if c.startswith("loss_")]:
            plt.plot(df["epoch"], df[col], label=col)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss ({run_name} {seed_name})")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(figures_dir / f"{prefix}_loss_curves.png", dpi=200)
        plt.close()

        # AUC curve
        if "auc_video" in df.columns:
            plt.figure(figsize=(6, 4))
            plt.plot(df["epoch"], df["auc_video"], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("AUC (video)")
            plt.title(f"AUC vs Epoch ({run_name} {seed_name})")
            plt.tight_layout()
            plt.savefig(figures_dir / f"{prefix}_auc_curve.png", dpi=200)
            plt.close()

    pred = seed_dir / "predictions.npz"
    if pred.exists():
        d = np.load(pred, allow_pickle=True)
        y_true = d["y_true"].astype(int)
        y_score = d["y_score"].astype(float)
        roc = roc_curve_data(y_true, y_score)
        pr = pr_curve_data(y_true, y_score)

        _save_curve_csv(figures_data_dir / f"{prefix}_roc.csv", x=roc.x, y=roc.y, thr=roc.thresholds, x_name="fpr", y_name="tpr")
        _save_curve_csv(
            figures_data_dir / f"{prefix}_pr.csv", x=pr.x, y=pr.y, thr=pr.thresholds, x_name="recall", y_name="precision"
        )

        auc = safe_roc_auc(y_true, y_score)
        plt.figure(figsize=(5, 4))
        plt.plot(roc.x, roc.y, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", linewidth=1)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC ({run_name} {seed_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"{prefix}_roc.png", dpi=200)
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.plot(pr.x, pr.y)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR ({run_name} {seed_name})")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{prefix}_pr.png", dpi=200)
        plt.close()


def _viz_embeddings_and_attention(*, seed_dir: Path, figures_dir: Path, figures_data_dir: Path) -> None:
    import sklearn.manifold

    config_path = seed_dir / "config.yaml"
    ckpt_path = seed_dir / "checkpoint_best.pt"
    if not config_path.exists() or not ckpt_path.exists():
        return

    run_cfg = _load_yaml(config_path)
    feature_cache = Path(run_cfg["feature_cache"])
    test_dir = feature_cache / str(run_cfg.get("eval_split", "Test"))
    test_files = sorted(test_dir.glob("*.npz"))
    if not test_files:
        return

    sample = load_npz_item(test_files[0])
    raw_dim = int(sample["tokens"].shape[-1])
    num_segments = int(sample["tokens"].shape[0])
    max_k = int(sample["tokens"].shape[1])
    k = int(run_cfg.get("k", min(10, max_k)))

    model = build_model(kind=run_cfg["kind"], input_dim=raw_dim, num_segments=num_segments, max_k=max_k, model_cfg=run_cfg.get("model", {}) or {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Collect segment embeddings for t-SNE (limited)
    max_videos = int(run_cfg.get("viz_max_videos", 120))
    embs: List[np.ndarray] = []
    labs: List[int] = []
    for p in test_files[:max_videos]:
        pack = load_npz_item(p)
        tokens = pack["tokens"][:, :k, :].astype(np.float32, copy=False)
        masks = pack["masks"][:, :k].astype(np.uint8, copy=False)
        t = torch.from_numpy(tokens).unsqueeze(0).to(device)
        m = torch.from_numpy(masks).unsqueeze(0).to(device)
        out = model(t, m)
        z = out.segment_embeddings.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (N,D)
        embs.append(z)
        labs.extend([int(pack["label"])] * z.shape[0])

    if not embs:
        return
    X = np.concatenate(embs, axis=0)
    y = np.asarray(labs, dtype=np.int64)
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto", perplexity=30)
    X2 = tsne.fit_transform(X)
    df = pd.DataFrame({"x": X2[:, 0], "y": X2[:, 1], "label": y})
    df.to_csv(figures_data_dir / f"{seed_dir.parent.name}_{seed_dir.name}_tsne.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(6, 5))
    for lab, name in [(0, "Normal"), (1, "Abnormal")]:
        sub = df[df["label"] == lab]
        plt.scatter(sub["x"], sub["y"], s=6, alpha=0.6, label=name)
    plt.legend()
    plt.title("t-SNE of Segment Embeddings")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{seed_dir.parent.name}_{seed_dir.name}_tsne.png", dpi=220)
    plt.close()

    # Attention heatmap for one video (token weights)
    pack = load_npz_item(test_files[0])
    tokens = pack["tokens"][:, :k, :].astype(np.float32, copy=False)
    masks = pack["masks"][:, :k].astype(np.uint8, copy=False)
    t = torch.from_numpy(tokens).unsqueeze(0).to(device)
    m = torch.from_numpy(masks).unsqueeze(0).to(device)
    out = model(t, m)
    w = out.token_weights.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (N,K)
    np.savetxt(figures_data_dir / f"{seed_dir.parent.name}_{seed_dir.name}_token_weights.csv", w, delimiter=",")

    plt.figure(figsize=(8, 4))
    plt.imshow(w, aspect="auto", cmap="viridis")
    plt.colorbar(label="Token weight")
    plt.xlabel("Target index")
    plt.ylabel("Segment index")
    plt.title("Token Contribution Heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{seed_dir.parent.name}_{seed_dir.name}_token_heatmap.png", dpi=220)
    plt.close()


def _viz_qualitative_videos(
    *,
    seed_dir: Path,
    out_dir: Path,
    max_videos: int,
    max_frames: int,
    upscale: int,
) -> None:
    import cv2

    pred_path = seed_dir / "predictions.npz"
    if not pred_path.exists():
        return
    d = np.load(pred_path, allow_pickle=True)
    uids = d["uids"].astype(object).tolist()
    seg_scores = d["seg_scores"].astype(np.float32)
    uid_to_scores = {str(u): seg_scores[i] for i, u in enumerate(uids)}

    # Pick a few videos: prefer abnormal first.
    y_true = d["y_true"].astype(int)
    order = list(np.argsort(-y_true))  # abnormal first
    picked = [uids[i] for i in order[:max_videos]]

    data_dir = ensure_dir(out_dir.parent / "qualitative_data")

    for uid in picked:
        # Find cache npz (search within seed config cache)
        cfg_path = seed_dir / "config.yaml"
        if not cfg_path.exists():
            continue
        run_cfg = _load_yaml(cfg_path)
        cache_dir = Path(run_cfg["feature_cache"]) / str(run_cfg.get("eval_split", "Test"))
        npz_path = cache_dir / f"{uid}.npz"
        if not npz_path.exists():
            continue
        pack = load_npz_item(npz_path)
        meta = pack.get("meta") or {}
        frames_dir = meta.get("frames_dir")
        prefix = meta.get("frames_prefix")
        ext = meta.get("frame_ext", ".png")
        if not frames_dir or not prefix:
            continue

        tokens = pack["tokens"].astype(np.float32, copy=False)
        masks = pack["masks"].astype(np.uint8, copy=False)
        track_ids = pack.get("track_ids")
        k_cfg = int(run_cfg.get("k", min(10, int(tokens.shape[1]))))
        k_draw = max(1, min(k_cfg, int(tokens.shape[1]), 5))

        frames_dir_p = Path(frames_dir)
        files = sorted(frames_dir_p.glob(f"{prefix}_*{ext}"))
        if not files:
            continue

        scores = uid_to_scores.get(str(uid))
        if scores is None:
            continue
        nseg = int(scores.shape[0])
        peak_segs = np.argsort(scores)[-3:].astype(int).tolist() if nseg > 0 else []
        idx_min = int(meta.get("frame_idx_min", 0))
        idx_max = int(meta.get("frame_idx_max", idx_min + 1))
        denom = max(1, idx_max - idx_min)

        timeline_path = data_dir / f"{seed_dir.parent.name}_{seed_dir.name}_{uid}_timeline.csv"
        with timeline_path.open("w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["segment_id", "score", "is_peak"])
            peak_set = set(int(x) for x in peak_segs)
            for i, sc in enumerate(scores.tolist()):
                wcsv.writerow([int(i), float(sc), 1 if int(i) in peak_set else 0])

        # Prepare writer
        first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
        if first is None:
            continue
        h0, w0 = first.shape[:2]
        w = w0 * upscale
        h = h0 * upscale
        out_path = out_dir / f"{seed_dir.parent.name}_{seed_dir.name}_{uid}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 10, (w, h))

        for fp in files[:max_frames]:
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

            # Parse frame index from filename suffix
            stem = fp.stem
            frame_idx = idx_min
            if "_" in stem:
                try:
                    frame_idx = int(stem.rsplit("_", 1)[1])
                except Exception:
                    frame_idx = idx_min
            seg_id = int((frame_idx - idx_min) / float(denom) * nseg)
            seg_id = max(0, min(nseg - 1, seg_id))
            s = float(scores[seg_id])

            # Draw (approx) segment-level target boxes + ids
            geom = tokens[seg_id, :k_draw, 0:4]
            mseg = masks[seg_id, :k_draw]
            tids = track_ids[seg_id, :k_draw] if track_ids is not None else None
            for j in range(k_draw):
                if int(mseg[j]) == 0:
                    continue
                cx, cy, bw, bh = [float(x) for x in geom[j].tolist()]
                x1 = int((cx - bw / 2.0) * w)
                y1 = int((cy - bh / 2.0) * h)
                x2 = int((cx + bw / 2.0) * w)
                y2 = int((cy + bh / 2.0) * h)
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                tid = None
                if tids is not None:
                    try:
                        tid_i = int(tids[j])
                        tid = tid_i if tid_i >= 0 else None
                    except Exception:
                        tid = None
                tag = f"id={tid}" if tid is not None else f"k={j}"
                cv2.putText(
                    img,
                    tag,
                    (x1, max(10, y1 + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

            # Overlay score text
            cv2.putText(img, f"uid={uid}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, f"seg={seg_id} score={s:.3f}", (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if peak_segs:
                cv2.putText(
                    img,
                    f"peaks={','.join(str(x) for x in peak_segs)}",
                    (5, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                )

            # Simple timeline bar at bottom
            bar_h = max(8, h // 20)
            y0 = h - bar_h
            cv2.rectangle(img, (0, y0), (w, h), (0, 0, 0), thickness=-1)
            x_cur = int((seg_id + 0.5) / nseg * w)
            cv2.line(img, (x_cur, y0), (x_cur, h), (0, 255, 255), 2)
            for ps in peak_segs:
                x_peak = int((int(ps) + 0.5) / nseg * w)
                cv2.line(img, (x_peak, y0), (x_peak, h), (0, 0, 255), 1)

            writer.write(img)

        writer.release()


def _save_curve_csv(path: Path, *, x: np.ndarray, y: np.ndarray, thr: np.ndarray, x_name: str, y_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([x_name, y_name, "threshold"])
        for xi, yi, ti in zip(x, y, thr, strict=False):
            w.writerow([float(xi), float(yi), float(ti)])


def _find_latest_exp(outputs_root: Path) -> Optional[Path]:
    cands = [p for p in outputs_root.glob("exp_*") if p.is_dir()]
    if not cands:
        return None

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
