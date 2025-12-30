from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from targetformer_crime.utils import ensure_dir


def _save_data_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_dataset_audit(
    manifest_csv: Path,
    figures_dir: Path,
    figures_data_dir: Path,
    prefix: str = "dataset",
) -> None:
    figures_dir = ensure_dir(figures_dir)
    figures_data_dir = ensure_dir(figures_data_dir)

    df = pd.read_csv(manifest_csv)

    # 1) category counts
    cat = df["category"].fillna("Unknown").value_counts().reset_index()
    cat.columns = ["category", "count"]
    _save_data_csv(figures_data_dir / f"{prefix}_category_counts.csv", cat)
    plt.figure(figsize=(10, 5))
    plt.bar(cat["category"], cat["count"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Category Counts")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{prefix}_category_counts.png", dpi=200)
    plt.close()

    # 2) duration histogram
    dur = df[["duration"]].dropna()
    _save_data_csv(figures_data_dir / f"{prefix}_durations.csv", dur)
    plt.figure(figsize=(6, 4))
    plt.hist(dur["duration"].astype(float), bins=30)
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.title("Duration Histogram")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{prefix}_duration_hist.png", dpi=200)
    plt.close()

    # 3) fps distribution
    fps = df[["fps"]].dropna()
    _save_data_csv(figures_data_dir / f"{prefix}_fps.csv", fps)
    plt.figure(figsize=(6, 4))
    plt.hist(fps["fps"].astype(float), bins=30)
    plt.xlabel("FPS")
    plt.ylabel("Count")
    plt.title("FPS Distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{prefix}_fps_hist.png", dpi=200)
    plt.close()

    # 4) resolution distribution (scatter)
    res = df[["width", "height"]].dropna()
    _save_data_csv(figures_data_dir / f"{prefix}_resolution.csv", res)
    plt.figure(figsize=(6, 4))
    plt.scatter(res["width"].astype(float), res["height"].astype(float), s=8, alpha=0.6)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Resolution Distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{prefix}_resolution_scatter.png", dpi=200)
    plt.close()

