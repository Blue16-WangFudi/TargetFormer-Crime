from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, repo_root: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return load_yaml(path)


@dataclass(frozen=True)
class Paths:
    datasets_root: Path
    outputs_root: Path


def get_paths(cfg: Dict[str, Any], repo_root: Path) -> Paths:
    paths = cfg.get("paths", {}) or {}

    datasets_root_raw = paths.get("datasets_root")
    if datasets_root_raw is None:
        # Auto-detect common dataset mounts. Prefer an explicit config/env override.
        candidates = [
            Path("/datasets"),
            (repo_root.parent / "datasets"),
            (repo_root / "datasets"),
        ]

        def _looks_like_split_root(p: Path) -> bool:
            return (p / "Train").exists() and (p / "Test").exists()

        datasets_root = next((p for p in candidates if p.exists() and _looks_like_split_root(p)), Path("../datasets"))
    else:
        datasets_root = Path(datasets_root_raw)

    outputs_root = Path(paths.get("outputs_root", "outputs"))

    override = (  # optional runtime override
        __import__("os").environ.get("TFC_DATASETS_ROOT") or __import__("os").environ.get("DATASETS_ROOT")
    )
    if override:
        datasets_root = Path(override)

    if not datasets_root.is_absolute():
        datasets_root = (repo_root / datasets_root).resolve()
    if not outputs_root.is_absolute():
        outputs_root = (repo_root / outputs_root).resolve()

    # If the configured path doesn't exist (e.g., after moving the dataset),
    # fall back to auto-detection so the pipeline keeps working.
    if not datasets_root.exists():
        candidates = [
            (repo_root / "datasets"),
            Path("/datasets"),
            (repo_root.parent / "datasets"),
        ]

        def _looks_like_split_root(p: Path) -> bool:
            return (p / "Train").exists() and (p / "Test").exists()

        for cand in candidates:
            if cand.exists() and _looks_like_split_root(cand):
                datasets_root = cand.resolve()
                break

    return Paths(datasets_root=datasets_root, outputs_root=outputs_root)        
