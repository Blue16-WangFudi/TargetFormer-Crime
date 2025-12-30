from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from targetformer_crime.workflows.visualize import viz_latest_experiment


def run_viz(cfg: Dict[str, Any], repo_root: Path) -> None:
    viz_latest_experiment(cfg=cfg, repo_root=repo_root)

