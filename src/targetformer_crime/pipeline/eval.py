from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from targetformer_crime.workflows.evaluation import eval_latest_experiment


def run_eval(cfg: Dict[str, Any], repo_root: Path) -> None:
    eval_latest_experiment(cfg=cfg, repo_root=repo_root)

