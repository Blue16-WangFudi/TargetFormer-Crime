from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from targetformer_crime.pipeline.audit import run_audit
from targetformer_crime.pipeline.eval import run_eval
from targetformer_crime.pipeline.preprocess import run_preprocess
from targetformer_crime.pipeline.train import run_train
from targetformer_crime.pipeline.viz import run_viz


def run_smoke(cfg: Dict[str, Any], repo_root: Path) -> None:
    run_audit(cfg, repo_root=repo_root)
    run_preprocess(cfg, repo_root=repo_root)
    run_train(cfg, repo_root=repo_root)
    run_eval(cfg, repo_root=repo_root)
    run_viz(cfg, repo_root=repo_root)

