from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from targetformer_crime.config import get_paths
from targetformer_crime.utils import dump_json, dump_yaml, ensure_dir, set_determinism, utc_timestamp
from targetformer_crime.workflows.experiments import run_experiments


def run_train(cfg: Dict[str, Any], repo_root: Path) -> None:
    set_determinism(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", True)))
    paths = get_paths(cfg, repo_root=repo_root)

    exp_root = ensure_dir(repo_root / Path(cfg.get("experiments", {}).get("exp_root", "outputs")))
    stamp = utc_timestamp()
    out_dir = ensure_dir(exp_root / f"exp_{stamp}")

    dump_yaml(out_dir / "config.yaml", cfg)
    dump_json(out_dir / "paths.json", {"datasets_root": str(paths.datasets_root), "repo_root": str(repo_root)})

    result = run_experiments(cfg=cfg, repo_root=repo_root, out_dir=out_dir)
    dump_json(out_dir / "summary.json", result)
