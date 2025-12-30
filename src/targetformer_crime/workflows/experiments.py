from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

from targetformer_crime.utils import dump_json, dump_yaml, ensure_dir
from targetformer_crime.workflows.training import TrainResult, train_one_run


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def run_experiments(*, cfg: Dict[str, Any], repo_root: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir = ensure_dir(out_dir)

    runs = (cfg.get("experiments", {}) or {}).get("runs")
    train_defaults = cfg.get("train", {}) or {}
    if not runs:
        # single-run mode via cfg.train
        train_cfg = train_defaults
        runs = [
            {
                "name": str(train_cfg.get("exp_name", "single_run")),
                "kind": str((train_cfg.get("model", {}) or {}).get("name", "targetformer")),
                "seeds": train_cfg.get("seeds", [0]),
                "epochs": int(train_cfg.get("epochs", 10)),
                "k": int(train_cfg.get("k", 10)),
                "model": train_cfg.get("model", {}) or {},
                "optim": train_cfg.get("optim", {}) or {},
                "loss": train_cfg.get("loss", {}) or {},
                "feature_cache": str(train_cfg.get("feature_cache", (cfg.get("preprocess", {}) or {}).get("out_dir", "outputs/precomputed"))),
            }
        ]

    base_optim = cfg.get("optim", {}) or train_defaults.get("optim", {}) or {}
    base_loss = cfg.get("loss", {}) or train_defaults.get("loss", {}) or {}

    results: List[TrainResult] = []

    for run in runs:
        run_name = str(run.get("name"))
        kind = str(run.get("kind", "targetformer"))
        seeds = list(run.get("seeds") or (cfg.get("train", {}) or {}).get("seeds") or [0])

        for seed in seeds:
            seed_dir = ensure_dir(out_dir / run_name / f"seed_{int(seed)}")

            run_cfg = {
                "name": run_name,
                "kind": kind,
                "seed": int(seed),
                "deterministic": bool(cfg.get("deterministic", True)),
                "feature_cache": str(
                    run.get("feature_cache")
                    or train_defaults.get("feature_cache")
                    or (cfg.get("preprocess", {}) or {}).get("out_dir", "outputs/precomputed")
                ),
                "epochs": int(run.get("epochs") or train_defaults.get("epochs") or cfg.get("epochs_default") or 10),
                "k": int(run.get("k") or train_defaults.get("k") or 10),
                "grad_accum_steps": int(
                    run.get("grad_accum_steps") or train_defaults.get("grad_accum_steps") or cfg.get("grad_accum_steps") or 1
                ),
                "amp": bool(run.get("amp") if "amp" in run else train_defaults.get("amp", cfg.get("amp", True))),
                "steps_per_epoch": run.get("steps_per_epoch") or train_defaults.get("steps_per_epoch"),
                "eval_split": str((cfg.get("eval", {}) or {}).get("split", "Test")),
                "eval_max_videos": run.get("eval_max_videos") or train_defaults.get("eval_max_videos"),
                "save_every": int(run.get("save_every") or train_defaults.get("save_every") or cfg.get("save_every") or 1),
                "io_cache_items": int(run.get("io_cache_items") or train_defaults.get("io_cache_items") or 64),
                "optim": _deep_merge(base_optim, run.get("optim") or {}),
                "loss": _deep_merge(base_loss, run.get("loss") or {}),
                "model": _deep_merge((cfg.get("train", {}) or {}).get("model", {}) or {}, run.get("model") or {}),
                "ablation": _deep_merge((cfg.get("train", {}) or {}).get("ablation", {}) or {}, run.get("ablation") or {}),
            }

            tr = train_one_run(run_cfg=run_cfg, out_dir=seed_dir, repo_root=repo_root)
            results.append(tr)

    summary = _summarize(results)
    dump_json(out_dir / "results_summary.json", summary)
    _write_results_table(out_dir / "results_table.csv", results=results, summary=summary)
    dump_json(out_dir / "metrics.json", summary)
    _write_results_table(out_dir / "metrics.csv", results=results, summary=summary)
    return summary


def _summarize(results: List[TrainResult]) -> Dict[str, Any]:
    by_run: Dict[str, List[TrainResult]] = {}
    for r in results:
        by_run.setdefault(r.run_name, []).append(r)

    out: Dict[str, Any] = {"runs": {}, "num_runs": len(by_run)}
    for name, rs in by_run.items():
        best_aucs = [r.best_auc for r in rs]
        best_aps = [r.best_ap for r in rs]
        out["runs"][name] = {
            "n_seeds": len(rs),
            "best_auc_mean": statistics.fmean(best_aucs) if best_aucs else float("nan"),
            "best_auc_std": statistics.pstdev(best_aucs) if len(best_aucs) > 1 else 0.0,
            "best_ap_mean": statistics.fmean(best_aps) if best_aps else float("nan"),
            "best_ap_std": statistics.pstdev(best_aps) if len(best_aps) > 1 else 0.0,
        }
    return out


def _write_results_table(path: Path, *, results: List[TrainResult], summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_name",
                "kind",
                "seed",
                "best_epoch",
                "best_auc",
                "best_ap",
                "last_auc",
                "last_ap",
                "num_train",
                "num_test",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.run_name,
                    r.kind,
                    r.seed,
                    r.best_epoch,
                    f"{r.best_auc:.6f}",
                    f"{r.best_ap:.6f}",
                    f"{r.last_auc:.6f}",
                    f"{r.last_ap:.6f}",
                    r.num_train,
                    r.num_test,
                ]
            )
