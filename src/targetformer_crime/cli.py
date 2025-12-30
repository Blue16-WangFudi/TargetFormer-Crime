from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from targetformer_crime.config import get_paths, load_config
from targetformer_crime.utils import ensure_dir


def _repo_root() -> Path:
    # cli.py lives in: <repo>/src/targetformer_crime/cli.py
    return Path(__file__).resolve().parents[2]


def cmd_audit(args: argparse.Namespace) -> None:
    from targetformer_crime.pipeline.audit import run_audit

    repo_root = _repo_root()
    cfg = load_config(args.config, repo_root=repo_root)
    run_audit(cfg, repo_root=repo_root)


def cmd_preprocess(args: argparse.Namespace) -> None:
    from targetformer_crime.pipeline.preprocess import run_preprocess

    repo_root = _repo_root()
    cfg = load_config(args.config, repo_root=repo_root)
    run_preprocess(cfg, repo_root=repo_root)


def cmd_train(args: argparse.Namespace) -> None:
    from targetformer_crime.pipeline.train import run_train

    repo_root = _repo_root()
    cfg = load_config(args.config, repo_root=repo_root)
    run_train(cfg, repo_root=repo_root)


def cmd_eval(args: argparse.Namespace) -> None:
    from targetformer_crime.pipeline.eval import run_eval

    repo_root = _repo_root()
    cfg = load_config(args.config, repo_root=repo_root)
    run_eval(cfg, repo_root=repo_root)


def cmd_viz(args: argparse.Namespace) -> None:
    from targetformer_crime.pipeline.viz import run_viz

    repo_root = _repo_root()
    cfg = load_config(args.config, repo_root=repo_root)
    run_viz(cfg, repo_root=repo_root)


def cmd_smoke(args: argparse.Namespace) -> None:
    from targetformer_crime.pipeline.smoke import run_smoke

    repo_root = _repo_root()
    cfg = load_config(args.config, repo_root=repo_root)
    run_smoke(cfg, repo_root=repo_root)


def cmd_clean(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    outputs = repo_root / "outputs"
    if outputs.exists():
        if not args.yes:
            raise SystemExit("Refusing to delete outputs/ without --yes.")
        shutil.rmtree(outputs)
    ensure_dir(outputs)


def cmd_paper(_: argparse.Namespace) -> None:
    from targetformer_crime.pipeline.paper import compile_paper

    repo_root = _repo_root()
    compile_paper(repo_root=repo_root)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="targetformer-crime")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _cfg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", type=str, required=True)

    sp = sub.add_parser("audit")
    _cfg(sp)
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("preprocess")
    _cfg(sp)
    sp.set_defaults(func=cmd_preprocess)

    sp = sub.add_parser("train")
    _cfg(sp)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("eval")
    _cfg(sp)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("viz")
    _cfg(sp)
    sp.set_defaults(func=cmd_viz)

    sp = sub.add_parser("smoke")
    _cfg(sp)
    sp.set_defaults(func=cmd_smoke)

    sp = sub.add_parser("paper")
    sp.set_defaults(func=cmd_paper)

    sp = sub.add_parser("clean")
    sp.add_argument("--yes", action="store_true")
    sp.set_defaults(func=cmd_clean)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
