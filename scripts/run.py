from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def _inject_src() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    sys.path.insert(0, str(src))


def main() -> None:
    if platform.system().lower() == "windows":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    _inject_src()
    from targetformer_crime.cli import main as _main

    _main()


if __name__ == "__main__":
    main()
