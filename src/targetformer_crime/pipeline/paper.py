from __future__ import annotations

import os
import subprocess
from pathlib import Path


def compile_paper(repo_root: Path) -> None:
    latex_dir = (repo_root.parent / "latex").resolve()
    main_tex = latex_dir / "main.tex"
    if not main_tex.exists():
        raise FileNotFoundError(f"latex template not found: {main_tex}")

    jobname = main_tex.stem
    out_pdf = latex_dir / f"{jobname}.pdf"
    jobname_flag = []

    # On Windows, an open PDF viewer can lock the output file and make xelatex fail
    # with "Unable to open <jobname>.pdf". Detect this and compile to a fallback
    # jobname instead so `make paper` still succeeds.
    if out_pdf.exists():
        try:
            with out_pdf.open("rb+"):
                pass
        except OSError:
            fallback = f"{jobname}_build"
            jobname_flag = ["-jobname", fallback]

    # Two-pass XeLaTeX compile (bib handled separately if needed)
    for _ in range(2):
        subprocess.check_call(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", *jobname_flag, main_tex.name],
            cwd=str(latex_dir),
            env={**os.environ, "TEXMFOUTPUT": str(latex_dir)},
        )
