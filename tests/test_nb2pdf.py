"""Tests for toolbox.notebooks.nb2pdf: import safety, parameterized entry point.

Pure stdlib + matplotlib (Agg), headless, deterministic — safe for CI.
"""

import json
import subprocess
import sys
from pathlib import Path

from toolbox.notebooks import nb2pdf

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = Path(nb2pdf.__file__)
TITLE = "Fixture × Test — a title"


def write_fixture_notebook(path):
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Fixture Section\n",
                "\n",
                "Some **bold** text and a λ symbol.\n",
                "\n",
                "- first item\n",
                "- second item\n",
                "\n",
                "| A | B |\n",
                "|---|---|\n",
                "| 1 | 2 |",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": [
                 "x = 40 + 2\n",
                 "print('answer', x)",
             ]},
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": [
                 "import matplotlib.pyplot as plt\n",
                 "plt.plot([0, 1], [0, 1])\n",
                 "plt.show()\n",
                 "print('plotted')",
             ]},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


def test_import_is_side_effect_free(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import toolbox.notebooks.nb2pdf"],
        cwd=tmp_path,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_convert_writes_expected_tex(tmp_path, capsys):
    nb_path = write_fixture_notebook(tmp_path / "fixture.ipynb")
    outdir = tmp_path / "out"
    tex_path = nb2pdf.convert(nb_path, TITLE, outdir)
    assert tex_path == outdir / "notebook.tex"
    assert "tex written," in capsys.readouterr().out
    tex = tex_path.read_text(encoding="utf-8")
    assert r"\title{Fixture $\times$ Test --- a title}" in tex
    assert r"\section{Fixture Section}" in tex
    assert r"\textbf{bold}" in tex
    assert r"$\lambda$" in tex
    assert r"\item first item" in tex
    assert "1 & 2" in tex
    assert "x = 40 + 2" in tex
    assert "answer 42" in tex
    assert "plotted" in tex
    assert r"\includegraphics" in tex
    assert (outdir / "nb_chart.png").exists()


def test_convert_default_outdir_is_pdf_build_beside_notebook(tmp_path):
    nb_path = write_fixture_notebook(tmp_path / "fixture.ipynb")
    tex_path = nb2pdf.convert(nb_path, TITLE)
    assert tex_path == tmp_path / "pdf_build" / "notebook.tex"
    assert tex_path.exists()


def test_main_cli(tmp_path):
    nb_path = write_fixture_notebook(tmp_path / "fixture.ipynb")
    outdir = tmp_path / "cli-out"
    rc = nb2pdf.main([str(nb_path), "--title", TITLE, "--outdir", str(outdir)])
    assert rc == 0
    assert (outdir / "notebook.tex").exists()


def test_no_hardcoded_notebook_path_or_title():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "bbp_x_james_stein" not in source
    assert "James--Stein" not in source
    assert "James–Stein" not in source
    assert "/home/hatch" not in source
