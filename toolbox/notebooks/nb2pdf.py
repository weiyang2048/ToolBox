"""Execute a Jupyter notebook and typeset it as a LaTeX document.

Every code cell runs in one shared namespace, in order; its source is
typeset as a listing, its captured stdout as an "Output" listing, and
the first figure it saves is embedded. Markdown cells become LaTeX
sections, tables, and lists. The document is written as ``notebook.tex``
(plus ``nb_chart.png`` when the notebook plots) into the output
directory, ready for pdflatex.

Importing this module runs nothing. Call ``convert(notebook, title)``
or use the command line::

    python nb2pdf.py NOTEBOOK --title "Document title" [--outdir DIR]
"""

import argparse
import io
import json
import os
import re
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

TEX_NAME = "notebook.tex"
FIG_NAME = "nb_chart.png"

UNI = {"λ": r"$\lambda$", "₋": r"$_{-}$", "×": r"$\times$",
       "–": "--", "—": "---", "²": r"$^2$", "→": r"$\to$",
       "μ": r"$\mu$", "Σ": r"$\Sigma$", "σ": r"$\sigma$"}
ASCII = {"λ": "lambda", "ξ": "xi", "μ": "mu", "Σ": "Sigma",
         "σ": "sigma", "×": "x", "–": "--", "—": "---",
         "→": "->", "²": "^2", "₋": "_"}


def to_ascii(s):
    for k, v in ASCII.items():
        s = s.replace(k, v)
    return s


def uni(s):
    for k, v in UNI.items():
        s = s.replace(k, v)
    return s


def esc(s):
    # split out math spans ($$..$$ and $..$) so specials inside math survive
    parts = re.split(r'(\$\$.*?\$\$|\$[^$]*?\$)', s)
    for i in range(0, len(parts), 2):
        t = parts[i]
        t = t.replace("&", r"\&").replace("%", r"\%")
        t = t.replace("_", r"\_").replace("#", r"\#")
        parts[i] = t
    return "".join(parts)


def inline(s):
    s = uni(s)
    s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
    s = re.sub(r'(?<!\\)\*(.+?)(?<!\\)\*', r'\\textit{\1}', s)
    s = re.sub(r'`(.+?)`', r'\\texttt{\1}', s)
    return esc(s)


def md_to_tex(src):
    lines = src.split("\n")
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("### "):
            out.append("\\subsubsection{" + inline(ln[4:]) + "}")
        elif ln.startswith("## "):
            t = re.sub(r"^\d+\.\s*", "", ln[3:])
            out.append("\\subsection{" + inline(t) + "}")
        elif ln.startswith("# "):
            out.append("\\section{" + inline(ln[2:]) + "}")
        elif re.match(r'^\|.*\|\s*$', ln):
            rows = []
            while i < len(lines) and re.match(r'^\|.*\|\s*$', lines[i]):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r'^[\s\-|:]+$', lines[i].replace("|", "")):
                    rows.append(cells)
                i += 1
            ncol = len(rows[0])
            out.append("\\begin{center}\n\\begin{tabular}{" + "l" * ncol + "}")
            for r in rows:
                out.append(" & ".join(inline(c) for c in r) + r" \\")
            out.append("\\end{tabular}\n\\end{center}")
            continue
        elif ln.strip().startswith("- "):
            out.append("\\begin{itemize}")
            while i < len(lines) and lines[i].strip().startswith("- "):
                out.append("\\item " + inline(lines[i].strip()[2:]))
                i += 1
            out.append("\\end{itemize}")
            continue
        elif ln.strip() == "":
            out.append("")
        else:
            out.append(inline(ln) + "\n")
        i += 1
    return "\n".join(out)


def render_cells(nb, fig_path):
    """Execute the notebook's cells in one namespace; return the LaTeX body.

    ``plt.show()`` in a code cell is rewritten to save the figure to
    *fig_path*; the first cell after which the figure exists gets it
    embedded, exactly once.
    """
    ns = {}
    tex = []
    fig_saved = False
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            tex.append(md_to_tex(src))
            continue
        tex.append("\\begin{lstlisting}\n" + to_ascii(src) + "\n\\end{lstlisting}")
        run = src.replace("plt.show()", f"plt.savefig('{fig_path}', dpi=150)")
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(run, "<nb>", "exec"), ns)
        out = buf.getvalue().strip()
        if out:
            tex.append("\\textbf{Output:}\n\\begin{lstlisting}\n"
                       + to_ascii(out) + "\n\\end{lstlisting}")
        if os.path.exists(fig_path) and not fig_saved:
            tex.append("\\begin{center}\n\\includegraphics[width=0.95\\textwidth]{"
                       + FIG_NAME + "}\n\\end{center}")
            fig_saved = True
    return "\n\n".join(tex)


def build_document(body, title):
    """Wrap a rendered body in the document preamble with the given title."""
    return (r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,graphicx,xcolor,listings}
\lstset{breaklines=true,basicstyle=\ttfamily\small,columns=fullflexible,keepspaces=true}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}
\setlength{\parskip}{0.6em}
\title{""" + inline(title) + r"""}
\author{Executed notebook printout}
\date{\today}
\begin{document}
\maketitle
""" + body + r"""
\end{document}
""")


def convert(notebook_path, title, outdir=None):
    """Execute *notebook_path* and write its typeset document.

    *outdir* defaults to a ``pdf_build`` directory beside the notebook.
    Returns the path of the written ``notebook.tex``.
    """
    notebook_path = Path(notebook_path)
    outdir = Path(outdir) if outdir is not None else notebook_path.parent / "pdf_build"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)
    body = render_cells(nb, outdir / FIG_NAME)
    tex_path = outdir / TEX_NAME
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(build_document(body, title))
    print("tex written,", len(body), "chars")
    return tex_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Execute a Jupyter notebook and typeset it as LaTeX.")
    parser.add_argument("notebook", help="path to the .ipynb file")
    parser.add_argument("--title", required=True, help="document title")
    parser.add_argument("--outdir", default=None,
                        help="output directory (default: pdf_build beside the notebook)")
    args = parser.parse_args(argv)
    convert(args.notebook, args.title, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
