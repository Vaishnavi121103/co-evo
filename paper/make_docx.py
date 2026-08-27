"""Convert the IEEE LaTeX paper to .docx.

Pandoc handles the structure, tables, maths and cross-references; this script
handles the parts pandoc cannot know about:

* **Figures.** The LaTeX build uses vector PDFs, which Word will not render.
  The same figures are emitted as PNG by ``make_figures.py``, so the graphics
  paths are rewritten to ``.png`` before conversion.
* **Custom macros.** ``\\evrate`` and ``\\osc`` are expanded to their symbols,
  since pandoc drops unknown macros silently and would otherwise leave gaps
  where the two headline metrics are named.
* **IEEE page setup.** The result is post-processed into two columns on
  US Letter with the margins IEEE specifies, so the .docx approximates the
  submitted layout rather than arriving as a single-column draft.

The .tex remains the source of record: regenerate rather than editing the
.docx, or the two will drift apart.

Usage
-----
    python paper/make_docx.py
    python paper/make_docx.py --single-column
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "main.tex"
OUT = HERE / "main.docx"

# Macros defined in the preamble that pandoc will not expand on its own.
MACROS = {
    r"\evrate": "*E*~tail~",
    r"\osc": "Ω",
}


def preprocess(tex: str) -> str:
    """Make the LaTeX pandoc-friendly without changing its meaning."""
    # Word cannot display the vector PDFs; the PNG twins are already built.
    tex = re.sub(r"(\\includegraphics\[[^\]]*\]\{)([^}]+)(\})",
                 lambda m: m.group(1) + m.group(2) + ".png" + m.group(3), tex)
    # Expand the two custom metric macros. The definitions have to go first:
    # expanding names while the \newcommand lines are still present would
    # rewrite the definitions themselves into invalid LaTeX.
    for name in MACROS:
        tex = re.sub(r"\\newcommand\{" + re.escape(name) + r"\}\{[^\n]*\}\n",
                     "", tex)
    # Longest name first, so a macro that is a prefix of another cannot eat it.
    for name in sorted(MACROS, key=len, reverse=True):
        tex = tex.replace(name + "{}", MACROS[name]).replace(name, MACROS[name])
    # Drop packages that only affect PDF typesetting and confuse the reader.
    for pkg in ("balance", "hyperref"):
        tex = re.sub(r"\\usepackage(\[[^\]]*\])?\{" + pkg + r"\}\n", "", tex)
    tex = tex.replace("\\balance\n", "")
    return tex


def two_column(docx: Path, columns: int = 2) -> None:
    """Set IEEE page geometry on the generated file.

    A .docx is a zip of XML; the page setup lives in one ``sectPr`` element at
    the end of the body, so this rewrites that rather than regenerating the
    document.
    """
    import zipfile

    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    # US Letter, IEEE margins, in twentieths of a point.
    pgsz = f'<w:pgSz w:w="12240" w:h="15840"/>'
    mar = ('<w:pgMar w:top="1080" w:right="900" w:bottom="1440" '
           'w:left="900" w:header="720" w:footer="720" w:gutter="0"/>')
    cols = (f'<w:cols w:num="{columns}" w:space="360" w:equalWidth="1"/>'
            if columns > 1 else '<w:cols w:space="720"/>')
    sect = f"<w:sectPr xmlns:w=\"{ns}\">{pgsz}{mar}{cols}</w:sectPr>"

    tmp = docx.with_suffix(".tmp.docx")
    with zipfile.ZipFile(docx) as zin, zipfile.ZipFile(
            tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                xml = data.decode("utf-8")
                if "<w:sectPr" in xml:
                    xml = re.sub(r"<w:sectPr[^>]*>.*?</w:sectPr>", sect, xml,
                                 flags=re.S)
                else:
                    xml = xml.replace("</w:body>", sect + "</w:body>")
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(docx)


def build(columns: int) -> Path:
    import pypandoc

    tex = preprocess(TEX.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        staged = Path(td) / "main.tex"
        staged.write_text(tex, encoding="utf-8")
        # Figures are referenced relative to the source, so the resource path
        # has to point back at the real figures directory.
        pypandoc.convert_file(
            str(staged), "docx", format="latex", outputfile=str(OUT),
            extra_args=[
                # os.pathsep, not ":" -- on Windows a colon splits the drive
                # letter and every image path silently fails to resolve.
                "--resource-path="
                + os.pathsep.join([str(HERE), str(HERE / "figures")]),
                "--from=latex+raw_tex",
                "--number-sections",
                "--wrap=preserve",
            ],
        )
    two_column(OUT, columns)
    return OUT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--single-column", action="store_true",
                    help="single-column layout (easier to comment on)")
    args = ap.parse_args()

    if not (HERE / "figures" / "fig1_dynamics.png").exists():
        raise SystemExit("PNG figures missing - run paper/make_figures.py first")

    out = build(1 if args.single_column else 2)
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB, "
          f"{'1' if args.single_column else '2'}-column)")
    print("The .tex stays the source of record: regenerate rather than "
          "editing the .docx.")


if __name__ == "__main__":
    main()
