"""Convert the IEEE LaTeX paper to .docx using the official IEEE template.

Pandoc handles structure, tables, maths and cross-references. Everything below
covers what pandoc cannot know about.

**The template supplies the look.** ``conference-template-a4.docx`` is passed as
pandoc's reference document, so A4 page size, margins, fonts and the IEEE
paragraph styles come from IEEE rather than from us.

**Style names have to be remapped.** Pandoc emits its own style identifiers
(``Title``, ``ImageCaption``, ...) which are not the ones the IEEE template
defines (``papertitle``, ``figurecaption``, ...). Without the remapping the
document silently falls back to Normal and looks nothing like the template,
which is easy to miss because no error is raised.

**Sectioning has to be rebuilt.** Pandoc copies a single section layout from
the reference document and picks the wrong one --- the template's final,
single-column section. Real IEEE layout is single-column title and abstract
followed by a two-column body, which is two sections separated by a continuous
break. That break is inserted here.

**Figures and macros.** The LaTeX build uses vector PDFs, which Word will not
render, so graphics paths are rewritten to the PNG twins that
``make_figures.py`` also emits. The custom ``\\evrate`` and ``\\osc`` macros are
expanded, since pandoc drops unknown macros silently and would leave gaps
exactly where the two headline metrics are named.

The ``.tex`` remains the source of record: regenerate rather than editing the
``.docx``, or the two will drift apart.

Usage
-----
    python paper/make_docx.py
    python paper/make_docx.py --single-column   # easier to comment on
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "main.tex"
OUT = HERE / "main.docx"
TEMPLATE = HERE / "conference-template-a4.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Preamble macros pandoc will not expand on its own.
MACROS = {r"\evrate": "*E*~tail~", r"\osc": "\u03a9"}

# Pandoc's style identifiers -> the identifiers the IEEE template defines.
# Anything left unmapped falls back to Normal and loses the template's look.
STYLE_MAP = {
    "Title": "papertitle",
    "Author": "Author",
    "ImageCaption": "figurecaption",
    "TableCaption": "tablehead",
    "Bibliography": "references",
    "FirstParagraph": "BodyText",
    "Compact": "BodyText",
    "SourceCode": "BodyText",
}


def preprocess(tex: str) -> str:
    """Make the LaTeX pandoc-friendly without changing its meaning."""
    # Word cannot display the vector PDFs; the PNG twins are already built.
    tex = re.sub(r"(\\includegraphics\[[^\]]*\]\{)([^}]+)(\})",
                 lambda m: m.group(1) + m.group(2) + ".png" + m.group(3), tex)
    # Strip the \newcommand definitions before expanding uses of those macros:
    # expanding first would rewrite the definitions into invalid LaTeX.
    for name in MACROS:
        tex = re.sub(r"\\newcommand\{" + re.escape(name) + r"\}\{[^\n]*\}\n",
                     "", tex)
    # Longest first, so a macro that prefixes another cannot consume it.
    for name in sorted(MACROS, key=len, reverse=True):
        tex = tex.replace(name + "{}", MACROS[name]).replace(name, MACROS[name])
    # Packages that only affect PDF typesetting.
    for pkg in ("balance", "hyperref"):
        tex = re.sub(r"\\usepackage(\[[^\]]*\])?\{" + pkg + r"\}\n", "", tex)
    return tex.replace("\\balance\n", "")


def _sect(columns: int, continuous: bool) -> str:
    """A section-properties block on the template's A4 geometry."""
    typ = '<w:type w:val="continuous"/>' if continuous else ""
    cols = (f'<w:cols w:num="{columns}" w:space="18pt" w:equalWidth="1"/>'
            if columns > 1 else '<w:cols w:space="36pt"/>')
    return (f'<w:sectPr xmlns:w="{W}">{typ}'
            f'<w:pgSz w:w="595.30pt" w:h="841.90pt" w:code="9"/>'
            f'<w:pgMar w:top="36pt" w:right="45pt" w:bottom="36pt" '
            f'w:left="45pt" w:header="0pt" w:footer="0pt" w:gutter="0pt"/>'
            f'{cols}</w:sectPr>')


def _apply_styles(xml: str) -> tuple[str, dict]:
    counts: dict[str, int] = {}

    def sub(m: re.Match) -> str:
        old = m.group(1)
        new = STYLE_MAP.get(old)
        if new is None:
            return m.group(0)
        counts[f"{old} -> {new}"] = counts.get(f"{old} -> {new}", 0) + 1
        return m.group(0).replace(f'w:val="{old}"', f'w:val="{new}"')

    return re.sub(r'<w:pStyle w:val="([^"]+)"\s*/?>', sub, xml), counts


def _split_sections(xml: str, columns: int) -> tuple[str, bool]:
    """Single-column front matter, then a two-column body.

    A ``sectPr`` inside a paragraph governs the section *ending* at that
    paragraph, so placing a single-column block in the last front-matter
    paragraph leaves the two-column block at the end of the body to govern
    everything after it.
    """
    if columns < 2:
        return xml, False
    # The keywords paragraph is the last of the front matter.
    paras = list(re.finditer(r"<w:p\b.*?</w:p>", xml, re.S))
    anchor = None
    for p in paras:
        if 'w:val="Keywords"' in p.group(0) or "Index Terms" in p.group(0):
            anchor = p
    if anchor is None:                       # fall back to the abstract
        for p in paras:
            if 'w:val="Abstract"' in p.group(0):
                anchor = p
    if anchor is None:
        return xml, False

    block = anchor.group(0)
    front = _sect(1, continuous=False)
    if "<w:pPr>" in block:
        patched = block.replace("<w:pPr>", "<w:pPr>" + front, 1)
    else:
        patched = block.replace("<w:p ", "<w:p ", 1)
        patched = re.sub(r"(<w:p\b[^>]*>)", r"\1<w:pPr>" + front + "</w:pPr>",
                         patched, count=1)
    return xml[:anchor.start()] + patched + xml[anchor.end():], True


def _set_body_section(xml: str, sect: str) -> str:
    """Replace only the body-level ``sectPr``, the one just before ``</w:body>``.

    Done by index rather than by regex: a lazy pattern anchored on
    ``</w:body>`` will happily start at the front-matter section inserted
    above and swallow the entire document between the two, which produces a
    valid but empty file rather than an error.
    """
    end = xml.rfind("</w:body>")
    if end == -1:
        return xml
    start = xml.rfind("<w:sectPr", 0, end)
    # Only a *body-level* sectPr sits directly before </w:body>; a paragraph
    # one would be followed by </w:pPr></w:p>.
    if start != -1 and "</w:p>" not in xml[start:end]:
        return xml[:start] + sect + xml[end:]
    return xml[:end] + sect + xml[end:]


def postprocess(docx: Path, columns: int) -> dict:
    tmp = docx.with_suffix(".tmp.docx")
    report: dict = {}
    with zipfile.ZipFile(docx) as zin, zipfile.ZipFile(
            tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                xml = data.decode("utf-8")
                xml, report["styles"] = _apply_styles(xml)
                xml, report["split"] = _split_sections(xml, columns)
                xml = _set_body_section(
                    xml, _sect(columns, continuous=bool(report["split"])))
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(docx)
    return report


def build(columns: int) -> tuple[Path, dict]:
    import pypandoc

    tex = preprocess(TEX.read_text(encoding="utf-8"))
    args = [
        # os.pathsep, not ":" -- on Windows a colon splits the drive letter and
        # every image silently fails to resolve, with no error raised.
        "--resource-path=" + os.pathsep.join([str(HERE), str(HERE / "figures")]),
        "--from=latex+raw_tex",
        "--number-sections",
        "--wrap=preserve",
    ]
    if TEMPLATE.exists():
        args.append(f"--reference-doc={TEMPLATE}")
    with tempfile.TemporaryDirectory() as td:
        staged = Path(td) / "main.tex"
        staged.write_text(tex, encoding="utf-8")
        pypandoc.convert_file(str(staged), "docx", format="latex",
                              outputfile=str(OUT), extra_args=args)
    return OUT, postprocess(OUT, columns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--single-column", action="store_true")
    args = ap.parse_args()

    if not (HERE / "figures" / "fig1_dynamics.png").exists():
        raise SystemExit("PNG figures missing - run paper/make_figures.py first")
    if not TEMPLATE.exists():
        print(f"warning: {TEMPLATE.name} not found; falling back to pandoc "
              "defaults, which will not look like an IEEE submission")

    columns = 1 if args.single_column else 2
    out, report = build(columns)
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB, "
          f"{columns}-column, template={TEMPLATE.name if TEMPLATE.exists() else 'none'})")
    for k, v in sorted(report.get("styles", {}).items()):
        print(f"  restyled {v:3d}x  {k}")
    print(f"  single-column front matter: {report.get('split')}")
    print("The .tex stays the source of record: regenerate rather than "
          "editing the .docx.")


if __name__ == "__main__":
    main()
