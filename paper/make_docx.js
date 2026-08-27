/**
 * Build the Word version of the paper in JavaScript.
 *
 * A LaTeX -> docx converter written directly against the OOXML model, rather
 * than shelling out to pandoc. It reads `main.tex` and the IEEE template and
 * emits `main-js.docx`.
 *
 * How it keeps the IEEE look: the template's `styles.xml` is lifted out of
 * `conference-template-a4.docx` and handed to the `docx` library as
 * `externalStyles`, so paragraphs can reference IEEE's own style ids
 * (`papertitle`, `Abstract`, `figurecaption`, ...) and inherit their exact
 * fonts and spacing. The page is set to A4 with a single-column front matter
 * section followed by a two-column body, which is how IEEE lays the format out.
 *
 * Known limitation, stated rather than hidden: maths is rendered as Unicode
 * text, not as native Word equations. The pandoc route (`make_docx.py`)
 * produces real OMML equation objects and is the better choice when the maths
 * matters. In this paper the maths is simple enough --- superscripts,
 * subscripts, Greek letters and inequalities --- that Unicode is faithful, but
 * a reader who wants to *edit* equations in Word should use the Python route.
 *
 * Usage:
 *   node paper/make_docx.js
 *   node paper/make_docx.js --single-column
 */

const fs = require("fs");
const path = require("path");
const JSZip = require("jszip");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, SectionType, BorderStyle,
} = require("docx");

const HERE = __dirname;
const TEX = path.join(HERE, "main.tex");
const TEMPLATE = path.join(HERE, "conference-template-a4.docx");
const FIGS = path.join(HERE, "figures");
const OUT = path.join(HERE, "main-js.docx");

/* ------------------------------------------------------------------ maths */
// LaTeX fragments -> Unicode. Applied to inline maths and to running text.
const SYMBOLS = [
  [/\\times/g, "×"], [/\\pm/g, "±"], [/\\leq/g, "≤"],
  [/\\geq/g, "≥"], [/\\approx/g, "≈"], [/\\ldots/g, "…"],
  [/\\dots/g, "…"], [/\\rightarrow/g, "→"], [/\\omega/g, "ω"],
  [/\\Omega/g, "Ω"], [/\\osc/g, "Ω"], [/\\evrate/g, "E_{tail}"],
  [/\\sim/g, "~"],
  [/\\%/g, "%"], [/\\&/g, "&"], [/\\_/g, "_"], [/\\#/g, "#"],
  [/---/g, "—"], [/--/g, "–"], [/``/g, "“"], [/''/g, "”"],
];
const SUP = { "0":"⁰","1":"¹","2":"²","3":"³","4":"⁴",
  "5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹",
  "-":"⁻","−":"⁻","+":"⁺" };
const SUB = { "0":"₀","1":"₁","2":"₂","3":"₃","4":"₄",
  "5":"₅","6":"₆","7":"₇","8":"₈","9":"₉" };

function toUnicodeScript(s, table) {
  return [...s].every((c) => table[c]) ? [...s].map((c) => table[c]).join("") : null;
}

/** Render one inline-maths body (the text between `$`) as Unicode. */
function renderMath(m) {
  let s = m;
  for (const [re, to] of SYMBOLS) s = s.replace(re, to);
  s = s.replace(/\\mathrm\{([^}]*)\}/g, "$1").replace(/\\emph\{([^}]*)\}/g, "$1");
  s = s.replace(/\\text\{([^}]*)\}/g, "$1");
  // E_{\mathrm{tail}} -> E_tail ; x^{2} -> superscript where representable
  s = s.replace(/\^\{([^}]*)\}|\^(\S)/g, (_, a, b) => {
    const t = a !== undefined ? a : b;
    return toUnicodeScript(t, SUP) || "^" + t;
  });
  s = s.replace(/_\{([^}]*)\}|_(\S)/g, (_, a, b) => {
    const t = a !== undefined ? a : b;
    return toUnicodeScript(t, SUB) || "_" + t;
  });
  return s.replace(/[{}]/g, "").replace(/\\,/g, " ").trim();
}

/* ------------------------------------------------------- inline formatting */
/**
 * Turn a LaTeX paragraph into styled runs.
 * Handles \textbf, \emph/\textit, \texttt, \cite and inline maths; everything
 * else is flattened to plain text rather than dropped silently.
 */
// label -> the number its float actually carries, filled in before conversion.
const LABELS = {};

function runs(text, refs) {
  const out = [];
  // Order matters: maths first, so a `$...$` containing braces is not eaten
  // by the command matcher below.
  const re = /\$([^$]*)\$|\\(textbf|emph|textit|texttt)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}|\\cite\{([^}]*)\}|\\ref\{([^}]*)\}/g;
  let last = 0, m;
  const plain = (s) => {
    let t = s;
    for (const [r, to] of SYMBOLS) t = t.replace(r, to);
    return t.replace(/\\[a-zA-Z]+\s?/g, "").replace(/[{}]/g, "").replace(/~/g, " ");
  };
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun(plain(text.slice(last, m.index))));
    if (m[1] !== undefined) {
      out.push(new TextRun(renderMath(m[1])));
    } else if (m[2]) {
      const inner = plain(m[3]);
      if (m[2] === "textbf") out.push(new TextRun({ text: inner, bold: true }));
      else if (m[2] === "texttt") out.push(new TextRun({ text: inner, font: "Consolas" }));
      else out.push(new TextRun({ text: inner, italics: true }));
    } else if (m[4] !== undefined) {
      // Numeric citations, in the order the bibliography defines them.
      const nums = m[4].split(",").map((k) => refs.indexOf(k.trim()) + 1)
        .filter((n) => n > 0);
      out.push(new TextRun(`[${nums.join("], [")}]`));
    } else if (m[5] !== undefined) {
      // Resolve to the number the float actually carries; emitting the bare
      // label suffix would render "Fig. dynamics" in running text.
      out.push(new TextRun(LABELS[m[5]] || m[5].replace(/^(fig|tab|sec):/, "")));
    }
    last = re.lastIndex;
  }
  if (last < text.length) out.push(new TextRun(plain(text.slice(last))));
  return out.length ? out : [new TextRun("")];
}

/* ----------------------------------------------------------- tex utilities */
function braceArg(s, from) {
  // Return the balanced {...} argument starting at or after `from`.
  const start = s.indexOf("{", from);
  if (start === -1) return null;
  let depth = 0;
  for (let i = start; i < s.length; i++) {
    if (s[i] === "{") depth++;
    else if (s[i] === "}" && --depth === 0)
      return { body: s.slice(start + 1, i), end: i + 1 };
  }
  return null;
}

function environments(tex, name) {
  const out = [];
  const re = new RegExp(`\\\\begin\\{${name}\\}([\\s\\S]*?)\\\\end\\{${name}\\}`, "g");
  let m;
  while ((m = re.exec(tex)) !== null) out.push({ body: m[1], index: m.index });
  return out;
}

/* --------------------------------------------------------------- the build */
function buildTable(body, refs) {
  // Parse a tabular: strip rules, split on \\ then &.
  const inner = body.replace(/\\(top|mid|bottom)rule/g, "")
    .replace(/\\begin\{tabular\}\{[^}]*\}/, "")
    .replace(/\\end\{tabular\}/, "");
  const rows = inner.split(/\\\\/).map((r) => r.trim()).filter(Boolean);
  if (!rows.length) return null;
  const cells = rows.map((r) =>
    r.split("&").map((c) => c.replace(/\\multicolumn\{\d+\}\{[^}]*\}/, "").trim()));
  const width = Math.max(...cells.map((c) => c.length));
  const border = { style: BorderStyle.SINGLE, size: 2, color: "000000" };
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: cells.map((row, ri) => new TableRow({
      children: Array.from({ length: width }, (_, ci) => new TableCell({
        borders: ri === 0
          ? { top: border, bottom: border, left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE } }
          : { top: { style: BorderStyle.NONE }, bottom: ri === cells.length - 1 ? border : { style: BorderStyle.NONE }, left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE } },
        children: [new Paragraph({
          style: ri === 0 ? "tablehead" : "tablecopy",
          alignment: ci === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
          children: runs(row[ci] || "", refs),
        })],
      })),
    })),
  });
}

function convert(tex, refs, singleColumn) {
  const front = [];   // title, authors, abstract, keywords
  const body = [];

  // ---- front matter -------------------------------------------------------
  const title = braceArg(tex, tex.indexOf("\\title"));
  front.push(new Paragraph({
    style: "papertitle",
    children: runs((title ? title.body : "").replace(/\\\\/g, " "), refs),
  }));
  const authorBlock = braceArg(tex, tex.indexOf("\\author"));
  if (authorBlock) {
    const name = /\\IEEEauthorblockN\{([^}]*)\}/.exec(authorBlock.body);
    const aff = /\\IEEEauthorblockA\{([\s\S]*?)\}\s*\}?\s*$/.exec(authorBlock.body);
    if (name) front.push(new Paragraph({ style: "Author", children: runs(name[1], refs) }));
    if (aff) {
      aff[1].split(/\\\\/).map((l) => l.trim()).filter(Boolean).forEach((l) =>
        front.push(new Paragraph({ style: "Affiliation", children: runs(l, refs) })));
    }
  }
  const abs = environments(tex, "abstract")[0];
  if (abs) {
    front.push(new Paragraph({
      style: "Abstract",
      children: [new TextRun({ text: "Abstract—", bold: true, italics: true }),
        ...runs(abs.body.trim(), refs)],
    }));
  }
  const kw = environments(tex, "IEEEkeywords")[0];
  if (kw) {
    front.push(new Paragraph({
      style: "Keywords",
      children: [new TextRun({ text: "Index Terms—", bold: true, italics: true }),
        ...runs(kw.body.trim(), refs)],
    }));
  }

  // ---- body ---------------------------------------------------------------
  const start = tex.indexOf("\\section");
  const end = tex.indexOf("\\begin{thebibliography}");
  let src = tex.slice(start, end === -1 ? undefined : end);

  // Pull float environments out first, replacing them with placeholders so
  // that paragraph splitting does not tear them apart.
  const floats = [];
  src = src.replace(/\\begin\{(figure|table)\}[\s\S]*?\\end\{\1\}/g, (blk, kind) => {
    floats.push({ kind, blk });
    return `\n\n FLOAT${floats.length - 1} \n\n`;
  });

  for (const chunk of src.split(/\n\s*\n/)) {
    const t = chunk.trim();
    if (!t) continue;

    const ph = /^ FLOAT(\d+) $/.exec(t);
    if (ph) {
      const { kind, blk } = floats[+ph[1]];
      const cap = braceArg(blk, blk.indexOf("\\caption"));
      if (kind === "figure") {
        const img = /\\includegraphics\[[^\]]*\]\{([^}]+)\}/.exec(blk);
        if (img) {
          const file = path.join(FIGS, img[1] + ".png");
          if (fs.existsSync(file)) {
            // Column width at A4 with IEEE margins is ~3.4in; PNGs are 200dpi.
            body.push(new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new ImageRun({
                data: fs.readFileSync(file), type: "png",
                transformation: { width: 320, height: Math.round(320 * imgRatio(file)) },
              })],
            }));
          }
        }
        const n = body.filter((p) => p.__fig).length + 1;
        const p = new Paragraph({
          style: "figurecaption",
          children: [new TextRun({ text: `Fig. ${figCount(body)}. ` }),
            ...runs(cap ? cap.body : "", refs)],
        });
        p.__fig = true;
        body.push(p);
      } else {
        const p = new Paragraph({
          style: "tablehead",
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: `TABLE ${roman(tabCount(body))}. ` }),
            ...runs(cap ? cap.body : "", refs)],
        });
        p.__tab = true;
        body.push(p);
        const tbl = buildTable(blk, refs);
        if (tbl) body.push(tbl);
      }
      continue;
    }

    const sec = /^\\section\*?\{/.test(t) ? braceArg(t, 0) : null;
    if (sec) {
      body.push(new Paragraph({ style: "Heading1", children: runs(sec.body, refs) }));
      const rest = t.slice(sec.end).trim();
      if (rest) body.push(new Paragraph({ style: "BodyText", children: runs(rest, refs) }));
      continue;
    }
    const sub = /^\\subsection\*?\{/.test(t) ? braceArg(t, 0) : null;
    if (sub) {
      body.push(new Paragraph({ style: "Heading2", children: runs(sub.body, refs) }));
      const rest = t.slice(sub.end).trim();
      if (rest) body.push(new Paragraph({ style: "BodyText", children: runs(rest, refs) }));
      continue;
    }
    if (/^\\begin\{itemize\}/.test(t)) {
      t.replace(/\\begin\{itemize\}|\\end\{itemize\}/g, "")
        .split(/\\item/).map((i) => i.trim()).filter(Boolean)
        .forEach((i) => body.push(new Paragraph({
          style: "BodyText", bullet: { level: 0 }, children: runs(i, refs),
        })));
      continue;
    }
    if (/^\\(label|graphicspath|newcommand|maketitle)/.test(t)) continue;
    body.push(new Paragraph({ style: "BodyText", children: runs(t, refs) }));
  }

  // ---- references ---------------------------------------------------------
  body.push(new Paragraph({ style: "Heading1", children: [new TextRun("References")] }));
  const bib = environments(tex, "thebibliography")[0];
  if (bib) {
    const items = bib.body.split(/\\bibitem\{[^}]*\}/).slice(1);
    items.forEach((it, i) => body.push(new Paragraph({
      style: "references",
      children: [new TextRun(`[${i + 1}] `), ...runs(it.trim(), refs)],
    })));
  }
  return { front, body };
}

const figCount = (b) => b.filter((p) => p.__fig).length + 1;
const tabCount = (b) => b.filter((p) => p.__tab).length + 1;
const roman = (n) => ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII"][n] || String(n);

/** Aspect ratio from the PNG header, so figures are not distorted. */
function imgRatio(file) {
  const buf = fs.readFileSync(file);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  return h / w;
}

/* --------------------------------------------------------------------- main */
(async function main() {
  const singleColumn = process.argv.includes("--single-column");
  if (!fs.existsSync(TEX)) throw new Error(`missing ${TEX}`);
  if (!fs.existsSync(path.join(FIGS, "fig1_dynamics.png")))
    throw new Error("PNG figures missing - run paper/make_figures.py first");

  const tex = fs.readFileSync(TEX, "utf8");
  const refs = [...tex.matchAll(/\\bibitem\{([^}]*)\}/g)].map((m) => m[1]);

  // Inherit IEEE's paragraph styles so `papertitle`, `Abstract`, `Heading1`
  // and the rest resolve to the template's real formatting.
  let externalStyles;
  if (fs.existsSync(TEMPLATE)) {
    const zip = await JSZip.loadAsync(fs.readFileSync(TEMPLATE));
    externalStyles = await zip.file("word/styles.xml").async("string");
  } else {
    console.warn("warning: conference-template-a4.docx not found; output will "
      + "not carry IEEE formatting");
  }

  // Number the floats in source order, so \ref resolves the way LaTeX would.
  let nFig = 0, nTab = 0;
  for (const m of tex.matchAll(/\\begin\{(figure|table)\}[\s\S]*?\\end\{\1\}/g)) {
    const lbl = /\\label\{([^}]*)\}/.exec(m[0]);
    if (lbl) LABELS[lbl[1]] = m[1] === "figure" ? String(++nFig) : roman(++nTab);
  }
  // Section labels resolve to their section number.
  let nSec = 0;
  for (const m of tex.matchAll(/\\section\*?\{[^}]*\}\s*\n?\\label\{([^}]*)\}/g)) {
    LABELS[m[1]] = roman(++nSec);
  }

  const { front, body } = convert(tex, refs, singleColumn);

  // A4, IEEE margins. Front matter spans the page; the body is two columns.
  const page = { size: { width: 11906, height: 16838 },
    margin: { top: 720, right: 900, bottom: 720, left: 900 } };
  const sections = singleColumn
    ? [{ properties: { page }, children: [...front, ...body] }]
    : [
        { properties: { page, column: { count: 1 } }, children: front },
        { properties: { type: SectionType.CONTINUOUS, page,
            column: { count: 2, space: 360, equalWidth: true } }, children: body },
      ];

  const doc = new Document({ externalStyles, sections });
  fs.writeFileSync(OUT, await Packer.toBuffer(doc));

  const kb = (fs.statSync(OUT).size / 1024).toFixed(0);
  console.log(`wrote ${OUT}  (${kb} KB, ${singleColumn ? 1 : 2}-column, `
    + `template=${externalStyles ? "conference-template-a4.docx" : "none"})`);
  console.log(`  front matter: ${front.length} paragraphs`);
  console.log(`  body: ${body.length} blocks, ${refs.length} references`);
  console.log("  note: maths is Unicode text, not native Word equations - use "
    + "make_docx.py if the equations must be editable in Word.");
})().catch((e) => { console.error(e); process.exit(1); });
