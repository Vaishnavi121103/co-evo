"""Generate every paper figure from the real result files.

Reads only committed run artifacts under ``results/`` -- no synthetic or
hand-entered numbers anywhere -- and writes vector PDFs (for LaTeX) plus PNGs
(for slides and quick inspection) into ``paper/figures/``.

Styling targets IEEE two-column proceedings: Times-family serif to match the
body text, 8 pt labels, 3.5 in single-column and 7.16 in double-column widths,
so figures drop in at 100% scale without resampling text.

Any figure whose source study has not finished is skipped with a warning rather
than drawn from partial data, so the paper can never quietly contain a plot
averaged over an uneven number of seeds.

Usage
-----
    python paper/make_figures.py
    python paper/make_figures.py --formats pdf        # PDF only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"

COL_W, DBL_W = 3.5, 7.16                  # IEEE column widths, inches

# Colour-blind-safe categorical trio, validated for both light and print.
C_FULL, C_FIFO, C_HARD = "#2a78d6", "#eb6834", "#1baf7a"
C_INK, C_GRID, C_MUTED = "#111418", "#d9dee3", "#6b757d"
SEL_COLOR = {"full_replay": C_FULL, "bounded_buffer": C_FIFO, "hard_mining": C_HARD}
SEL_LABEL = {"full_replay": "full replay", "bounded_buffer": "bounded buffer (FIFO)",
             "hard_mining": "hard mining"}
CAD_LABEL = {"every_round": "every round", "every_n": "every 3rd round",
             "threshold": "threshold", "minimax": "minimax", "never": "never"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.edgecolor": C_INK, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.color": C_INK, "ytick.color": C_INK, "text.color": C_INK,
        "axes.labelcolor": C_INK, "grid.color": C_GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name: str, formats: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.{'/'.join(formats)}")


def complete(path: Path, expect: int | None = None) -> pd.DataFrame | None:
    """Load a sweep CSV, keeping only seeds where every cell finished."""
    if not path.exists():
        print(f"  [skip] {path.relative_to(ROOT)} missing")
        return None
    df = pd.read_csv(path)
    per = df.groupby("seed").size()
    keep = per[per == per.max()].index
    df = df[df.seed.isin(keep)]
    if expect is not None and len(df) < expect:
        print(f"  [skip] {path.relative_to(ROOT)} incomplete "
              f"({len(df)}/{expect} rows)")
        return None
    return df


def _grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- fig 1
def fig_dynamics(formats: list[str]) -> None:
    src = RES / "ember_pilot_v6" / "result.json"
    if not src.exists():
        print("  [skip] fig1: pilot result.json missing")
        return
    R = json.loads(src.read_text(encoding="utf-8"))["rounds"]
    r = [x["round"] for x in R]
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(COL_W, 3.0), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.18})

    a1.plot(r, [x["evasion_rate"] for x in R], "-o", ms=3, lw=1.3,
            color=C_FULL, label="evasion rate")
    a1.plot(r, [x["attack_success_rate"] for x in R], "-v", ms=3, lw=1.3,
            color=C_FIFO, label="attack success")
    a1.plot(r, [x["clean_accuracy"] for x in R], "--s", ms=3, lw=1.1,
            color=C_HARD, label="clean accuracy")
    a1.set_ylim(-0.03, 1.05)
    a1.set_ylabel("rate")
    a1.legend(loc="center right", handlelength=1.6)
    _grid(a1)

    a2.plot(r, [x["mean_queries"] for x in R], "-^", ms=3, lw=1.3, color=C_FIFO)
    a2.fill_between(r, [x["mean_queries"] for x in R], alpha=0.10, color=C_FIFO)
    a2.set_ylabel("queries\nper sample")
    a2.set_xlabel("co-evolution round")
    a2.set_xticks(r)
    _grid(a2)
    save(fig, "fig1_dynamics", formats)


# ---------------------------------------------------------------- fig 2
def fig_policies(formats: list[str]) -> None:
    df = complete(RES / "ember_multiseed" / "multiseed_raw.csv", 45)
    if df is None:
        return
    g = df.groupby(["cadence", "data_selection"]).mean_evasion_tail
    mean, sd = g.mean(), g.std(ddof=1)
    rows = sorted(mean.index, key=lambda k: mean[k])
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    y = np.arange(len(rows))
    ax.barh(y, [mean[k] for k in rows], xerr=[sd[k] for k in rows],
            color=[SEL_COLOR[k[1]] for k in rows], height=0.68,
            error_kw=dict(ecolor=C_MUTED, lw=0.7, capsize=1.8), zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{CAD_LABEL[c]} / {SEL_LABEL[s]}" for c, s in rows])
    ax.invert_yaxis()
    ax.set_xlabel("settled evasion rate (lower is better)")
    ax.set_xlim(0, max(mean) * 1.28)
    for i, k in enumerate(rows):
        ax.text(mean[k] + sd[k] + max(mean) * .02, i, f"{mean[k]:.3f}",
                va="center", fontsize=7)
    _grid(ax, "x")
    save(fig, "fig2_policies", formats)


# ---------------------------------------------------------------- fig 3
def fig_marginals(formats: list[str]) -> None:
    df = complete(RES / "ember_multiseed" / "multiseed_raw.csv", 45)
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W * 0.62, 2.0))
    for ax, factor, labels, title in (
            (axes[0], "data_selection", SEL_LABEL, "retention policy"),
            (axes[1], "cadence", CAD_LABEL, "cadence")):
        g = df.groupby(factor).mean_evasion_tail
        mean, sd = g.mean(), g.std(ddof=1)
        order = sorted(mean.index, key=lambda k: mean[k])
        x = np.arange(len(order))
        cols = ([SEL_COLOR[k] for k in order] if factor == "data_selection"
                else [C_MUTED] * len(order))
        ax.bar(x, [mean[k] for k in order], yerr=[sd[k] for k in order],
               color=cols, width=0.6,
               error_kw=dict(ecolor=C_INK, lw=0.7, capsize=2), zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[k] for k in order], rotation=18, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 0.46)
        for i, k in enumerate(order):
            ax.text(i, mean[k] + sd[k] + .012, f"{mean[k]:.3f}",
                    ha="center", fontsize=6.8)
        _grid(ax)
    axes[0].set_ylabel("settled evasion rate")
    save(fig, "fig3_marginals", formats)


# ---------------------------------------------------------------- fig 4
def fig_capacity(formats: list[str]) -> None:
    """The capacity gradient: separates retention size from eviction rule."""
    c8 = complete(RES / "ember_cap800" / "multiseed_raw.csv", 10)
    main = complete(RES / "ember_multiseed" / "multiseed_raw.csv", 45)
    if c8 is None or main is None:
        return
    er = main[main.cadence == "every_round"]
    fr = er[er.data_selection == "full_replay"].mean_evasion_tail

    caps = [200, 800]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for sel, col in (("bounded_buffer", C_FIFO), ("hard_mining", C_HARD)):
        m, s = [], []
        for cap, src in zip(caps, (er, c8)):
            v = src[src.data_selection == sel].mean_evasion_tail
            m.append(v.mean())
            s.append(v.std(ddof=1))
        ax.errorbar(caps, m, yerr=s, marker="o", ms=4, lw=1.3, capsize=2,
                    color=col, label=SEL_LABEL[sel], zorder=3)
    ax.axhline(fr.mean(), ls="--", lw=1.0, color=C_FULL, zorder=2)
    ax.fill_between([150, 900], fr.mean() - fr.std(ddof=1),
                    fr.mean() + fr.std(ddof=1), color=C_FULL, alpha=.12, zorder=1)
    ax.text(860, fr.mean() + .022, "unbounded retention\n(full replay)",
            ha="right", fontsize=6.8, color=C_FULL)
    ax.set_xscale("log")
    ax.set_xticks(caps)
    ax.set_xticklabels([str(c) for c in caps])
    ax.set_xlim(150, 900)
    ax.set_xlabel("adversarial replay capacity (samples retained)")
    ax.set_ylabel("settled evasion rate")
    ax.legend(loc="upper right", handlelength=1.8)
    _grid(ax)
    save(fig, "fig4_capacity", formats)


# ---------------------------------------------------------------- fig 5
def fig_frontier(formats: list[str]) -> None:
    main = complete(RES / "ember_multiseed" / "multiseed_raw.csv", 45)
    if main is None:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    g = main.groupby(["cadence", "data_selection"])
    for (cad, sel), s in g:
        ax.scatter(s.total_retrain_seconds.mean(), s.mean_evasion_tail.mean(),
                   s=26, color=SEL_COLOR[sel], edgecolor="white", lw=.6, zorder=3)
    for name, path, mk in (("minimax", RES / "ember_minimax" / "multiseed_raw.csv", "D"),
                           ("frozen", RES / "ember_baseline_frozen" / "multiseed_raw.csv", "s")):
        d = complete(path, 5)
        if d is None:
            continue
        ax.scatter(d.total_retrain_seconds.mean(), d.mean_evasion_tail.mean(),
                   s=30, marker=mk, facecolor="white", edgecolor=C_INK,
                   lw=1.0, zorder=4)
        ax.annotate(name, (d.total_retrain_seconds.mean(), d.mean_evasion_tail.mean()),
                    textcoords="offset points", xytext=(6, -1), fontsize=7)
    best = g.mean_evasion_tail.mean().idxmin()
    bx = g.total_retrain_seconds.mean()[best]
    by = g.mean_evasion_tail.mean()[best]
    ax.annotate("best", (bx, by), textcoords="offset points", xytext=(6, 2), fontsize=7)
    ax.set_xlabel("total retraining cost (s)")
    ax.set_ylabel("settled evasion rate")
    handles = [plt.Line2D([], [], marker="o", ls="", color=SEL_COLOR[k],
                          label=SEL_LABEL[k]) for k in SEL_LABEL]
    handles += [plt.Line2D([], [], marker="s", ls="", mfc="white", mec=C_INK,
                           label="baselines")]
    ax.legend(handles=handles, loc="center right", handlelength=1.2)
    _grid(ax, "both")
    save(fig, "fig5_frontier", formats)


# ---------------------------------------------------------------- fig 6
def fig_mode(formats: list[str]) -> None:
    ft = complete(RES / "ember_mode_axis" / "multiseed_raw.csv", 45)
    main = complete(RES / "ember_multiseed" / "multiseed_raw.csv", 45)
    if ft is None or main is None:
        return
    sels = ["full_replay", "bounded_buffer", "hard_mining"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    x = np.arange(len(sels))
    w = 0.36
    for off, d, lab, hatch in ((-w / 2, main, "retrain from scratch", None),
                               (w / 2, ft, "fine-tune", "///")):
        m = [d[d.data_selection == s].mean_evasion_tail.mean() for s in sels]
        e = [d[d.data_selection == s].mean_evasion_tail.std(ddof=1) for s in sels]
        ax.bar(x + off, m, w, yerr=e, label=lab, hatch=hatch,
               color=[SEL_COLOR[s] for s in sels],
               edgecolor="white", lw=.5,
               error_kw=dict(ecolor=C_INK, lw=.7, capsize=2), zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([SEL_LABEL[s] for s in sels], rotation=12, ha="right")
    ax.set_ylabel("settled evasion rate")
    handles = [plt.Line2D([], [], marker="s", ls="", color=C_MUTED, label="from scratch"),
               plt.Line2D([], [], marker="s", ls="", mfc="white", mec=C_MUTED,
                          label="fine-tune")]
    ax.legend(handles=handles, loc="upper left")
    _grid(ax)
    save(fig, "fig6_mode", formats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formats", default="pdf,png")
    args = ap.parse_args()
    formats = args.formats.split(",")
    style()
    print("generating figures from real result files ->", OUT)
    for fn in (fig_dynamics, fig_policies, fig_marginals,
               fig_capacity, fig_frontier, fig_mode):
        fn(formats)
    print("done")


if __name__ == "__main__":
    main()
