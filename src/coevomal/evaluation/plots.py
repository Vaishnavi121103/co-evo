"""Plotting helpers for co-evolution results (matplotlib, headless-safe)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display required
import matplotlib.pyplot as plt  # noqa: E402

from coevomal.evaluation.metrics import ExperimentResult  # noqa: E402


def plot_run(result: ExperimentResult, out_path: str | Path, title: str = "") -> Path:
    """Plot evasion rate, attacker query complexity and clean accuracy."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rounds = [r.round for r in result.rounds]
    eva = result.evasion_rates
    asr = [r.attack_success_rate for r in result.rounds]
    pre = [r.pre_evasive_rate for r in result.rounds]
    queries = result.mean_queries
    clean = [r.clean_accuracy for r in result.rounds]
    retrain_rounds = [r.round for r in result.rounds if r.retrained]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Grey bands mark rounds where the defender actually retrained, so the
    # cadence policy is readable straight off the plot.
    for rr in retrain_rounds:
        ax1.axvline(rr, color="grey", alpha=0.15)
    ax1.plot(rounds, eva, marker="o", label="evasion rate")
    # Separating attack success from the defender's pre-existing false
    # negatives keeps the attacker from being credited with the classifier's
    # own misses.
    ax1.plot(rounds, asr, marker="v", label="attack success (of samples caught)")
    ax1.plot(rounds, pre, marker="x", linestyle=":", alpha=0.7,
             label="pre-evasive (defender FN)")
    ax1.plot(rounds, clean, marker="s", linestyle="--", label="clean accuracy")
    ax1.set_ylabel("rate")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="center right", fontsize=8)
    ax1.set_title(title or "Co-evolution dynamics")

    ax2.plot(rounds, queries, marker="^", color="tab:red", label="mean queries/sample")
    ax2.set_xlabel("round")
    ax2.set_ylabel("attacker queries")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_frontier(rows, out_path: str | Path) -> Path:
    """Scatter the robustness-vs-cost frontier across policies.

    ``rows`` is an iterable of dicts with keys ``policy``,
    ``total_retrain_seconds`` and ``mean_evasion_tail``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        x = row["total_retrain_seconds"]
        y = row["mean_evasion_tail"]
        ax.scatter(x, y)
        ax.annotate(row["policy"], (x, y), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("total retraining cost (seconds)")
    ax.set_ylabel("settled evasion rate (lower = more robust)")
    ax.set_title("Robustness-vs-retraining-cost frontier")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
