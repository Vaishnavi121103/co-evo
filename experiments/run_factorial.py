"""Factorial retraining-policy sweep -- the core experiment of the thesis.

Crosses the study's independent variables:

* **cadence**        x {every_round, every_n, threshold}
* **data_selection** x {full_replay, hard_mining, bounded_buffer}
* (secondary) **attacker** x {dqn, ppo}

Every cell is run through the *same* orchestrator and scored with the *same*
metrics, so differences in convergence / oscillation / cost are attributable
to the policy variable rather than to a bundled defense mechanism. Results
are written as a tidy CSV plus a robustness-vs-cost frontier plot.

Start small: the default grid below is a 3x3 cadence x data_selection sweep
with a single attacker, which is the reduced grid the roadmap recommends
before scaling up. Widen ``ATTACKERS`` / add axes once compute allows.
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

from coevomal.config import ExperimentConfig
from coevomal.orchestrator import CoEvolutionOrchestrator

CADENCES = ["every_round", "every_n", "threshold"]
DATA_SELECTION = ["full_replay", "hard_mining", "bounded_buffer"]
ATTACKERS = ["dqn"]  # add "ppo" for the secondary robustness axis


def run_factorial(base: ExperimentConfig, out_dir: Path, quiet: bool = False) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    combos = list(itertools.product(ATTACKERS, CADENCES, DATA_SELECTION))
    for i, (atk, cadence, selection) in enumerate(combos):
        name = f"{atk}__{cadence}__{selection}"
        cfg = base.replace(
            **{
                "name": name,
                "attacker.name": atk,
                "retrain.cadence": cadence,
                "retrain.data_selection": selection,
            }
        )
        print(f"\n===== [{i + 1}/{len(combos)}] {name} =====")
        orch = CoEvolutionOrchestrator(cfg, verbose=not quiet)
        orch.run()
        orch.save(out_dir)
        summ = orch.result.summary(cfg.convergence_window, cfg.convergence_tol)
        rows.append(
            {
                "policy": name,
                "attacker": atk,
                "cadence": cadence,
                "data_selection": selection,
                **summ,
            }
        )

    # ---- tidy CSV -----------------------------------------------------------
    csv_path = out_dir / "factorial_summary.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")

    # ---- frontier plot ------------------------------------------------------
    try:
        from coevomal.evaluation.plots import plot_frontier

        plot_frontier(rows, out_dir / "frontier.png")
        print(f"wrote {out_dir / 'frontier.png'}")
    except Exception as exc:  # best-effort
        print(f"[warn] frontier plot failed: {exc}")

    return csv_path


if __name__ == "__main__":
    run_factorial(ExperimentConfig(), out_dir=Path("results/factorial"))
