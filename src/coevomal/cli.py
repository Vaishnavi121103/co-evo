"""Command-line interface.

Examples
--------
Run a single co-evolution experiment from a config file::

    python -m coevomal run --config configs/smoke.yaml

Run the built-in default (synthetic, DQN, every-round/full-replay)::

    python -m coevomal run

Run the factorial policy sweep described in the roadmap::

    python -m coevomal factorial --config configs/factorial.yaml --out results/factorial
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coevomal.config import ExperimentConfig
from coevomal.orchestrator import CoEvolutionOrchestrator


def _cmd_run(args: argparse.Namespace) -> None:
    cfg = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.name:
        cfg.name = args.name
    orch = CoEvolutionOrchestrator(cfg, verbose=not args.quiet)
    orch.run()
    out = orch.save(args.out)
    summary = orch.result.summary(cfg.convergence_window, cfg.convergence_tol)
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"saved to: {out}")


def _cmd_factorial(args: argparse.Namespace) -> None:
    from experiments.run_factorial import run_factorial

    base = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    run_factorial(base, out_dir=Path(args.out), quiet=args.quiet)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="coevomal", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a single co-evolution experiment")
    r.add_argument("--config", type=str, default=None, help="path to a YAML config")
    r.add_argument("--name", type=str, default=None, help="override experiment name")
    r.add_argument("--rounds", type=int, default=None, help="override number of rounds")
    r.add_argument("--out", type=str, default=None, help="output directory")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=_cmd_run)

    f = sub.add_parser("factorial", help="run the factorial retraining-policy sweep")
    f.add_argument("--config", type=str, default=None, help="base YAML config")
    f.add_argument("--out", type=str, default="results/factorial")
    f.add_argument("--quiet", action="store_true")
    f.set_defaults(func=_cmd_factorial)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
