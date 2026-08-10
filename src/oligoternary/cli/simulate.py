"""Command-line interface for Amber molecular-dynamics runs."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from oligoternary.simulation import (
    SimulationError,
    load_simulation_config,
    run_simulation,
    validate_simulation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oligoternary-simulate",
        description="Validate or run an Amber molecular-dynamics workflow",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate the configuration and Amber input files"
    )
    validate_parser.add_argument("config", help="path to the simulation YAML file")

    run_parser = subparsers.add_parser("run", help="run all Amber stages in order")
    run_parser.add_argument("config", help="path to the simulation YAML file")
    run_parser.add_argument(
        "--dry-run", action="store_true", help="print commands without writing files"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_simulation_config(args.config)
        if args.action == "validate":
            atom_count = validate_simulation(config)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "project": config.project,
                        "atom_count": atom_count,
                        "stages": [stage.name for stage in config.stages],
                    },
                    indent=2,
                )
            )
            return 0

        plans = run_simulation(config, dry_run=args.dry_run)
        print(
            json.dumps(
                {
                    "project": config.project,
                    "dry_run": args.dry_run,
                    "stages": [plan.to_dict() for plan in plans],
                },
                indent=2,
            )
        )
        return 0
    except SimulationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
