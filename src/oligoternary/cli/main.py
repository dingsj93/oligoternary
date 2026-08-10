"""Installed command-line interface for workflow validation and execution."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from ..workflow.config import ConfigError, load_config
from ..workflow.runner import WorkflowError, WorkflowRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oligoternary",
        description="Validate or run a configuration-driven OligoTernary workflow",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate a YAML or JSON run specification"
    )
    validate_parser.add_argument("config", help="path to workflow config")

    run_parser = subparsers.add_parser("run", help="run a validated workflow")
    run_parser.add_argument("config", help="path to workflow config")
    run_parser.add_argument(
        "--dry-run", action="store_true", help="print the stage plan without writing outputs"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.action == "validate":
            print(
                json.dumps(
                    {
                        "valid": True,
                        "project": config.project,
                        "stages": [stage.name for stage in config.stages],
                    },
                    indent=2,
                )
            )
            return 0

        manifest = WorkflowRunner(config).run(dry_run=args.dry_run)
        print(json.dumps(manifest.to_dict(), indent=2))
        return 0 if manifest.overall_status in {"succeeded", "incomplete"} else 1
    except (ConfigError, WorkflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
