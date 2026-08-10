"""Command-line interface for RESP charge fitting."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from oligoternary.simulation.resp import (
    RespError,
    fit_resp_charges,
    load_resp_config,
    validate_resp_inputs,
    write_resp_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oligoternary-charge",
        description="Validate or fit multi-conformer two-stage RESP charges",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action, help_text in (
        ("validate", "validate atom mapping, geometries, grids, and ESP values"),
        ("fit", "fit RESP charges and write charges.csv"),
    ):
        command = subparsers.add_parser(action, help=help_text)
        command.add_argument("config", help="path to the RESP YAML file")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_resp_config(args.config)
        if args.action == "validate":
            print(json.dumps({"valid": True, **validate_resp_inputs(config)}, indent=2))
            return 0

        result = fit_resp_charges(config)
        charges_path, report_path = write_resp_result(config, result)
        print(
            json.dumps(
                {
                    "project": config.project,
                    "charges": str(charges_path),
                    "report": str(report_path),
                    "total_charge_e": float(result.stage2_charges.sum()),
                    "relative_rms_error": result.relative_rms_errors,
                },
                indent=2,
            )
        )
        return 0
    except (RespError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
