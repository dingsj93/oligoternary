"""Check the Rosetta helper required by linker reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Optional, Sequence

from oligoternary.workflow.tools import ToolConfig, preflight_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve and check Rosetta molfile_to_params.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--molfile2params",
        help="Rosetta molfile_to_params.py (overrides environment and Rosetta root)",
    )
    parser.add_argument("--rosetta-root", help="Rosetta installation root")
    parser.add_argument("--no-probe", action="store_true", help="Skip version/help commands")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = ToolConfig(
        molfile_to_params=args.molfile2params,
        rosetta_root=args.rosetta_root,
    )
    reports = preflight_tools(
        config,
        probe=not args.no_probe,
    )

    if args.json:
        print(json.dumps([asdict(report) for report in reports], indent=2))
    else:
        for report in reports:
            print(f"[{report.name}] {'READY' if report.ready else 'MISSING'}")
            print(f"  resolved_path: {report.resolved_path or '-'}")
            print(f"  source: {report.source}")
            print(f"  exists: {report.exists}")
            print(f"  executable: {report.executable}")
            print(f"  version_probe: {report.version_probe or 'not run'}")
            print(f"  diagnostic: {report.diagnostic}")

    return 0 if all(report.ready for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
