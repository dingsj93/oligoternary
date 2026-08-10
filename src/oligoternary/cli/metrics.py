"""Command-line interface for structural metrics."""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional

from oligoternary.cli.stage_result import add_stage_result_arguments, write_cli_result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Batch analysis of PROTAC ternary complex PDB files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  oligoternary-metrics -i models -o metrics.csv --poi-chain A \
    --warhead-chain B --e3-chain C --protac-chains B L X \
    -p params/LNK.params params/E3L.params
        """,
    )
    parser.add_argument("-i", "--input-folder", required=True, help="Folder of PDB files")
    parser.add_argument("-o", "--output-csv", required=True, help="Output CSV path")
    parser.add_argument("--poi-chain", required=True, help="POI chain ID")
    parser.add_argument("--warhead-chain", required=True, help="Warhead chain ID")
    parser.add_argument("--e3-chain", required=True, help="E3 ligase chain ID")
    parser.add_argument(
        "--protac-chains", nargs="+", required=True,
        help="PROTAC component chains",
    )
    parser.add_argument(
        "-p", "--params-files", nargs="*", default=[],
        help="PyRosetta parameter files for non-standard residues",
    )
    parser.add_argument("--score-function", default="ref2015", help="Rosetta score function")
    parser.add_argument("--probe-radius", type=float, default=1.4, help="SASA probe radius")
    parser.add_argument("--num-processes", type=int, default=1, help="Worker process count")
    parser.add_argument("--seed", type=int, default=42, help="Non-negative PyRosetta random seed")
    parser.add_argument("--linker-chain", default=None, help="Linker chain ID (e.g. L)")
    parser.add_argument("--linker-to-warhead-atom", default=None, help="Linker warhead connect atom")
    parser.add_argument("--linker-to-e3l-atom", default=None, help="Linker E3L connect atom")
    parser.add_argument("--warhead-anchor-label", default=None, help="Warhead anchor label")
    parser.add_argument("--e3l-anchor-label", default=None, help="E3L anchor label")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow exit zero when at least one input PDB is analyzed successfully",
    )
    add_stage_result_arguments(parser)
    return parser.parse_args(argv)


def build_linker_geometry_kwargs(args) -> Optional[Dict[str, Any]]:
    keys = (
        args.linker_chain,
        args.linker_to_warhead_atom,
        args.linker_to_e3l_atom,
        args.warhead_anchor_label,
        args.e3l_anchor_label,
    )
    n_set = sum(1 for k in keys if k)
    if n_set == 0:
        return None
    if n_set != len(keys):
        raise ValueError(
            "--linker-chain / --linker-to-warhead-atom / --linker-to-e3l-atom / "
            "--warhead-anchor-label / --e3l-anchor-label must be all set or all omitted "
            f"({n_set}/{len(keys)} provided)"
        )
    return dict(
        linker_chain=args.linker_chain,
        linker_to_warhead_atom=args.linker_to_warhead_atom,
        warhead_anchor_label=args.warhead_anchor_label,
        linker_to_e3l_atom=args.linker_to_e3l_atom,
        e3l_anchor_label=args.e3l_anchor_label,
    )


def main() -> int:
    args = parse_arguments()
    if args.seed < 0:
        print("--seed must be >= 0", file=sys.stderr)
        return 2
    if args.num_processes < 1:
        print("--num-processes must be >= 1", file=sys.stderr)
        return 2
    # Keep --help and config inspection usable without importing PyRosetta.
    from oligoternary.analysis.metrics import BatchAnalyzer

    try:
        linker_geometry_kwargs = build_linker_geometry_kwargs(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    batch_analyzer = BatchAnalyzer(args.score_function, args.probe_radius)
    exit_code = batch_analyzer.process_pdb_folder(
        args.input_folder,
        args.output_csv,
        args.poi_chain,
        args.warhead_chain,
        args.e3_chain,
        args.protac_chains,
        args.params_files,
        args.num_processes,
        linker_geometry_kwargs=linker_geometry_kwargs,
        strict=not args.allow_partial,
        random_seed=args.seed,
    )
    try:
        write_cli_result(
            args,
            total_count=batch_analyzer.last_total_count,
            succeeded_count=batch_analyzer.last_succeeded_count,
            failed_inputs=batch_analyzer.last_failed_inputs,
        )
    except (OSError, ValueError) as exc:
        print(f"Cannot write Stage result summary: {exc}", file=sys.stderr)
        return 2
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
