"""Command-line interface for Rosetta linker reconstruction."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Optional, Sequence


from oligoternary.modeling.constants import (
    ATTACHMENT_ENDPOINT_CANDIDATES,
    CONFORMER_RANKING,
    CONFORMER_RANKINGS,
    MIN_ATTACHMENT_ANGLE_DEGREES,
    PREPARE_CONFORMER_CANDIDATES,
)
from oligoternary.modeling.runtime import default_prepare_process_count
from oligoternary.cli.stage_result import add_stage_result_arguments, write_cli_result
from oligoternary.workflow.tools import ToolConfig, require_tools, resolve_tools


DEFAULT_MOLFILE2PARAMS = None


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Reconstruct and refine an OligoTernary linker with Rosetta",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode", type=str, choices=["single", "batch"], required=True
    )

    # Linker parameters
    parser.add_argument(
        "--molfile2params",
        type=str,
        default=DEFAULT_MOLFILE2PARAMS,
        help=(
            "Path to molfile_to_params.py. Discovery order: this option, "
            "MOLFILE_TO_PARAMS, --rosetta-root/ROSETTA_ROOT, PATH"
        ),
    )
    parser.add_argument(
        "--rosetta-root",
        dest="rosetta_root",
        help="Rosetta installation root used to discover molfile_to_params.py",
    )
    parser.add_argument("--linker-prefix", type=str, required=True)
    parser.add_argument("--linker-smiles", type=str, required=True)
    parser.add_argument("--linker-chain", type=str, required=True)

    # E3L / target parameters
    parser.add_argument("--e3l-smiles", type=str, required=True)
    parser.add_argument("--e3l-chain", type=str, required=True)
    parser.add_argument("--warhead-chain", type=str, required=True)
    parser.add_argument("--poi-chain", type=str, required=True)
    parser.add_argument("--e3-chain", type=str, required=True)
    parser.add_argument(
        "--warhead-anchor-label",
        type=str,
        required=True,
        help="Warhead anchor atom (format: chain:resnum:atomname)",
    )
    parser.add_argument(
        "--e3l-anchor-label",
        type=str,
        required=True,
        help="E3L anchor atom (format: chain:resnum:atomname)",
    )
    parser.add_argument(
        "--linker-to-warhead-atom",
        type=str,
        help=(
            "Optional linker connect atom name for the warhead-facing side. "
            "Use this when `[*]` marker order differs from runtime bond role order."
        ),
    )
    parser.add_argument(
        "--linker-to-e3l-atom",
        type=str,
        help=(
            "Optional linker connect atom name for the E3L-facing side. "
            "Use this when `[*]` marker order differs from runtime bond role order."
        ),
    )

    # Processing parameters
    parser.add_argument("--relax-cycles", type=int, default=5)
    parser.add_argument("--minimize-steps", type=int, default=1000)
    parser.add_argument("--minimize-tolerance", type=float, default=0.001)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Non-negative random seed used for linker length/conformer preparation",
    )
    parser.add_argument(
        "--endpoint-candidates",
        type=int,
        default=ATTACHMENT_ENDPOINT_CANDIDATES,
        help="Number of attachment endpoint placements to search",
    )
    parser.add_argument(
        "--conformers-per-endpoint",
        type=int,
        default=PREPARE_CONFORMER_CANDIDATES,
        help="Number of RDKit conformers sampled for each endpoint placement",
    )
    parser.add_argument(
        "--minimum-attachment-angle-degrees",
        type=float,
        default=MIN_ATTACHMENT_ANGLE_DEGREES,
        help="Minimum linker attachment angle in degrees; 0 disables this filter",
    )
    parser.add_argument(
        "--conformer-ranking",
        choices=CONFORMER_RANKINGS,
        default=CONFORMER_RANKING,
        help="Priority used to rank conformers after embedding",
    )

    # I/O
    parser.add_argument(
        "--pdb-file", type=str, help="Input PDB (single mode)"
    )
    parser.add_argument(
        "--pdb-files", type=str, nargs="+", help="Input PDBs/dirs (batch mode)"
    )
    parser.add_argument("--output-dir", type=str, required=True)
    add_stage_result_arguments(parser)
    parser.add_argument(
        "--num-processes", type=int, default=default_prepare_process_count()
    )
    parser.add_argument(
        "--rosetta-num-processes",
        type=int,
        default=24,
        help=(
            "PyRosetta stage concurrency in batch mode. Uses 'spawn' start "
            "method so each worker initializes PyRosetta independently."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow exit zero when at least one batch input succeeds",
    )

    return parser


def _failed_input_names(tsv_path: Optional[str]) -> list[str]:
    if not tsv_path:
        return []
    with open(tsv_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "input" not in reader.fieldnames:
            raise ValueError(f"failed input table has no 'input' column: {tsv_path}")
        return [row["input"] for row in reader if row.get("input")]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI main entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.relax_cycles < 1:
        parser.error("--relax-cycles must be >= 1")
    if args.minimize_steps < 1:
        parser.error("--minimize-steps must be >= 1")
    if args.minimize_tolerance <= 0:
        parser.error("--minimize-tolerance must be > 0")
    if args.endpoint_candidates < 1:
        parser.error("--endpoint-candidates must be >= 1")
    if args.conformers_per_endpoint < 1:
        parser.error("--conformers-per-endpoint must be >= 1")
    if not 0 <= args.minimum_attachment_angle_degrees <= 180:
        parser.error("--minimum-attachment-angle-degrees must be between 0 and 180")
    if args.num_processes < 1:
        parser.error("--num-processes must be >= 1")
    if args.rosetta_num_processes < 1:
        parser.error("--rosetta-num-processes must be >= 1")

    if args.mode == "single":
        if not args.pdb_file:
            parser.error("--pdb-file is required for single mode")
    elif not args.pdb_files:
        parser.error("--pdb-files is required for batch mode")

    try:
        tool_paths = require_tools(
            resolve_tools(
                ToolConfig(
                    molfile_to_params=args.molfile2params,
                    rosetta_root=args.rosetta_root,
                )
            ),
            ("molfile_to_params",),
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    # Importing the modeling implementation imports PyRosetta. Keep it after
    # argument and external-tool validation so discovery itself stays light.
    from oligoternary.modeling import LinkerModeler

    try:
        modeler = LinkerModeler(
            molfile2params=tool_paths["molfile_to_params"],
            linker_prefix=args.linker_prefix,
            linker_smiles=args.linker_smiles,
            linker_chain=args.linker_chain,
            e3l_smiles=args.e3l_smiles,
            e3l_chain=args.e3l_chain,
            warhead_chain=args.warhead_chain,
            poi_chain=args.poi_chain,
            e3_chain=args.e3_chain,
            warhead_anchor_label=args.warhead_anchor_label,
            e3l_anchor_label=args.e3l_anchor_label,
            linker_to_warhead_atom=args.linker_to_warhead_atom,
            linker_to_e3l_atom=args.linker_to_e3l_atom,
            endpoint_candidates=args.endpoint_candidates,
            conformers_per_endpoint=args.conformers_per_endpoint,
            minimum_attachment_angle_degrees=args.minimum_attachment_angle_degrees,
            conformer_ranking=args.conformer_ranking,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.mode == "single":
        try:
            ok = modeler.run(
                pdb_file=args.pdb_file,
                output_dir=args.output_dir,
                relax_cycles=args.relax_cycles,
                minimize_steps=args.minimize_steps,
                minimize_tolerance=args.minimize_tolerance,
                random_seed=args.seed,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"linker refinement failed: {exc}", file=sys.stderr)
            return 1
        try:
            write_cli_result(
                args,
                total_count=1,
                succeeded_count=1 if ok else 0,
                failed_inputs=[] if ok else [os.path.basename(args.pdb_file)],
            )
        except (OSError, ValueError) as exc:
            print(f"cannot write Stage result summary: {exc}", file=sys.stderr)
            return 2
        return 0 if ok else 1

    try:
        summary = modeler.run_batch(
            pdb_files=args.pdb_files,
            output_dir=args.output_dir,
            num_processes=args.num_processes,
            rosetta_num_processes=args.rosetta_num_processes,
            relax_cycles=args.relax_cycles,
            minimize_steps=args.minimize_steps,
            minimize_tolerance=args.minimize_tolerance,
            strict=not args.allow_partial,
            random_seed=args.seed,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"linker refinement failed: {exc}", file=sys.stderr)
        return 1
    try:
        failed_inputs = _failed_input_names(summary.failed_inputs_tsv)
        write_cli_result(
            args,
            total_count=summary.total_inputs,
            succeeded_count=summary.succeeded_count,
            failed_inputs=failed_inputs,
        )
    except (OSError, ValueError) as exc:
        print(f"cannot write Stage result summary: {exc}", file=sys.stderr)
        return 2
    return 0 if summary.exit_ok(strict=not args.allow_partial) else 1


if __name__ == "__main__":
    sys.exit(main())
