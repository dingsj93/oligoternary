"""Command-line E2 active-site accessibility screening."""

from __future__ import annotations

import argparse
import sys

from oligoternary.cli.stage_result import add_stage_result_arguments, write_cli_result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Align an E3-E2 reference and screen E2 active-site accessibility"
        )
    )
    parser.add_argument(
        "--input-pdb", required=True, help="Prepared ternary-complex PDB"
    )
    parser.add_argument(
        "--reference-pdb",
        required=True,
        help="Catalytic E3-E2 reference PDB or PDB.GZ",
    )
    parser.add_argument("--output-json", required=True, help="Screen result JSON")
    parser.add_argument("--poi-chain", required=True, help="POI chain in the input PDB")
    parser.add_argument("--e3-chain", required=True, help="E3 chain in the input PDB")
    parser.add_argument(
        "--reference-e3-chain",
        required=True,
        help="Matching E3 chain in the reference",
    )
    parser.add_argument("--e2-chain", required=True, help="E2 chain in the reference")
    parser.add_argument(
        "--e2-residue", required=True, type=int, help="Catalytic Cys number"
    )
    parser.add_argument("--e2-atom", default="SG", help="Catalytic atom name (default: SG)")
    parser.add_argument(
        "--minimum-alignment-residues",
        type=int,
        default=30,
        help="Minimum aligned E3 C-alpha residues (default: 30)",
    )
    parser.add_argument(
        "--maximum-alignment-rmsd",
        type=float,
        default=3.0,
        help="Maximum E3 alignment RMSD in A (default: 3.0)",
    )
    parser.add_argument(
        "--min-lysine-distance",
        type=float,
        default=0.0,
        help="Lower Lys NZ-to-active-site distance in A (default: 0)",
    )
    parser.add_argument(
        "--max-lysine-distance",
        type=float,
        default=25.0,
        help="Exclusive upper Lys NZ-to-active-site distance in A (default: 25)",
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=2.0,
        help="POI-E2 heavy-atom contact cutoff in A (default: 2.0)",
    )
    parser.add_argument(
        "--severe-cutoff",
        type=float,
        default=1.5,
        help="POI-E2 severe-clash cutoff in A (default: 1.5)",
    )
    parser.add_argument(
        "--minimum-separation",
        type=float,
        default=3.0,
        help="Required POI-E2 minimum heavy-atom distance in A (default: 3.0)",
    )
    parser.add_argument(
        "--maximum-contacts",
        type=int,
        default=5,
        help="Reject this many or more contacts (default: 5)",
    )
    add_stage_result_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_arguments(argv)
    if not 0 <= args.min_lysine_distance < args.max_lysine_distance:
        print("Lysine distance limits must satisfy 0 <= minimum < maximum", file=sys.stderr)
        return 2
    if min(args.contact_cutoff, args.severe_cutoff, args.minimum_separation) < 0:
        print("Steric distance cutoffs must be non-negative", file=sys.stderr)
        return 2
    if args.severe_cutoff > args.contact_cutoff:
        print("--severe-cutoff cannot exceed --contact-cutoff", file=sys.stderr)
        return 2
    if args.maximum_contacts < 1:
        print("--maximum-contacts must be at least 1", file=sys.stderr)
        return 2
    if args.minimum_alignment_residues < 3:
        print("--minimum-alignment-residues must be at least 3", file=sys.stderr)
        return 2
    if args.maximum_alignment_rmsd <= 0:
        print("--maximum-alignment-rmsd must be greater than 0", file=sys.stderr)
        return 2

    # Keep command discovery and --help usable without scientific dependencies.
    try:
        from oligoternary.analysis.e2_accessibility import (
            screen_e2_accessibility,
            write_e2_accessibility_result,
        )

        result = screen_e2_accessibility(
            args.input_pdb,
            args.reference_pdb,
            poi_chain=args.poi_chain,
            mobile_e3_chain=args.e3_chain,
            reference_e3_chain=args.reference_e3_chain,
            e2_chain=args.e2_chain,
            e2_residue=args.e2_residue,
            e2_atom=args.e2_atom,
            minimum_alignment_residues=args.minimum_alignment_residues,
            maximum_alignment_rmsd=args.maximum_alignment_rmsd,
            lysine_minimum_distance=args.min_lysine_distance,
            lysine_maximum_distance=args.max_lysine_distance,
            contact_cutoff=args.contact_cutoff,
            severe_cutoff=args.severe_cutoff,
            minimum_separation=args.minimum_separation,
            maximum_contacts=args.maximum_contacts,
        )
        write_e2_accessibility_result(result, args.output_json)
        write_cli_result(
            args,
            total_count=1,
            succeeded_count=int(result.passed),
            failed_inputs=[] if result.passed else [args.input_pdb],
        )
    except (ImportError, OSError, ValueError) as exc:
        print(f"E2 accessibility screen failed: {exc}", file=sys.stderr)
        return 2

    closest = result.lysines[0]
    status = "PASS" if result.passed else "FAIL"
    in_range_count = sum(lysine.in_range for lysine in result.lysines)
    print(
        f"{status}: {in_range_count}/{len(result.lysines)} lysines meet the "
        "E2 active-site accessibility criterion; "
        f"closest is Lys{closest.residue_number} at {closest.distance_angstrom:.2f} A; "
        f"E3 alignment is {result.alignment_rmsd_angstrom:.2f} A over "
        f"{result.alignment_residue_count} residues; "
        f"POI-E2 minimum distance is "
        f"{result.minimum_poi_e2_distance_angstrom:.2f} A"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
