"""Shared result-summary options for stage CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from oligoternary.workflow.artifacts import ArtifactSpec, write_result_summary


def add_stage_result_arguments(parser: argparse.ArgumentParser) -> None:
    """Add result path and declared Artifact bindings to a Stage parser."""

    parser.add_argument(
        "--result-summary",
        help="Write the normalized Stage result JSON consumed by the workflow",
    )
    parser.add_argument(
        "--result-artifact",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help=(
            "Bind one declared output Artifact into the result summary; repeat for "
            "multiple Artifacts"
        ),
    )


def parse_artifact_specs(values: Sequence[str]) -> list[ArtifactSpec]:
    """Parse repeatable ROLE=PATH declarations."""

    artifacts = []
    roles = set()
    paths = set()
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not role or not raw_path:
            raise ValueError(f"invalid --result-artifact {value!r}; expected ROLE=PATH")
        if role in roles:
            raise ValueError(f"duplicate --result-artifact role: {role}")
        resolved_path = Path(raw_path).expanduser().resolve()
        if resolved_path in paths:
            raise ValueError(f"duplicate --result-artifact path: {resolved_path}")
        artifacts.append(ArtifactSpec(role=role, path=resolved_path))
        roles.add(role)
        paths.add(resolved_path)
    return artifacts


def write_cli_result(
    args,
    *,
    total_count: int,
    succeeded_count: int,
    failed_inputs: Iterable[str],
) -> Path | None:
    """Write a stage result when requested."""

    if not args.result_summary:
        if args.result_artifact:
            raise ValueError("--result-artifact requires --result-summary")
        return None
    artifacts = parse_artifact_specs(args.result_artifact)
    if not artifacts:
        raise ValueError("--result-summary requires at least one --result-artifact ROLE=PATH")
    return write_result_summary(
        Path(args.result_summary),
        artifacts=artifacts,
        total_count=total_count,
        succeeded_count=succeeded_count,
        failed_inputs=list(failed_inputs),
    )
