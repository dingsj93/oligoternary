"""Collect lightweight run metadata without importing scientific runtimes."""

from __future__ import annotations

import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable, Optional, Sequence

from oligoternary.workflow.tools import ToolResolution


DEFAULT_PACKAGE_NAMES = (
    "oligoternary",
    "PyYAML",
    "numpy",
    "scipy",
    "pandas",
    "biopython",
    "rdkit",
    "loguru",
    "pyrosetta",
)


def _package_versions(names: Sequence[str]) -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _tool_record(report: ToolResolution) -> dict[str, object]:
    return {
        "name": report.name,
        "resolved_path": report.resolved_path,
        "source": report.source,
        "ready": report.ready,
        "version_probe": report.version_probe,
        "diagnostic": report.diagnostic,
    }


def _conda_environment_name() -> Optional[str]:
    explicit = os.environ.get("CONDA_DEFAULT_ENV")
    if explicit:
        return explicit
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        return Path(prefix).name
    interpreter_prefix = Path(sys.prefix)
    if interpreter_prefix.parent.name == "envs":
        return interpreter_prefix.name
    return None


def collect_provenance(
    config_path: Optional[Path],
    *,
    tool_reports: Iterable[ToolResolution] = (),
    package_names: Sequence[str] = DEFAULT_PACKAGE_NAMES,
) -> dict[str, object]:
    """Return JSON-safe metadata describing the workflow runtime."""

    run_specification = (
        str(Path(config_path).expanduser().resolve())
        if config_path is not None
        else None
    )
    return {
        "schema_version": 1,
        "run_specification": run_specification,
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
        },
        "environment": {
            "conda": _conda_environment_name(),
            "platform": platform.platform(),
        },
        "packages": _package_versions(package_names),
        "external_tools": [_tool_record(report) for report in tool_reports],
    }
