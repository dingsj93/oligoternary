"""Persistent run-state implementation for the workflow module."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .artifacts import ArtifactRecord, StageResultSummary


STAGE_STATUSES = {"pending", "running", "succeeded", "failed", "skipped"}
RUN_STATUSES = {"pending", "running", "succeeded", "failed", "incomplete"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageState:
    name: str
    status: str
    adapter: str
    artifact: str
    depends_on: List[str]
    artifacts: List[ArtifactRecord] = field(default_factory=list)
    result_summary: Optional[StageResultSummary] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    return_code: Optional[int] = None
    message: Optional[str] = None
    command: List[str] = field(default_factory=list)
    cwd: Optional[str] = None
    stdout_log: Optional[str] = None
    stderr_log: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in STAGE_STATUSES:
            raise ValueError(f"invalid stage status: {self.status}")


@dataclass
class RunManifest:
    schema_version: int
    project: str
    config_path: str
    output_dir: str
    overall_status: str
    started_at: str
    updated_at: str
    dry_run: bool
    stages: List[StageState]
    provenance: Dict[str, Any] = field(default_factory=dict)
    run_id: str = ""

    def __post_init__(self) -> None:
        if self.overall_status not in RUN_STATUSES:
            raise ValueError(f"invalid run status: {self.overall_status}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_manifest(
    path: Path, manifest: RunManifest, *, update_timestamp: bool = True
) -> None:
    """Atomically replace the manifest so an interrupted write is not mistaken as state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if update_timestamp:
        manifest.updated_at = utc_now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
