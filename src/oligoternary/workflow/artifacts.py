"""Artifact records and stage result summaries."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence


_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".params": "text/plain",
    ".pdb": "chemical/x-pdb",
    ".sdf": "chemical/x-mdl-sdfile",
    ".tsv": "text/tab-separated-values",
}
_DIRECTORY_MEDIA_TYPE = "application/vnd.oligoternary.directory"
STAGE_NAME_ENV = "OLIGOTERNARY_STAGE_NAME"
RUN_SPEC_PATH_ENV = "OLIGOTERNARY_RUN_SPEC_PATH"


@dataclass(frozen=True)
class ArtifactSpec:
    """One output declared by a stage adapter."""

    role: str
    path: Path
    media_type: Optional[str] = None


@dataclass(frozen=True)
class ArtifactRecord:
    """Stored metadata for one ready output."""

    role: str
    path: str
    media_type: str
    size: int


@dataclass(frozen=True)
class StageResultSummary:
    """Normalized batch outcome reported by a stage command."""

    path: str
    schema_version: int
    stage: str
    run_specification: str
    artifacts: List[ArtifactRecord]
    total_count: int
    succeeded_count: int
    failed_count: int
    failed_inputs: List[str]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StageResultSummary":
        values = dict(raw)
        values["artifacts"] = [
            ArtifactRecord(**artifact) for artifact in values["artifacts"]
        ]
        return cls(**values)


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_exact_fields(
    raw: Mapping[str, Any], required: set[str], context: str
) -> None:
    unknown = sorted(set(raw) - required)
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"{context} is missing: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def read_result_summary(path: Path) -> StageResultSummary:
    """Read and validate a stage result summary."""

    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read result summary {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"result summary {path} must contain an object")
    required = {
        "schema_version",
        "stage",
        "run_specification",
        "artifacts",
        "total_count",
        "succeeded_count",
        "failed_count",
        "failed_inputs",
    }
    _require_exact_fields(raw, required, f"result summary {path}")

    if raw["schema_version"] != 1 or type(raw["schema_version"]) is not int:
        raise ValueError(f"result summary {path} schema_version must be integer 1")
    stage = _require_string(raw["stage"], f"result summary {path} stage")
    run_specification = _require_string(
        raw["run_specification"], f"result summary {path} run_specification"
    )

    raw_artifacts = raw["artifacts"]
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError(f"result summary {path} artifacts must be a non-empty list")
    artifacts = []
    roles: set[str] = set()
    artifact_paths: set[str] = set()
    for index, raw_artifact in enumerate(raw_artifacts):
        context = f"result summary {path} artifacts[{index}]"
        if not isinstance(raw_artifact, Mapping):
            raise ValueError(f"{context} must be an object")
        _require_exact_fields(
            raw_artifact, {"role", "path", "media_type", "size"}, context
        )
        role = _require_string(raw_artifact["role"], f"{context}.role")
        artifact_path = _require_string(raw_artifact["path"], f"{context}.path")
        media_type = _require_string(
            raw_artifact["media_type"], f"{context}.media_type"
        )
        size = raw_artifact["size"]
        if type(size) is not int or size < 1:
            raise ValueError(f"{context}.size must be a positive integer")
        if role in roles:
            raise ValueError(f"result summary {path} artifacts has duplicate role: {role}")
        if artifact_path in artifact_paths:
            raise ValueError(
                f"result summary {path} artifacts has duplicate path: {artifact_path}"
            )
        artifacts.append(
            ArtifactRecord(
                role=role,
                path=artifact_path,
                media_type=media_type,
                size=size,
            )
        )
        roles.add(role)
        artifact_paths.add(artifact_path)

    counts = []
    for name in ("total_count", "succeeded_count", "failed_count"):
        value = raw[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"result summary {path} {name} must be a non-negative integer")
        counts.append(value)
    total_count, succeeded_count, failed_count = counts
    if total_count == 0:
        raise ValueError(f"result summary {path} total_count must be positive")
    if total_count != succeeded_count + failed_count:
        raise ValueError(
            f"result summary {path} counts must satisfy "
            "total_count = succeeded_count + failed_count"
        )

    failed_inputs = raw["failed_inputs"]
    if not isinstance(failed_inputs, list) or any(
        not isinstance(item, str) or not item for item in failed_inputs
    ):
        raise ValueError(f"result summary {path} failed_inputs must be a list of names")
    if len(failed_inputs) != failed_count:
        raise ValueError(
            f"result summary {path} failed_inputs length must equal failed_count"
        )
    return StageResultSummary(
        path=str(path),
        schema_version=1,
        stage=stage,
        run_specification=run_specification,
        artifacts=artifacts,
        total_count=total_count,
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        failed_inputs=list(failed_inputs),
    )


def write_result_summary(
    path: Path,
    *,
    artifacts: Sequence[ArtifactSpec],
    total_count: int,
    succeeded_count: int,
    failed_inputs: Sequence[str],
    stage: Optional[str] = None,
    run_specification_path: Optional[Path] = None,
) -> Path:
    """Atomically write a stage result summary."""

    stage_name = _require_string(
        stage or os.environ.get(STAGE_NAME_ENV), "Stage result stage"
    )
    raw_run_specification_path = run_specification_path or os.environ.get(
        RUN_SPEC_PATH_ENV
    )
    if raw_run_specification_path is None:
        raise ValueError(
            f"Stage result run specification path is required ({RUN_SPEC_PATH_ENV})"
        )
    resolved_run_specification = Path(raw_run_specification_path).expanduser().resolve()
    if not resolved_run_specification.is_file():
        raise ValueError(
            f"Stage result run specification does not exist: {resolved_run_specification}"
        )

    if not artifacts:
        raise ValueError("Stage result artifacts must be a non-empty list")
    records = []
    roles: set[str] = set()
    artifact_paths: set[Path] = set()
    for artifact in artifacts:
        role = _require_string(artifact.role, "Stage result Artifact role")
        artifact_path = artifact.path.expanduser().resolve()
        if role in roles:
            raise ValueError(f"Stage result artifacts has duplicate role: {role}")
        if artifact_path in artifact_paths:
            raise ValueError(f"Stage result artifacts has duplicate path: {artifact_path}")
        record = record_artifact(
            artifact_path,
            role=role,
            media_type=artifact.media_type,
        )
        if record is None:
            raise ValueError(
                f"cannot record Stage result Artifact {role!r}: {artifact_path}"
            )
        records.append(record)
        roles.add(role)
        artifact_paths.add(artifact_path)

    if type(total_count) is not int or total_count < 1:
        raise ValueError("Stage result total_count must be a positive integer")
    if type(succeeded_count) is not int or not 0 <= succeeded_count <= total_count:
        raise ValueError(
            "Stage result succeeded_count must be between zero and total_count"
        )
    failed_count = total_count - succeeded_count
    if isinstance(failed_inputs, (str, bytes)) or any(
        not isinstance(item, str) or not item for item in failed_inputs
    ):
        raise ValueError("Stage result failed_inputs must be a list of names")
    if len(failed_inputs) != failed_count:
        raise ValueError("Stage result failed_inputs length must equal failed_count")

    target = Path(path).expanduser().resolve()
    for artifact_path in artifact_paths:
        if (
            target == artifact_path
            or target in artifact_path.parents
            or artifact_path in target.parents
        ):
            raise ValueError(
                f"Stage result path overlaps declared Artifact: {target} and {artifact_path}"
            )
    payload = {
        "schema_version": 1,
        "stage": stage_name,
        "run_specification": str(resolved_run_specification),
        "artifacts": [
            {
                "role": record.role,
                "path": record.path,
                "media_type": record.media_type,
                "size": record.size,
            }
            for record in records
        ],
        "total_count": total_count,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "failed_inputs": list(failed_inputs),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        read_result_summary(temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def infer_media_type(path: Path) -> str:
    if path.is_dir():
        return _DIRECTORY_MEDIA_TYPE
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _directory_size(path: Path) -> Optional[int]:
    size = 0
    for entry in path.rglob("*"):
        if entry.is_symlink():
            return None
        if entry.is_file():
            size += entry.stat().st_size
        elif not entry.is_dir():
            return None
    return size


def record_artifact(
    path: Path,
    *,
    role: str = "primary",
    media_type: Optional[str] = None,
) -> Optional[ArtifactRecord]:
    """Record a non-empty file or directory without following symbolic links."""

    if path.is_symlink():
        return None
    try:
        if path.is_file():
            size = path.stat().st_size
        elif path.is_dir():
            size = _directory_size(path)
        else:
            return None
    except OSError:
        return None
    if not size:
        return None
    return ArtifactRecord(
        role=role,
        path=str(path),
        media_type=media_type or infer_media_type(path),
        size=size,
    )
