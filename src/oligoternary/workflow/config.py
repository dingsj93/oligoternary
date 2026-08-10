"""Load and validate the version-1 workflow configuration interface."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from .artifacts import ArtifactSpec
from ..modeling.specification import (
    RefinementSpec,
    ScientificSpecError,
    compile_refinement_command,
    parse_refinement_spec,
)


class ConfigError(ValueError):
    """Raised when a workflow configuration is invalid."""


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration for one implementation behind the stage adapter seam."""

    type: str
    artifacts: Tuple[ArtifactSpec, ...]
    command: Tuple[str, ...] = ()
    cwd: Optional[Path] = None
    result_summary: Optional[Path] = None
    refinement: Optional[RefinementSpec] = None
    timeout_seconds: Optional[float] = None

    @property
    def artifact(self) -> Path:
        """The primary Artifact path retained for version-1 callers."""

        return self.artifacts[0].path


@dataclass(frozen=True)
class StageConfig:
    """One ordered workflow stage."""

    name: str
    depends_on: Tuple[str, ...]
    adapter: AdapterConfig


@dataclass(frozen=True)
class WorkflowConfig:
    """Validated workflow specification with paths resolved against its file."""

    version: int
    project: str
    output_dir: Path
    stages: Tuple[StageConfig, ...]
    config_path: Path

    @property
    def config_dir(self) -> Path:
        return self.config_path.parent

    def stage(self, name: str) -> StageConfig:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise ConfigError(f"unknown stage: {name}")


_ROOT_KEYS = {"version", "project", "output_dir", "stages"}
_STAGE_KEYS = {"name", "depends_on", "adapter"}
_ADAPTER_KEYS = {
    "type",
    "artifact",
    "artifacts",
    "command",
    "cwd",
    "result_summary",
    "timeout_seconds",
}
_ARTIFACT_KEYS = {"role", "path", "media_type"}
_ADAPTER_TYPES = {"command", "existing-artifact", "linker-refinement"}
_SAFE_STAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _reject_unknown_keys(
    mapping: Mapping[str, Any], allowed: set[str], context: str
) -> None:
    non_string = [repr(key) for key in mapping if not isinstance(key, str)]
    if non_string:
        raise ConfigError(f"{context} field names must be strings: {', '.join(non_string)}")
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"{context} has unknown field(s): {', '.join(unknown)}")


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a mapping")
    return value


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a non-empty string")
    return value.strip()


def _resolve_path(value: Any, config_dir: Path, context: str) -> Path:
    raw = _require_nonempty_string(value, context)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _load_raw(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc

    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"cannot parse config {path}: {exc}") from exc
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigError("PyYAML is required to read YAML configs") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"cannot parse config {path}: {exc}") from exc
    else:
        raise ConfigError("config file must use .json, .yaml, or .yml")

    return _require_mapping(data, "config root")


def _parse_adapter(
    raw_value: Any, config_dir: Path, stage_name: str
) -> AdapterConfig:
    context = f"stage {stage_name!r} adapter"
    raw = _require_mapping(raw_value, context)

    adapter_type = _require_nonempty_string(raw.get("type"), f"{context}.type")
    if adapter_type not in _ADAPTER_TYPES:
        allowed = ", ".join(sorted(_ADAPTER_TYPES))
        raise ConfigError(f"{context}.type must be one of: {allowed}")

    if adapter_type == "linker-refinement":
        try:
            refinement = parse_refinement_spec(raw, config_dir)
        except ScientificSpecError as exc:
            raise ConfigError(f"{context}: {exc}") from exc
        return AdapterConfig(
            type=adapter_type,
            artifacts=(
                ArtifactSpec(
                    role="refined_structure",
                    path=refinement.artifact,
                    media_type="chemical/x-pdb",
                ),
                ArtifactSpec(
                    role="linker_params",
                    path=refinement.linker_params,
                    media_type="text/plain",
                ),
                ArtifactSpec(
                    role="e3_ligand_params",
                    path=refinement.e3_ligand_params,
                    media_type="text/plain",
                ),
            ),
            command=compile_refinement_command(refinement),
            cwd=config_dir,
            result_summary=refinement.result_summary,
            refinement=refinement,
        )

    _reject_unknown_keys(raw, _ADAPTER_KEYS, context)

    if "artifact" in raw and "artifacts" in raw:
        raise ConfigError(f"{context} accepts either artifact or artifacts, not both")
    if "artifact" in raw:
        artifacts = (
            ArtifactSpec(
                role="primary",
                path=_resolve_path(
                    raw["artifact"], config_dir, f"{context}.artifact"
                ),
            ),
        )
    elif "artifacts" in raw:
        raw_artifacts = raw["artifacts"]
        if (
            not isinstance(raw_artifacts, Sequence)
            or isinstance(raw_artifacts, (str, bytes))
            or not raw_artifacts
        ):
            raise ConfigError(f"{context}.artifacts must be a non-empty list")
        parsed_artifacts = []
        roles: set[str] = set()
        for index, raw_artifact_value in enumerate(raw_artifacts):
            artifact_context = f"{context}.artifacts[{index}]"
            raw_artifact = _require_mapping(raw_artifact_value, artifact_context)
            _reject_unknown_keys(raw_artifact, _ARTIFACT_KEYS, artifact_context)
            role = _require_nonempty_string(
                raw_artifact.get("role"), f"{artifact_context}.role"
            )
            if role in roles:
                raise ConfigError(f"{context}.artifacts has duplicate role: {role}")
            media_type = (
                _require_nonempty_string(
                    raw_artifact["media_type"], f"{artifact_context}.media_type"
                )
                if "media_type" in raw_artifact
                else None
            )
            parsed_artifacts.append(
                ArtifactSpec(
                    role=role,
                    path=_resolve_path(
                        raw_artifact.get("path"),
                        config_dir,
                        f"{artifact_context}.path",
                    ),
                    media_type=media_type,
                )
            )
            roles.add(role)
        artifacts = tuple(parsed_artifacts)
    else:
        raise ConfigError(f"{context} requires artifact or artifacts")

    command: Tuple[str, ...] = ()
    cwd: Optional[Path] = None
    timeout_seconds: Optional[float] = None
    result_summary = (
        _resolve_path(
            raw["result_summary"], config_dir, f"{context}.result_summary"
        )
        if "result_summary" in raw
        else None
    )
    if adapter_type == "command":
        raw_command = raw.get("command")
        if (
            not isinstance(raw_command, Sequence)
            or isinstance(raw_command, (str, bytes))
            or not raw_command
        ):
            raise ConfigError(f"{context}.command must be a non-empty list of strings")
        if any(not isinstance(item, str) or not item for item in raw_command):
            raise ConfigError(f"{context}.command must contain only non-empty strings")
        if raw_command.count("{stage-result}") > 1:
            raise ConfigError(f"{context}.command accepts one {{stage-result}} placeholder")
        command_items = []
        for item in raw_command:
            if item == "{python}":
                command_items.append(sys.executable)
            elif item == "{stage-result}":
                if result_summary is None:
                    raise ConfigError(
                        f"{context}.command uses {{stage-result}} without result_summary"
                    )
                command_items.extend(["--result-summary", str(result_summary)])
                for artifact in artifacts:
                    command_items.extend(
                        ["--result-artifact", f"{artifact.role}={artifact.path}"]
                    )
            else:
                command_items.append(item)
        command = tuple(command_items)
        cwd = (
            _resolve_path(raw["cwd"], config_dir, f"{context}.cwd")
            if "cwd" in raw
            else config_dir
        )
        if "timeout_seconds" in raw:
            raw_timeout = raw["timeout_seconds"]
            if (
                isinstance(raw_timeout, bool)
                or not isinstance(raw_timeout, (int, float))
                or raw_timeout <= 0
            ):
                raise ConfigError(f"{context}.timeout_seconds must be a positive number")
            timeout_seconds = float(raw_timeout)
    else:
        forbidden = sorted(
            key for key in ("command", "cwd", "timeout_seconds") if key in raw
        )
        if forbidden:
            raise ConfigError(
                f"{context} does not accept field(s): {', '.join(forbidden)}"
            )

    return AdapterConfig(
        type=adapter_type,
        artifacts=artifacts,
        command=command,
        cwd=cwd,
        result_summary=result_summary,
        timeout_seconds=timeout_seconds,
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_output_artifacts(
    stages: Sequence[StageConfig], workflow_output_dir: Path
) -> None:
    """Reject outputs whose content identities can overwrite one another."""

    declared_artifacts: list[tuple[str, str, ArtifactSpec]] = []
    declared_summaries: list[tuple[str, Path]] = []
    for stage in stages:
        artifacts = stage.adapter.artifacts
        for artifact in artifacts:
            if _paths_overlap(artifact.path, workflow_output_dir):
                raise ConfigError(
                    f"stage {stage.name!r} Artifact overlaps workflow output_dir; "
                    "keep scientific Artifacts separate from manifests, logs, and locks: "
                    f"{artifact.path} and {workflow_output_dir}"
                )
        for index, first in enumerate(artifacts):
            for second in artifacts[index + 1 :]:
                if _paths_overlap(first.path, second.path):
                    raise ConfigError(
                        f"stage {stage.name!r} has overlapping output artifacts: "
                        f"{first.path} and {second.path}"
                    )

        for artifact in artifacts:
            for previous_stage, previous_type, previous in declared_artifacts:
                if (
                    _paths_overlap(artifact.path, previous.path)
                    and (
                        stage.adapter.type in {"command", "linker-refinement"}
                        or previous_type in {"command", "linker-refinement"}
                    )
                ):
                    raise ConfigError(
                        "overlapping output artifacts or inputs declared by stages "
                        f"{previous_stage!r} and {stage.name!r}: "
                        f"{previous.path} and {artifact.path}"
                    )
            declared_artifacts.append((stage.name, stage.adapter.type, artifact))

        summary = stage.adapter.result_summary
        if summary is not None:
            if _paths_overlap(summary, workflow_output_dir):
                raise ConfigError(
                    f"stage {stage.name!r} result_summary overlaps workflow "
                    "output_dir; keep Stage results separate from manifests, logs, "
                    f"and locks: {summary} and {workflow_output_dir}"
                )
            for artifact in artifacts:
                if _paths_overlap(summary, artifact.path):
                    raise ConfigError(
                        f"stage {stage.name!r} result_summary overlaps declared Artifact: "
                        f"{summary} and {artifact.path}"
                    )
            for previous_stage, previous_summary in declared_summaries:
                if _paths_overlap(summary, previous_summary):
                    raise ConfigError(
                        "overlapping result_summary paths declared by stages "
                        f"{previous_stage!r} and {stage.name!r}: "
                        f"{previous_summary} and {summary}"
                    )
            declared_summaries.append((stage.name, summary))

    for summary_stage, summary in declared_summaries:
        for artifact_stage, _adapter_type, artifact in declared_artifacts:
            if artifact_stage != summary_stage and _paths_overlap(summary, artifact.path):
                raise ConfigError(
                    f"stage {summary_stage!r} result_summary overlaps Artifact declared "
                    f"by stage {artifact_stage!r}: {summary} and {artifact.path}"
                )


def load_config(path: str | Path) -> WorkflowConfig:
    """Load YAML or JSON and return a strictly validated version-1 config."""

    config_path = Path(path).expanduser().resolve()
    config_dir = config_path.parent
    raw = _load_raw(config_path)
    _reject_unknown_keys(raw, _ROOT_KEYS, "config root")

    version = raw.get("version")
    if type(version) is not int or version != 1:
        raise ConfigError("config version must be integer 1")
    project = _require_nonempty_string(raw.get("project"), "project")
    output_dir = _resolve_path(raw.get("output_dir"), config_dir, "output_dir")

    raw_stages = raw.get("stages")
    if (
        not isinstance(raw_stages, Sequence)
        or isinstance(raw_stages, (str, bytes))
        or not raw_stages
    ):
        raise ConfigError("stages must be a non-empty list")

    stages = []
    seen: set[str] = set()
    for index, raw_stage_value in enumerate(raw_stages):
        context = f"stages[{index}]"
        raw_stage = _require_mapping(raw_stage_value, context)
        _reject_unknown_keys(raw_stage, _STAGE_KEYS, context)
        name = _require_nonempty_string(raw_stage.get("name"), f"{context}.name")
        if (
            name in {".", ".."}
            or _SAFE_STAGE_NAME.fullmatch(name) is None
            or "/" in name
            or "\\" in name
        ):
            raise ConfigError(
                f"{context}.name must be a safe identifier using letters, digits, '.', '_', or '-'"
            )
        if name in seen:
            raise ConfigError(f"duplicate stage name: {name}")

        raw_dependencies = raw_stage.get("depends_on", [])
        if (
            not isinstance(raw_dependencies, Sequence)
            or isinstance(raw_dependencies, (str, bytes))
        ):
            raise ConfigError(f"stage {name!r}.depends_on must be a list of names")
        dependencies = tuple(raw_dependencies)
        if any(not isinstance(item, str) or not item for item in dependencies):
            raise ConfigError(
                f"stage {name!r}.depends_on must contain non-empty strings"
            )
        if len(set(dependencies)) != len(dependencies):
            raise ConfigError(f"stage {name!r}.depends_on contains duplicates")
        missing_or_forward = [item for item in dependencies if item not in seen]
        if missing_or_forward:
            joined = ", ".join(missing_or_forward)
            raise ConfigError(
                f"stage {name!r} dependencies must name earlier stages: {joined}"
            )

        adapter = _parse_adapter(raw_stage.get("adapter"), config_dir, name)
        stages.append(StageConfig(name, dependencies, adapter))
        seen.add(name)

    _validate_output_artifacts(stages, output_dir)

    return WorkflowConfig(
        version=1,
        project=project,
        output_dir=output_dir,
        stages=tuple(stages),
        config_path=config_path,
    )
