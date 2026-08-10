"""Configuration objects for Amber molecular-dynamics workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class SimulationError(RuntimeError):
    """Raised when an Amber configuration or run is invalid."""


@dataclass(frozen=True)
class SimulationStage:
    """One Amber input file and its output options."""

    name: str
    input: Path
    reference: str | None = None
    trajectory: bool = False


@dataclass(frozen=True)
class SimulationConfig:
    """A validated, ordered Amber simulation specification."""

    version: int
    project: str
    engine: str
    topology: Path
    coordinates: Path
    output_dir: Path
    stages: tuple[SimulationStage, ...]
    config_path: Path


_ROOT_KEYS = {
    "version",
    "project",
    "engine",
    "topology",
    "coordinates",
    "output_dir",
    "stages",
}
_STAGE_KEYS = {"name", "input", "reference", "trajectory"}
_SAFE_STAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SimulationError(f"{context} must be a mapping")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SimulationError(f"{context} must be a non-empty string")
    return value.strip()


def _path(value: Any, config_dir: Path, context: str) -> Path:
    path = Path(_text(value, context)).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _engine(value: Any, config_dir: Path) -> str:
    engine = _text(value, "engine")
    if "/" not in engine:
        return engine
    path = Path(engine).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def _unknown_keys(raw: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(str(key) for key in set(raw) - allowed)
    if unknown:
        raise SimulationError(f"{context} has unknown field(s): {', '.join(unknown)}")


def load_simulation_config(path: str | Path) -> SimulationConfig:
    """Load an Amber simulation YAML file and resolve its paths."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), "config"
        )
    except (OSError, yaml.YAMLError) as exc:
        raise SimulationError(f"cannot read config {config_path}: {exc}") from exc

    _unknown_keys(raw, _ROOT_KEYS, "config")
    if raw.get("version") != 1:
        raise SimulationError("version must be 1")

    config_dir = config_path.parent
    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise SimulationError("stages must be a non-empty list")

    stages: list[SimulationStage] = []
    prior_names: set[str] = set()
    for index, value in enumerate(raw_stages):
        context = f"stages[{index}]"
        stage = _mapping(value, context)
        _unknown_keys(stage, _STAGE_KEYS, context)
        name = _text(stage.get("name"), f"{context}.name")
        if not _SAFE_STAGE_NAME.fullmatch(name):
            raise SimulationError(f"{context}.name is not a valid stage name: {name}")
        if name in prior_names:
            raise SimulationError(f"duplicate stage name: {name}")

        reference = stage.get("reference")
        if reference is not None:
            reference = _text(reference, f"{context}.reference")
            if reference != "initial" and reference not in prior_names:
                raise SimulationError(
                    f"{context}.reference must be 'initial' or an earlier stage name"
                )

        trajectory = stage.get("trajectory", False)
        if not isinstance(trajectory, bool):
            raise SimulationError(f"{context}.trajectory must be true or false")

        stages.append(
            SimulationStage(
                name=name,
                input=_path(stage.get("input"), config_dir, f"{context}.input"),
                reference=reference,
                trajectory=trajectory,
            )
        )
        prior_names.add(name)

    return SimulationConfig(
        version=1,
        project=_text(raw.get("project"), "project"),
        engine=_engine(raw.get("engine"), config_dir),
        topology=_path(raw.get("topology"), config_dir, "topology"),
        coordinates=_path(raw.get("coordinates"), config_dir, "coordinates"),
        output_dir=_path(raw.get("output_dir"), config_dir, "output_dir"),
        stages=tuple(stages),
        config_path=config_path,
    )


__all__ = [
    "SimulationConfig",
    "SimulationError",
    "SimulationStage",
    "load_simulation_config",
]
