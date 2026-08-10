"""Validation and execution of sequential Amber simulations."""

from __future__ import annotations

import gzip
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SimulationConfig, SimulationError


@dataclass(frozen=True)
class StageCommand:
    """Resolved command and outputs for one Amber stage."""

    name: str
    command: tuple[str, ...]
    output_dir: Path
    restart: Path
    trajectory: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "output_dir": str(self.output_dir),
            "restart": str(self.restart),
            "trajectory": str(self.trajectory) if self.trajectory else None,
        }


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_prmtop_atom_count(path: str | Path) -> int:
    """Read NATOM, the first value in the Amber topology POINTERS section."""

    source = Path(path)
    try:
        with _open_text(source) as handle:
            in_pointers = False
            for line in handle:
                if line.startswith("%FLAG"):
                    in_pointers = line.strip() == "%FLAG POINTERS"
                    continue
                if not in_pointers or line.startswith("%FORMAT"):
                    continue
                fields = line.split()
                if fields:
                    atom_count = int(fields[0])
                    if atom_count > 0:
                        return atom_count
                    break
    except (OSError, UnicodeError, ValueError) as exc:
        raise SimulationError(f"cannot read topology {source}: {exc}") from exc
    raise SimulationError(f"topology has no valid POINTERS NATOM value: {source}")


def read_coordinate_atom_count(path: str | Path) -> int:
    """Read NATOM and verify the Amber coordinate payload length."""

    source = Path(path)
    try:
        with _open_text(source) as handle:
            next(handle)
            fields = next(handle).split()
            atom_count = int(fields[0])
            value_count = 0
            for line in handle:
                for value in line.split():
                    float(value)
                    value_count += 1
    except (OSError, UnicodeError, StopIteration, ValueError, IndexError) as exc:
        raise SimulationError(f"cannot read coordinates {source}: {exc}") from exc
    if atom_count < 1:
        raise SimulationError(f"coordinates have an invalid NATOM value: {source}")
    coordinate_values = 3 * atom_count
    valid_value_counts = {
        payload_values + box_values
        for payload_values in (coordinate_values, 2 * coordinate_values)
        for box_values in (0, 3, 6)
    }
    if value_count not in valid_value_counts:
        raise SimulationError(
            f"coordinates contain {value_count} numerical values after NATOM; "
            f"expected coordinates for {atom_count} atoms with optional velocities and box"
        )
    return atom_count


def validate_simulation(config: SimulationConfig) -> int:
    """Validate Amber inputs and return their shared atom count."""

    required = [config.topology, config.coordinates]
    required.extend(stage.input for stage in config.stages)
    for path in required:
        if not path.is_file():
            raise SimulationError(f"input file does not exist: {path}")

    topology_atoms = read_prmtop_atom_count(config.topology)
    coordinate_atoms = read_coordinate_atom_count(config.coordinates)
    if topology_atoms != coordinate_atoms:
        raise SimulationError(
            "topology and coordinates contain different atom counts: "
            f"{topology_atoms} != {coordinate_atoms}"
        )
    return topology_atoms


def _materialized_name(path: Path) -> str:
    return path.name[:-3] if path.suffix.lower() == ".gz" else path.name


def _materialize(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".gz":
        with gzip.open(source, "rb") as source_handle, destination.open("wb") as target:
            shutil.copyfileobj(source_handle, target)
    else:
        shutil.copyfile(source, destination)


def build_stage_commands(config: SimulationConfig) -> tuple[StageCommand, ...]:
    """Build the ordered Amber commands without changing the filesystem."""

    inputs_dir = config.output_dir / "inputs"
    topology = inputs_dir / _materialized_name(config.topology)
    coordinates = inputs_dir / _materialized_name(config.coordinates)
    restarts: dict[str, Path] = {}
    plans: list[StageCommand] = []
    current_coordinates = coordinates

    for stage in config.stages:
        stage_dir = config.output_dir / stage.name
        restart = stage_dir / "restart"
        trajectory = stage_dir / "trajectory.nc" if stage.trajectory else None
        command = [
            config.engine,
            "-O",
            "-i",
            str(stage.input),
            "-o",
            str(stage_dir / "mdout"),
            "-p",
            str(topology),
            "-c",
            str(current_coordinates),
            "-r",
            str(restart),
            "-inf",
            str(stage_dir / "mdinfo"),
        ]
        if trajectory is not None:
            command.extend(["-x", str(trajectory)])
        if stage.reference is not None:
            reference = (
                coordinates
                if stage.reference == "initial"
                else restarts[stage.reference]
            )
            command.extend(["-ref", str(reference)])

        plans.append(
            StageCommand(
                name=stage.name,
                command=tuple(command),
                output_dir=stage_dir,
                restart=restart,
                trajectory=trajectory,
            )
        )
        restarts[stage.name] = restart
        current_coordinates = restart

    return tuple(plans)


def run_simulation(
    config: SimulationConfig, *, dry_run: bool = False
) -> tuple[StageCommand, ...]:
    """Validate and run all Amber stages in order."""

    validate_simulation(config)
    plans = build_stage_commands(config)
    if dry_run:
        return plans

    inputs_dir = config.output_dir / "inputs"
    _materialize(config.topology, inputs_dir / _materialized_name(config.topology))
    _materialize(
        config.coordinates, inputs_dir / _materialized_name(config.coordinates)
    )

    for plan in plans:
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(plan.command, check=False)
        except OSError as exc:
            raise SimulationError(
                f"cannot start Amber engine {config.engine}: {exc}"
            ) from exc
        if result.returncode != 0:
            raise SimulationError(
                f"Amber stage {plan.name!r} failed with exit code {result.returncode}"
            )
    return plans


__all__ = [
    "StageCommand",
    "build_stage_commands",
    "read_coordinate_atom_count",
    "read_prmtop_atom_count",
    "run_simulation",
    "validate_simulation",
]
