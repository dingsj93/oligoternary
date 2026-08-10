"""Multi-conformer two-stage RESP charge fitting."""

from __future__ import annotations

import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


BOHR_TO_ANGSTROM = 0.529177210903


class RespError(RuntimeError):
    """Raised when RESP inputs are invalid or the fit fails."""


@dataclass(frozen=True)
class RespConformer:
    """Geometry and electrostatic-potential data for one conformer."""

    name: str
    geometry: Path
    grid: Path
    esp: Path
    weight: float


@dataclass(frozen=True)
class RespConfig:
    """Configuration for a two-stage multi-conformer RESP fit."""

    version: int
    project: str
    atom_map: Path
    constraints: Path
    output_dir: Path
    conformers: tuple[RespConformer, ...]
    stage1_weight: float
    stage2_weight: float
    hyperbolic_b: float
    hydrogens_unrestrained: bool
    tolerance: float
    maximum_iterations: int
    config_path: Path


@dataclass(frozen=True)
class RespFitResult:
    """Fitted atom labels, charges, and ESP residual metrics."""

    labels: tuple[str, ...]
    elements: tuple[str, ...]
    treatments: tuple[str, ...]
    stage1_charges: np.ndarray
    stage2_charges: np.ndarray
    relative_rms_errors: dict[str, float]


@dataclass(frozen=True)
class _Atom:
    index: int
    label: str
    element: str
    treatment: str
    fixed_charge: float | None


@dataclass(frozen=True)
class _ConformerData:
    name: str
    inverse_distances: np.ndarray
    esp: np.ndarray
    weight: float


def _resolve(value: Any, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RespError(f"{name} must be a path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_resp_config(path: str | Path) -> RespConfig:
    """Load a RESP YAML configuration and resolve relative paths."""

    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RespError(f"cannot read config {config_path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise RespError("RESP config version must be 1")

    base = config_path.parent
    records = raw.get("conformers")
    if not isinstance(records, list) or not records:
        raise RespError("conformers must be a non-empty list")
    conformers: list[RespConformer] = []
    conformer_names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RespError(f"conformers[{index}] must be a mapping")
        name = str(record.get("name", "")).strip()
        try:
            weight = float(record.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise RespError(f"conformers[{index}] has an invalid weight") from exc
        if not name or not math.isfinite(weight) or weight <= 0:
            raise RespError(f"conformers[{index}] has an invalid name or weight")
        if name in conformer_names:
            raise RespError(f"duplicate conformer name: {name}")
        conformer_names.add(name)
        conformers.append(
            RespConformer(
                name=name,
                geometry=_resolve(record.get("geometry"), base, "geometry"),
                grid=_resolve(record.get("grid"), base, "grid"),
                esp=_resolve(record.get("esp"), base, "esp"),
                weight=weight,
            )
        )

    protocol = raw.get("resp")
    if not isinstance(protocol, dict):
        raise RespError("resp must be a mapping")
    project = str(raw.get("project", "")).strip()
    if not project:
        raise RespError("project must be a non-empty string")

    hydrogens_unrestrained = protocol.get("hydrogens_unrestrained", True)
    if not isinstance(hydrogens_unrestrained, bool):
        raise RespError("resp.hydrogens_unrestrained must be true or false")
    maximum_iterations = protocol.get("maximum_iterations", 200)
    if type(maximum_iterations) is not int or maximum_iterations < 1:
        raise RespError("resp.maximum_iterations must be a positive integer")
    try:
        stage1_weight = float(protocol.get("stage1_weight", 0.0005))
        stage2_weight = float(protocol.get("stage2_weight", 0.001))
        hyperbolic_b = float(protocol.get("hyperbolic_b", 0.1))
        tolerance = float(protocol.get("tolerance", 1.0e-8))
    except (TypeError, ValueError) as exc:
        raise RespError("RESP numerical settings must be numbers") from exc
    if (
        not all(
            math.isfinite(value)
            for value in (stage1_weight, stage2_weight, hyperbolic_b, tolerance)
        )
        or stage1_weight < 0
        or stage2_weight < 0
        or hyperbolic_b <= 0
        or tolerance <= 0
    ):
        raise RespError(
            "RESP weights must be non-negative and hyperbolic_b/tolerance must be positive"
        )

    return RespConfig(
        version=1,
        project=project,
        atom_map=_resolve(raw.get("atom_map"), base, "atom_map"),
        constraints=_resolve(raw.get("constraints"), base, "constraints"),
        output_dir=_resolve(raw.get("output_dir"), base, "output_dir"),
        conformers=tuple(conformers),
        stage1_weight=stage1_weight,
        stage2_weight=stage2_weight,
        hyperbolic_b=hyperbolic_b,
        hydrogens_unrestrained=hydrogens_unrestrained,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
        config_path=config_path,
    )


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _read_atom_map(path: Path) -> list[_Atom]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise RespError(f"cannot read atom map {path}: {exc}") from exc

    atoms: list[_Atom] = []
    for position, row in enumerate(rows, start=1):
        try:
            index = int(row["index_1based"])
            fixed_text = row.get("fixed_charge_e", "").strip()
            atoms.append(
                _Atom(
                    index=index,
                    label=row["model_label"].strip(),
                    element=row["element"].strip(),
                    treatment=row.get("charge_treatment", "fit_RESP").strip(),
                    fixed_charge=float(fixed_text) if fixed_text else None,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RespError(f"invalid atom-map row {position}: {exc}") from exc
    if not atoms or [atom.index for atom in atoms] != list(range(1, len(atoms) + 1)):
        raise RespError("atom-map indices must be consecutive and 1-based")
    return atoms


def _read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        atom_count = int(lines[0])
        rows = [line.split() for line in lines[2:] if line.strip()]
        elements = [row[0] for row in rows]
        coordinates = np.asarray(
            [[float(value) for value in row[1:4]] for row in rows], dtype=float
        )
    except (OSError, ValueError, IndexError) as exc:
        raise RespError(f"cannot read geometry {path}: {exc}") from exc
    if len(rows) != atom_count or coordinates.shape != (atom_count, 3):
        raise RespError(f"geometry atom count does not match its XYZ header: {path}")
    return elements, coordinates


def _read_array(path: Path) -> np.ndarray:
    try:
        with _open_text(path) as handle:
            return np.asarray(np.loadtxt(handle), dtype=float)
    except (OSError, ValueError) as exc:
        raise RespError(f"cannot read numerical data {path}: {exc}") from exc


def _validate_constraints(
    atoms: list[_Atom], constraints: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "model_total_charge_e",
        "stage2_fixed_custom_indices_1based",
        "stage2_equivalence_groups_1based",
    }
    missing = sorted(required - set(constraints))
    if missing:
        raise RespError(f"constraints are missing: {', '.join(missing)}")

    try:
        total_charge = float(constraints["model_total_charge_e"])
    except (TypeError, ValueError) as exc:
        raise RespError("model_total_charge_e must be a number") from exc
    if not math.isfinite(total_charge):
        raise RespError("model_total_charge_e must be finite")

    atom_count = len(atoms)
    fixed_raw = constraints["stage2_fixed_custom_indices_1based"]
    if not isinstance(fixed_raw, list):
        raise RespError("stage2_fixed_custom_indices_1based must be a list")
    fixed_indices: list[int] = []
    for value in fixed_raw:
        if type(value) is not int or not 1 <= value <= atom_count:
            raise RespError(
                f"stage2 fixed index must be between 1 and {atom_count}: {value!r}"
            )
        fixed_indices.append(value)
    if len(set(fixed_indices)) != len(fixed_indices):
        raise RespError("stage2 fixed indices must be unique")
    for index in fixed_indices:
        if atoms[index - 1].fixed_charge is not None:
            raise RespError(f"stage2 custom index is already fixed in atom_map: {index}")

    groups_raw = constraints["stage2_equivalence_groups_1based"]
    if not isinstance(groups_raw, list):
        raise RespError("stage2_equivalence_groups_1based must be a list")
    groups: list[list[int]] = []
    grouped_indices: set[int] = set()
    fixed_set = set(fixed_indices)
    for group_number, group in enumerate(groups_raw, start=1):
        if not isinstance(group, list) or len(group) < 2:
            raise RespError(f"stage2 equivalence group {group_number} needs at least 2 indices")
        indices: list[int] = []
        for value in group:
            if type(value) is not int or not 1 <= value <= atom_count:
                raise RespError(
                    f"stage2 equivalence index must be between 1 and {atom_count}: {value!r}"
                )
            indices.append(value)
        if len(set(indices)) != len(indices):
            raise RespError(f"stage2 equivalence group {group_number} contains duplicates")
        overlap = grouped_indices.intersection(indices)
        if overlap:
            raise RespError(
                "stage2 equivalence groups must be disjoint; repeated index: "
                + str(min(overlap))
            )
        for index in indices:
            if atoms[index - 1].fixed_charge is not None or index in fixed_set:
                raise RespError(
                    f"stage2 equivalence index is fixed and cannot be refitted: {index}"
                )
        grouped_indices.update(indices)
        groups.append(indices)

    return {
        "model_total_charge_e": total_charge,
        "stage2_fixed_custom_indices_1based": fixed_indices,
        "stage2_equivalence_groups_1based": groups,
    }


def _load_fit_inputs(
    config: RespConfig,
) -> tuple[list[_Atom], dict[str, Any], list[_ConformerData]]:
    atoms = _read_atom_map(config.atom_map)
    try:
        constraints = json.loads(config.constraints.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RespError(f"cannot read constraints {config.constraints}: {exc}") from exc
    if not isinstance(constraints, dict):
        raise RespError("constraints must be a JSON object")
    constraints = _validate_constraints(atoms, constraints)

    expected_elements = [atom.element for atom in atoms]
    conformer_data: list[_ConformerData] = []
    for conformer in config.conformers:
        elements, coordinates = _read_xyz(conformer.geometry)
        grid = _read_array(conformer.grid)
        esp = _read_array(conformer.esp).reshape(-1)
        if elements != expected_elements:
            raise RespError(f"atom order differs in conformer {conformer.name}")
        if grid.ndim != 2 or grid.shape[1] != 3 or len(grid) != len(esp):
            raise RespError(f"grid and ESP dimensions differ in {conformer.name}")
        distances = np.linalg.norm(grid[:, None, :] - coordinates[None, :, :], axis=2)
        if np.any(distances == 0):
            raise RespError(f"ESP grid overlaps an atom in {conformer.name}")
        conformer_data.append(
            _ConformerData(
                name=conformer.name,
                inverse_distances=BOHR_TO_ANGSTROM / distances,
                esp=esp,
                weight=conformer.weight,
            )
        )
    return atoms, constraints, conformer_data


def validate_resp_inputs(config: RespConfig) -> dict[str, Any]:
    """Load the atom mapping, constraints, geometries, grids, and ESP values."""

    atoms, constraints, conformers = _load_fit_inputs(config)
    return {
        "project": config.project,
        "atom_count": len(atoms),
        "conformer_count": len(conformers),
        "grid_points": {item.name: len(item.esp) for item in conformers},
        "total_charge_e": float(constraints["model_total_charge_e"]),
    }


def _constraint_row(
    atom_count: int, indices: list[int], signs: list[float]
) -> np.ndarray:
    row = np.zeros(atom_count)
    for index, sign in zip(indices, signs):
        row[index - 1] = sign
    return row


def _fit_stage(
    atoms: list[_Atom],
    conformers: list[_ConformerData],
    total_charge: float,
    fixed_charges: list[tuple[int, float]],
    equivalence_groups: list[list[int]],
    restraint_weight: float,
    hyperbolic_b: float,
    hydrogens_unrestrained: bool,
    tolerance: float,
    maximum_iterations: int,
) -> np.ndarray:
    atom_count = len(atoms)
    rows = [
        _constraint_row(
            atom_count, list(range(1, atom_count + 1)), [1.0] * atom_count
        )
    ]
    values = [total_charge]
    for index, charge in fixed_charges:
        rows.append(_constraint_row(atom_count, [index], [1.0]))
        values.append(charge)
    for group in equivalence_groups:
        for previous, current in zip(group, group[1:]):
            rows.append(_constraint_row(atom_count, [previous, current], [-1.0, 1.0]))
            values.append(0.0)

    dimension = atom_count + len(rows)
    normal = np.zeros((dimension, dimension))
    target = np.zeros(dimension)
    for conformer in conformers:
        inverse = conformer.inverse_distances
        scale = conformer.weight**2
        normal[:atom_count, :atom_count] += scale * np.einsum(
            "ij,ik->jk", inverse, inverse
        )
        target[:atom_count] += scale * np.einsum(
            "i,ij->j", conformer.esp, inverse
        )
    for offset, (row, value) in enumerate(zip(rows, values)):
        column = atom_count + offset
        normal[:atom_count, column] = row
        normal[column, :atom_count] = row
        target[column] = value

    try:
        solution = np.linalg.solve(normal, target)
        for _ in range(maximum_iterations):
            restrained = normal.copy()
            for index, atom in enumerate(atoms):
                if not hydrogens_unrestrained or atom.element != "H":
                    restrained[index, index] += (
                        restraint_weight
                        / math.sqrt(solution[index] ** 2 + hyperbolic_b**2)
                        * len(conformers)
                    )
            updated = np.linalg.solve(restrained, target)
            difference = float(
                np.max(np.abs(updated[:atom_count] - solution[:atom_count]))
            )
            solution = updated
            if difference <= tolerance:
                return solution[:atom_count]
    except np.linalg.LinAlgError as exc:
        raise RespError(f"RESP linear system could not be solved: {exc}") from exc
    raise RespError(f"RESP fit did not converge after {maximum_iterations} iterations")


def fit_resp_charges(config: RespConfig) -> RespFitResult:
    """Fit shared two-stage RESP charges across all configured conformers."""

    atoms, constraints, conformers = _load_fit_inputs(config)
    total_charge = float(constraints["model_total_charge_e"])
    fixed_charges = [
        (atom.index, float(atom.fixed_charge))
        for atom in atoms
        if atom.fixed_charge is not None
    ]
    stage1 = _fit_stage(
        atoms,
        conformers,
        total_charge,
        fixed_charges,
        [],
        config.stage1_weight,
        config.hyperbolic_b,
        config.hydrogens_unrestrained,
        config.tolerance,
        config.maximum_iterations,
    )

    stage2_fixed = fixed_charges + [
        (index, float(stage1[index - 1]))
        for index in constraints["stage2_fixed_custom_indices_1based"]
    ]
    equivalence_groups = constraints["stage2_equivalence_groups_1based"]
    stage2 = _fit_stage(
        atoms,
        conformers,
        total_charge,
        stage2_fixed,
        equivalence_groups,
        config.stage2_weight,
        config.hyperbolic_b,
        config.hydrogens_unrestrained,
        config.tolerance,
        config.maximum_iterations,
    )

    errors: dict[str, float] = {}
    for conformer in conformers:
        residual = conformer.inverse_distances @ stage2 - conformer.esp
        errors[conformer.name] = math.sqrt(
            float(np.dot(residual, residual))
            / float(np.dot(conformer.esp, conformer.esp))
        )
    return RespFitResult(
        labels=tuple(atom.label for atom in atoms),
        elements=tuple(atom.element for atom in atoms),
        treatments=tuple(atom.treatment for atom in atoms),
        stage1_charges=stage1,
        stage2_charges=stage2,
        relative_rms_errors=errors,
    )


def write_resp_result(config: RespConfig, result: RespFitResult) -> tuple[Path, Path]:
    """Write fitted charges and a compact fit report."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    charges_path = config.output_dir / "charges.csv"
    with charges_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index_1based",
                "model_label",
                "element",
                "charge_treatment",
                "stage1_RESP_e",
                "stage2_RESP_e",
            ]
        )
        for index, values in enumerate(
            zip(
                result.labels,
                result.elements,
                result.treatments,
                result.stage1_charges,
                result.stage2_charges,
            ),
            start=1,
        ):
            label, element, treatment, stage1, stage2 = values
            writer.writerow(
                [index, label, element, treatment, f"{stage1:.10f}", f"{stage2:.10f}"]
            )

    report_path = config.output_dir / "fit_report.json"
    report_path.write_text(
        json.dumps(
            {
                "project": config.project,
                "atom_count": len(result.labels),
                "conformer_count": len(config.conformers),
                "conformers": [
                    {"name": conformer.name, "weight": conformer.weight}
                    for conformer in config.conformers
                ],
                "protocol": {
                    "stage1_weight": config.stage1_weight,
                    "stage2_weight": config.stage2_weight,
                    "hyperbolic_b": config.hyperbolic_b,
                    "hydrogens_unrestrained": config.hydrogens_unrestrained,
                    "tolerance": config.tolerance,
                    "maximum_iterations": config.maximum_iterations,
                },
                "total_charge_e": float(np.sum(result.stage2_charges)),
                "relative_rms_error": result.relative_rms_errors,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return charges_path, report_path


__all__ = [
    "RespConfig",
    "RespConformer",
    "RespError",
    "RespFitResult",
    "fit_resp_charges",
    "load_resp_config",
    "validate_resp_inputs",
    "write_resp_result",
]
