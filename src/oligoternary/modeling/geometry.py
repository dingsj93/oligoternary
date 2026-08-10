"""Element geometry and attachment scoring helpers."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree

from oligoternary.modeling.constants import VDW_RADII
from oligoternary.modeling.types import HeavyAtomRecord

def normalize_element_symbol(element: str, atom_name: str) -> str:
    if element:
        normalized = element.strip().upper()
        if normalized:
            return normalized
    match = re.search(r"[A-Za-z]", atom_name)
    if match is None:
        raise ValueError(f"Cannot determine element for atom '{atom_name}'")
    return match.group(0).upper()


def vdw_radius_for_element(element: str) -> float:
    return VDW_RADII.get(element.upper(), 1.70)


def generate_attachment_directions(num_points: int) -> List[np.ndarray]:
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    directions = []
    for index in range(num_points):
        y = 1.0 - (2.0 * index) / (num_points - 1)
        radial = np.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * index
        directions.append(
            np.asarray(
                [np.cos(theta) * radial, y, np.sin(theta) * radial],
                dtype=float,
            )
        )
    return directions


def score_attachment_position(
    position: np.ndarray,
    linker_element: str,
    environment_atoms: List[HeavyAtomRecord],
    excluded_labels: List[str],
    anchor_center: np.ndarray,
    local_environment_radius: float = 4.0,
) -> Tuple[float, float]:
    linker_radius = vdw_radius_for_element(linker_element)
    min_clearance = float("inf")
    overlap_sum = 0.0
    excluded = set(excluded_labels)
    for atom in environment_atoms:
        if atom.label in excluded:
            continue
        anchor_distance = float(np.linalg.norm(atom.coord - anchor_center))
        if anchor_distance > local_environment_radius:
            continue
        distance = float(np.linalg.norm(position - atom.coord))
        clearance = distance - (linker_radius + vdw_radius_for_element(atom.element))
        min_clearance = min(min_clearance, clearance)
        if clearance < 0.0:
            overlap_sum += -clearance
    if min_clearance == float("inf"):
        min_clearance = local_environment_radius
    return overlap_sum, min_clearance


def score_rdkit_mol_environment_clashes(
    mol: Chem.Mol,
    environment_atoms: List[HeavyAtomRecord],
    *,
    start_point: int,
    end_point: int,
    warhead_anchor_label: str,
    e3l_anchor_label: str,
    query_radius: Optional[float] = None,
) -> Dict[str, float]:
    """Score an embedded RDKit linker conformer against environment heavy atoms."""
    conformer = mol.GetConformer()
    linker_positions = []
    linker_radii = []
    linker_indices = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        atom_index = atom.GetIdx()
        linker_indices.append(atom_index)
        position = conformer.GetAtomPosition(atom_index)
        linker_positions.append([position.x, position.y, position.z])
        linker_radii.append(vdw_radius_for_element(atom.GetSymbol()))
    return score_rdkit_environment_clashes(
        np.asarray(linker_positions, dtype=float),
        np.asarray(linker_radii, dtype=float),
        linker_indices,
        environment_atoms,
        start_point=start_point,
        end_point=end_point,
        warhead_anchor_label=warhead_anchor_label,
        e3l_anchor_label=e3l_anchor_label,
        query_radius=query_radius,
    )


def score_rdkit_environment_clashes(
    linker_positions: np.ndarray,
    linker_radii: np.ndarray,
    linker_indices: List[int],
    environment_atoms: List[HeavyAtomRecord],
    *,
    start_point: int,
    end_point: int,
    warhead_anchor_label: str,
    e3l_anchor_label: str,
    query_radius: Optional[float] = None,
) -> Dict[str, float]:
    """Score linker-environment vdW overlaps (exhaustive by default; optional spatial cutoff)."""
    if linker_positions.size == 0:
        raise ValueError("Failed to score linker clashes: no linker heavy atoms.")
    if not environment_atoms:
        raise ValueError("Failed to score linker clashes: empty environment.")

    if query_radius is not None:
        return _score_rdkit_environment_clashes_spatial(
            linker_positions,
            linker_radii,
            linker_indices,
            environment_atoms,
            start_point=start_point,
            end_point=end_point,
            warhead_anchor_label=warhead_anchor_label,
            e3l_anchor_label=e3l_anchor_label,
            query_radius=query_radius,
        )

    clash_count = 0
    overlap_sum = 0.0
    worst_overlap = 0.0
    min_distance = float("inf")
    anchor_overlap = 0.0

    for position, linker_radius, atom_index in zip(linker_positions, linker_radii, linker_indices):
        is_start = atom_index == start_point
        is_end = atom_index == end_point
        for env_atom in environment_atoms:
            if is_start and env_atom.label == warhead_anchor_label:
                continue
            if is_end and env_atom.label == e3l_anchor_label:
                continue
            distance = float(np.linalg.norm(position - env_atom.coord))
            min_distance = min(min_distance, distance)
            env_radius = vdw_radius_for_element(env_atom.element)
            overlap = linker_radius + env_radius - distance
            if overlap > 0.0:
                clash_count += 1
                overlap_sum += overlap
                worst_overlap = max(worst_overlap, overlap)
                if env_atom.label in (warhead_anchor_label, e3l_anchor_label):
                    anchor_overlap = max(anchor_overlap, overlap)

    if min_distance == float("inf"):
        raise ValueError("Failed to score linker clashes: empty environment.")

    return {
        "anchor_overlap": anchor_overlap,
        "count": clash_count,
        "overlap_sum": overlap_sum,
        "worst_overlap": worst_overlap,
        "min_distance": min_distance,
    }


def _score_rdkit_environment_clashes_spatial(
    linker_positions: np.ndarray,
    linker_radii: np.ndarray,
    linker_indices: List[int],
    environment_atoms: List[HeavyAtomRecord],
    *,
    start_point: int,
    end_point: int,
    warhead_anchor_label: str,
    e3l_anchor_label: str,
    query_radius: float,
) -> Dict[str, float]:
    env_coords = np.asarray([atom.coord for atom in environment_atoms], dtype=float)
    env_radii = np.asarray(
        [vdw_radius_for_element(atom.element) for atom in environment_atoms],
        dtype=float,
    )
    max_env_radius = float(np.max(env_radii)) if env_radii.size else 0.0
    tree = cKDTree(env_coords)

    clash_count = 0
    overlap_sum = 0.0
    worst_overlap = 0.0
    min_distance = float("inf")
    anchor_overlap = 0.0

    for position, linker_radius, atom_index in zip(linker_positions, linker_radii, linker_indices):
        is_start = atom_index == start_point
        is_end = atom_index == end_point
        search_radius = query_radius + float(linker_radius) + max_env_radius
        for env_idx in tree.query_ball_point(position, r=search_radius):
            env_atom = environment_atoms[env_idx]
            if is_start and env_atom.label == warhead_anchor_label:
                continue
            if is_end and env_atom.label == e3l_anchor_label:
                continue
            distance = float(np.linalg.norm(position - env_atom.coord))
            min_distance = min(min_distance, distance)
            env_radius = env_radii[env_idx]
            overlap = float(linker_radius) + env_radius - distance
            if overlap > 0.0:
                clash_count += 1
                overlap_sum += overlap
                worst_overlap = max(worst_overlap, overlap)
                if env_atom.label in (warhead_anchor_label, e3l_anchor_label):
                    anchor_overlap = max(anchor_overlap, overlap)

    if min_distance == float("inf"):
        raise ValueError("Failed to score linker clashes: empty environment.")

    return {
        "anchor_overlap": anchor_overlap,
        "count": clash_count,
        "overlap_sum": overlap_sum,
        "worst_overlap": worst_overlap,
        "min_distance": min_distance,
    }
