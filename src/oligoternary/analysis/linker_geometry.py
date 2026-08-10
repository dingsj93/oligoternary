"""Linker geometry metrics on a saved ternary-complex pose."""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

from pyrosetta import Pose
from pyrosetta.rosetta.core.id import AtomID

from oligoternary.modeling.types import parse_pdb_atom_label

def _resolve_residue_index(pose: Pose, chain: str, resnum: int, icode: str) -> int:
    pdb_info = pose.pdb_info()
    if pdb_info is None:
        raise ValueError("Pose lacks PDBInfo; cannot resolve residue indices.")
    matches = []
    for i in range(1, pose.total_residue() + 1):
        if (
            pdb_info.chain(i) == chain
            and pdb_info.number(i) == resnum
            and (pdb_info.icode(i) or " ") == icode
        ):
            matches.append(i)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one residue for chain={chain} resnum={resnum} icode='{icode}', "
            f"got {matches}"
        )
    return matches[0]


def _single_residue_index_for_chain(pose: Pose, chain: str) -> int:
    pdb_info = pose.pdb_info()
    if pdb_info is None:
        raise ValueError("Pose lacks PDBInfo.")
    matches = [i for i in range(1, pose.total_residue() + 1) if pdb_info.chain(i) == chain]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one residue on chain '{chain}', got {len(matches)}: {matches}"
        )
    return matches[0]


def _heavy_atom_records(pose: Pose) -> List[Tuple[int, int, float]]:
    """Return (residue_index, atom_index, lj_radius) for each heavy atom."""
    records: List[Tuple[int, int, float]] = []
    for residue_index in range(1, pose.total_residue() + 1):
        residue = pose.residue(residue_index)
        for atom_index in range(1, residue.natoms() + 1):
            atom_type = residue.atom_type(atom_index)
            if atom_type.is_hydrogen() or atom_type.is_virtual():
                continue
            try:
                radius = float(atom_type.lj_radius())
            except AttributeError as exc:
                raise ValueError(
                    f"Atom type for residue {residue_index} atom {atom_index} "
                    "does not expose lj_radius(); cannot compute vdW overlap."
                ) from exc
            records.append((residue_index, atom_index, radius))
    return records


def build_linker_environment_excluded_pair_keys(
    pose: Pose,
    linker_chain: str,
    linker_to_warhead_atom: str,
    warhead_anchor_label: str,
    linker_to_e3l_atom: str,
    e3l_anchor_label: str,
) -> set:
    """Exclude atom pairs separated by at most three bonds across an attachment."""
    linker_res_idx = _single_residue_index_for_chain(pose, linker_chain)
    keys = set()

    def heavy_atom_depths(residue, start_atom: int) -> Dict[int, int]:
        depths = {start_atom: 0}
        queue = deque([start_atom])
        while queue:
            atom_index = queue.popleft()
            if depths[atom_index] == 2:
                continue
            for neighbor_index in residue.bonded_neighbor(atom_index):
                atom_type = residue.atom_type(neighbor_index)
                if atom_type.is_hydrogen() or atom_type.is_virtual():
                    continue
                if neighbor_index not in depths:
                    depths[neighbor_index] = depths[atom_index] + 1
                    queue.append(neighbor_index)
        return depths

    for linker_atom_name, env_label in [
        (linker_to_warhead_atom, warhead_anchor_label),
        (linker_to_e3l_atom, e3l_anchor_label),
    ]:
        env_chain, env_resnum, env_icode, env_atom_name = parse_pdb_atom_label(env_label)
        env_res_idx = _resolve_residue_index(pose, env_chain, env_resnum, env_icode)
        linker_atom_idx = pose.residue(linker_res_idx).atom_index(linker_atom_name)
        env_atom_idx = pose.residue(env_res_idx).atom_index(env_atom_name)
        linker_residue = pose.residue(linker_res_idx)
        env_residue = pose.residue(env_res_idx)
        linker_depths = heavy_atom_depths(linker_residue, linker_atom_idx)
        env_depths = heavy_atom_depths(env_residue, env_atom_idx)
        for linker_index, linker_depth in linker_depths.items():
            for env_index, env_depth in env_depths.items():
                if linker_depth + 1 + env_depth <= 3:
                    keys.add(
                        frozenset(
                            ((linker_res_idx, linker_index), (env_res_idx, env_index))
                        )
                    )
    return keys


def compute_linker_environment_clashes(
    pose: Pose,
    linker_chain: str,
    excluded_pair_keys: set,
    overlap_tolerance: float = 0.0,
) -> Dict[str, float]:
    """Match refiner linker-environment clash counting (overlap_tolerance=0)."""
    linker_res_idx = _single_residue_index_for_chain(pose, linker_chain)
    linker_atoms: List[Tuple[int, int, float]] = []
    env_atoms: List[Tuple[int, int, float]] = []
    for record in _heavy_atom_records(pose):
        if record[0] == linker_res_idx:
            linker_atoms.append(record)
        else:
            env_atoms.append(record)

    clash_count = 0
    overlap_sum = 0.0
    worst_overlap = 0.0
    min_distance = float("inf")
    for lr, la, lrad in linker_atoms:
        l_xyz = pose.xyz(AtomID(la, lr))
        for er, ea, erad in env_atoms:
            pair_key = frozenset(((lr, la), (er, ea)))
            if pair_key in excluded_pair_keys:
                continue
            e_xyz = pose.xyz(AtomID(ea, er))
            distance = l_xyz.distance(e_xyz)
            if distance < min_distance:
                min_distance = distance
            overlap = lrad + erad - distance
            if overlap > overlap_tolerance:
                clash_count += 1
                overlap_sum += overlap
                worst_overlap = max(worst_overlap, overlap)
    if min_distance == float("inf"):
        min_distance = float("nan")
    return {
        "count": int(clash_count),
        "min_distance": float(min_distance),
        "overlap_sum": float(overlap_sum),
        "worst_overlap": float(worst_overlap),
    }


def compute_linker_extension(
    pose: Pose,
    linker_chain: str,
    linker_to_warhead_atom: str,
    linker_to_e3l_atom: str,
) -> Dict[str, float]:
    """BFS contour from actual path bond lengths; ratio = end-to-end / contour."""
    linker_res_idx = _single_residue_index_for_chain(pose, linker_chain)
    residue = pose.residue(linker_res_idx)
    start_atom_idx = residue.atom_index(linker_to_warhead_atom)
    end_atom_idx = residue.atom_index(linker_to_e3l_atom)

    # heavy-atom adjacency on the linker residue
    heavy_atoms = [
        i for i in range(1, residue.natoms() + 1)
        if not residue.atom_type(i).is_hydrogen()
        and not residue.atom_type(i).is_virtual()
    ]
    heavy_set = set(heavy_atoms)
    adj: Dict[int, List[int]] = {i: [] for i in heavy_atoms}
    for i in heavy_atoms:
        for j in residue.bonded_neighbor(i):
            if j in heavy_set:
                adj[i].append(j)

    # BFS shortest path in bonds from start to end
    if start_atom_idx == end_atom_idx:
        atom_path = [start_atom_idx]
    else:
        prev = {start_atom_idx: None}
        q = deque([start_atom_idx])
        found = False
        while q and not found:
            u = q.popleft()
            for v in adj[u]:
                if v in prev:
                    continue
                prev[v] = u
                if v == end_atom_idx:
                    found = True
                    break
                q.append(v)
        if not found:
            raise ValueError(
                f"No bond path between {linker_to_warhead_atom} and {linker_to_e3l_atom} "
                f"on linker residue {linker_res_idx}."
            )
        atom_path = [end_atom_idx]
        cur = end_atom_idx
        while prev[cur] is not None:
            cur = prev[cur]
            atom_path.append(cur)
        atom_path.reverse()

    bond_path_length = len(atom_path) - 1
    contour = 0.0
    for atom1, atom2 in zip(atom_path, atom_path[1:]):
        contour += float(
            pose.xyz(AtomID(atom1, linker_res_idx)).distance(
                pose.xyz(AtomID(atom2, linker_res_idx))
            )
        )
    end_to_end = float(
        pose.xyz(AtomID(start_atom_idx, linker_res_idx)).distance(
            pose.xyz(AtomID(end_atom_idx, linker_res_idx))
        )
    )
    ratio = float("nan") if contour == 0 else end_to_end / contour
    return {
        "linker_n_bonds_shortest_path": int(bond_path_length),
        "linker_contour_length": float(contour),
        "linker_end_to_end": end_to_end,
        "linker_extension_ratio": float(ratio),
    }


def compute_linker_geometry_metrics(
    pose: Pose,
    linker_chain: str,
    linker_to_warhead_atom: str,
    warhead_anchor_label: str,
    linker_to_e3l_atom: str,
    e3l_anchor_label: str,
) -> Dict[str, float]:
    """Aggregate linker-environment clashes and linker extension."""
    excluded_pair_keys = build_linker_environment_excluded_pair_keys(
        pose,
        linker_chain=linker_chain,
        linker_to_warhead_atom=linker_to_warhead_atom,
        warhead_anchor_label=warhead_anchor_label,
        linker_to_e3l_atom=linker_to_e3l_atom,
        e3l_anchor_label=e3l_anchor_label,
    )
    env = compute_linker_environment_clashes(pose, linker_chain, excluded_pair_keys)
    ext = compute_linker_extension(pose, linker_chain, linker_to_warhead_atom, linker_to_e3l_atom)
    return {
        "linker_env_clashes_after_relief": env["count"],
        "linker_env_min_distance_after_relief": env["min_distance"],
        "linker_env_overlap_sum_after_relief": env["overlap_sum"],
        "linker_env_worst_overlap_after_relief": env["worst_overlap"],
        **ext,
    }
