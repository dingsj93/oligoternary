"""Linker-environment clash detection and local relief."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyrosetta
from Bio.PDB import PDBParser
from loguru import logger
from pyrosetta import Pose, create_score_function
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
from pyrosetta.rosetta.core.select import get_residues_from_subset
from pyrosetta.rosetta.core.select.residue_selector import (
    NeighborhoodResidueSelector,
    NotResidueSelector,
)
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

from oligoternary.analysis.linker_geometry import (
    build_linker_environment_excluded_pair_keys,
    compute_linker_environment_clashes,
)
from oligoternary.modeling.constants import (
    LINKER_ENVIRONMENT_CLASH_MIN_DISTANCE,
    LINKER_ENVIRONMENT_CLASH_WARNING_OVERLAP,
)
from oligoternary.modeling.geometry import normalize_element_symbol
from oligoternary.modeling.params_io import (
    ideal_bond_length_from_atom_names,
    parse_heavy_bonds_from_params,
)
from oligoternary.modeling.scorefxn import build_cart_scorefxn
from oligoternary.modeling.types import parse_pdb_atom_label


class LinkerClashMixin:
    def _build_local_movemap(
        self,
        linker_residues: List[int],
        sidechain_residues: List[int],
        backbone_residues: Optional[List[int]] = None,
    ) -> MoveMap:
        movemap = MoveMap()
        movemap.set_bb(False)
        movemap.set_chi(False)
        movemap.set_jump(False)
        linker_residue_set = set(linker_residues)
        for residue_index in linker_residues:
            movemap.set_chi(residue_index, True)
        for residue_index in sidechain_residues:
            if residue_index in linker_residue_set:
                continue
            movemap.set_chi(residue_index, True)
        for residue_index in backbone_residues or []:
            movemap.set_bb(residue_index, True)
            movemap.set_chi(residue_index, True)
        return movemap

    def _save_pose_pdb(self, pose: Pose, filename: str) -> str:
        path = Path(filename)
        pose.dump_pdb(str(path))
        path.write_text(
            path.read_text(encoding="utf-8").replace(str(path), path.name),
            encoding="utf-8",
        )
        logger.info(f"Saved PDB: {filename}")
        return filename

    def _validate_saved_linker_geometry(
        self,
        pdb_file: str,
        linker_chain: str,
        tolerance: float = 0.25,
        environment_min_distance: float = LINKER_ENVIRONMENT_CLASH_MIN_DISTANCE,
        environment_excluded_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> List[str]:
        """Validate the saved linker residue's internal and environment geometry.

        In addition to ``BOND_TYPE`` bond-length checks, every linker heavy atom
        must be at least ``environment_min_distance`` Å from non-linker heavy
        atoms, except declared cross-residue covalent bonds supplied via
        ``environment_excluded_pairs`` as ``(linker_atom_name, non_linker_atom_label)``.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("validated_linker", pdb_file)
        linker_atoms: Dict[str, np.ndarray] = {}
        linker_atom_coords: List[Tuple[str, np.ndarray, str]] = []
        non_linker_atoms: List[Tuple[str, np.ndarray, str]] = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    resseq = residue.id[1]
                    for atom in residue:
                        if atom.element == "H":
                            continue
                        element = normalize_element_symbol(atom.element, atom.name)
                        label = f"{chain.id}:{resseq}:{atom.name}"
                        coord = np.asarray(atom.get_coord(), dtype=float)
                        if chain.id == linker_chain:
                            linker_atoms[atom.name] = coord
                            linker_atom_coords.append((atom.name, coord, element))
                        else:
                            non_linker_atoms.append((label, coord, element))
        if not linker_atoms:
            raise ValueError(
                f"Linker chain '{linker_chain}' not found in saved output {pdb_file}."
            )

        abnormalities: List[str] = []
        for atom1, atom2, order in parse_heavy_bonds_from_params(self.linker_params):
            if atom1 not in linker_atoms or atom2 not in linker_atoms:
                abnormalities.append(f"{atom1}-{atom2}: missing atom in saved linker residue")
                continue
            distance = float(np.linalg.norm(linker_atoms[atom1] - linker_atoms[atom2]))
            ideal = ideal_bond_length_from_atom_names(atom1, atom2, order)
            if not (ideal - tolerance <= distance <= ideal + tolerance):
                abnormalities.append(
                    f"{atom1}-{atom2}: {distance:.3f} A (ideal ~{ideal:.2f}, order={order})"
                )

        if environment_min_distance > 0.0:
            excluded_pair_keys = set(environment_excluded_pairs or [])
            for linker_name, linker_coord, _linker_elem in linker_atom_coords:
                for env_label, env_coord, _env_elem in non_linker_atoms:
                    if (linker_name, env_label) in excluded_pair_keys:
                        continue
                    distance = float(np.linalg.norm(linker_coord - env_coord))
                    if distance < environment_min_distance:
                        abnormalities.append(
                            f"L:{linker_name} vs {env_label}: {distance:.3f} A "
                            f"(< env_min {environment_min_distance:.2f} A)"
                        )
        return abnormalities

    def _get_heavy_neighbor_names(self, pose: Pose, res_idx: int, atom_name: str) -> List[str]:
        residue = pose.residue(res_idx)
        if not residue.has(atom_name):
            return []
        atom_idx = residue.atom_index(atom_name)
        neighbors = []
        for nbr_idx in range(1, residue.natoms() + 1):
            if nbr_idx == atom_idx:
                continue
            if residue.atom_is_hydrogen(nbr_idx):
                continue
            if residue.type().atoms_are_bonded(atom_idx, nbr_idx):
                neighbors.append(residue.atom_name(nbr_idx).strip())
        return neighbors

    def _build_environment_excluded_pairs(
        self,
        pose: Pose,
        linker_chain: str,
        linker_to_warhead_atom: str,
        warhead_anchor_label: str,
        linker_to_e3l_atom: str,
        e3l_anchor_label: str,
    ) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = [
            (linker_to_warhead_atom, warhead_anchor_label),
            (linker_to_e3l_atom, e3l_anchor_label),
        ]

        linker_res_idx = self._get_single_residue_index_for_chain(pose, linker_chain)
        warhead_chain, warhead_res, warhead_icode, warhead_atom = parse_pdb_atom_label(
            warhead_anchor_label
        )
        e3l_chain_id, e3l_res, e3l_icode, e3l_atom = parse_pdb_atom_label(
            e3l_anchor_label
        )
        warhead_res_idx = self._get_pdb_residue_index(
            pose, warhead_chain, warhead_res, warhead_icode
        )
        e3l_res_idx = self._get_pdb_residue_index(
            pose, e3l_chain_id, e3l_res, e3l_icode
        )

        for lnk_nbr in self._get_heavy_neighbor_names(pose, linker_res_idx, linker_to_warhead_atom):
            pairs.append((lnk_nbr, warhead_anchor_label))
        for wh_nbr in self._get_heavy_neighbor_names(pose, warhead_res_idx, warhead_atom):
            wh_label = f"{warhead_chain}:{warhead_res}:{wh_nbr}"
            pairs.append((linker_to_warhead_atom, wh_label))
            for lnk_nbr in self._get_heavy_neighbor_names(pose, linker_res_idx, linker_to_warhead_atom):
                pairs.append((lnk_nbr, wh_label))

        for lnk_nbr in self._get_heavy_neighbor_names(pose, linker_res_idx, linker_to_e3l_atom):
            pairs.append((lnk_nbr, e3l_anchor_label))
        for e3l_nbr in self._get_heavy_neighbor_names(pose, e3l_res_idx, e3l_atom):
            e3l_label = f"{e3l_chain_id}:{e3l_res}:{e3l_nbr}"
            pairs.append((linker_to_e3l_atom, e3l_label))
            for lnk_nbr in self._get_heavy_neighbor_names(pose, linker_res_idx, linker_to_e3l_atom):
                pairs.append((lnk_nbr, e3l_label))

        return pairs

    def _select_interface_residues(
        self,
        pose: Pose,
        linker_chain: str,
        e3l_chain: str,
        interface_cutoff: float,
        excluded_residues: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[int]]:
        excluded_residues = set(excluded_residues or [])
        linker_residues = self._get_chain_residue_indices(pose, linker_chain)
        if not linker_residues:
            raise ValueError(f"Linker chain '{linker_chain}' not found in pose.")

        linker_selector = self._build_residue_selector(linker_residues)
        neighbor_selector = NeighborhoodResidueSelector(
            linker_selector,
            interface_cutoff,
            True,
        )
        neighbor_residues = get_residues_from_subset(neighbor_selector.apply(pose))
        interface_residues = []
        for residue_index in neighbor_residues:
            chain_id = pose.pdb_info().chain(residue_index)
            if chain_id in {linker_chain, e3l_chain}:
                continue
            if residue_index in excluded_residues:
                continue
            interface_residues.append(residue_index)
        interface_residues = sorted(set(interface_residues))
        protein_count = sum(
            1 for residue_index in interface_residues
            if pose.residue(residue_index).is_protein()
        )
        logger.info(
            f"Selected {len(interface_residues)} movable interface residues within "
            f"{interface_cutoff:.1f} Å of linker chain {linker_chain} "
            f"(protein={protein_count}, nonprotein={len(interface_residues) - protein_count})"
        )
        return linker_residues, interface_residues

    def _add_coordinate_constraints_for_residues(
        self,
        pose: Pose,
        residue_indices: List[int],
        sigma: float,
    ) -> None:
        if not residue_indices:
            return
        fixed_ref_atom_id = AtomID(1, 1)
        count = 0
        for residue_index in residue_indices:
            residue = pose.residue(residue_index)
            for atom_index in range(1, residue.natoms() + 1):
                if residue.atom_type(atom_index).is_hydrogen():
                    continue
                atom_id = AtomID(atom_index, residue_index)
                coord_cst = CoordinateConstraint(
                    atom_id,
                    fixed_ref_atom_id,
                    pose.xyz(atom_id),
                    HarmonicFunc(0.0, sigma),
                )
                pose.add_constraint(coord_cst)
                count += 1
        logger.info(
            f"Added {count} heavy-atom coordinate constraints for local clash relief"
        )

    def _repack_interface_sidechains(
        self,
        pose: Pose,
        residue_indices: List[int],
    ) -> None:
        if not residue_indices:
            return

        selected_selector = self._build_residue_selector(residue_indices)
        not_selected = NotResidueSelector(selected_selector)

        task_factory = pyrosetta.rosetta.core.pack.task.TaskFactory()
        task_factory.push_back(
            pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking()
        )
        prevent_repacking = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
        task_factory.push_back(
            pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
                prevent_repacking,
                not_selected,
            )
        )

        scorefxn = create_score_function("ref2015")
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 1.0)

        packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)
        packer.task_factory(task_factory)
        initial_score = scorefxn(pose)
        packer.apply(pose)
        final_score = scorefxn(pose)
        logger.info(
            f"Interface sidechain repack: {initial_score:.2f} -> {final_score:.2f}"
        )

    def _count_linker_environment_clashes(
        self,
        pose: Pose,
        linker_chain: str,
        overlap_tolerance: float,
        excluded_pair_keys: Optional[set] = None,
    ) -> Dict[str, float]:
        return compute_linker_environment_clashes(
            pose,
            linker_chain,
            excluded_pair_keys or set(),
            overlap_tolerance=overlap_tolerance,
        )

    def relieve_local_clashes(
        self,
        pose: Pose,
        linker_chain: str,
        e3l_chain: str,
        warhead_anchor_label: str,
        e3l_anchor_label: str,
        linker_to_warhead_atom: str,
        linker_to_e3l_atom: str,
        interface_cutoff: float,
        overlap_tolerance: float,
        max_iter: int,
        tolerance: float,
    ) -> Tuple[Pose, Dict[str, float], Dict[str, float]]:
        """Local Cartesian minimization at the linker-E3L interface to relieve overlaps."""
        warhead_chain, warhead_residue_number, warhead_icode, _ = parse_pdb_atom_label(
            warhead_anchor_label
        )
        warhead_anchor_residue = self._get_pdb_residue_index(
            pose,
            warhead_chain,
            warhead_residue_number,
            warhead_icode,
        )
        e3l_chain_label, e3l_residue_number, e3l_icode, _ = parse_pdb_atom_label(
            e3l_anchor_label
        )
        if e3l_chain_label != e3l_chain:
            raise ValueError(
                f"E3L chain mismatch: got chain '{e3l_chain}' but anchor label "
                f"points to chain '{e3l_chain_label}'."
            )
        e3l_anchor_residue = self._get_pdb_residue_index(
            pose,
            e3l_chain_label,
            e3l_residue_number,
            e3l_icode,
        )
        excluded_residues = [warhead_anchor_residue, e3l_anchor_residue]
        anchor_residues = [warhead_anchor_residue, e3l_anchor_residue]
        excluded_pair_keys = build_linker_environment_excluded_pair_keys(
            pose,
            linker_chain=linker_chain,
            linker_to_warhead_atom=linker_to_warhead_atom,
            warhead_anchor_label=warhead_anchor_label,
            linker_to_e3l_atom=linker_to_e3l_atom,
            e3l_anchor_label=e3l_anchor_label,
        )

        before = self._count_linker_environment_clashes(
            pose,
            linker_chain=linker_chain,
            overlap_tolerance=overlap_tolerance,
            excluded_pair_keys=excluded_pair_keys,
        )

        linker_residues, interface_residues = self._select_interface_residues(
            pose,
            linker_chain=linker_chain,
            e3l_chain=e3l_chain,
            interface_cutoff=interface_cutoff,
            excluded_residues=excluded_residues,
        )
        protein_interface_residues = [
            residue_index for residue_index in interface_residues
            if pose.residue(residue_index).is_protein()
        ]
        constrained_interface_residues = [
            residue_index for residue_index in interface_residues
            if not pose.residue(residue_index).is_protein()
        ]
        self._add_coordinate_constraints_for_residues(
            pose,
            linker_residues,
            sigma=0.30,
        )
        self._add_coordinate_constraints_for_residues(
            pose,
            anchor_residues,
            sigma=0.15,
        )
        if protein_interface_residues:
            self._repack_interface_sidechains(pose, protein_interface_residues)
            self._add_coordinate_constraints_for_residues(
                pose,
                protein_interface_residues,
                sigma=0.25,
            )
        if constrained_interface_residues:
            self._add_coordinate_constraints_for_residues(
                pose,
                constrained_interface_residues,
                sigma=0.15,
            )
        if not protein_interface_residues and not constrained_interface_residues:
            logger.info("No movable interface residues selected; minimizing linker and anchors only.")

        scorefxn = build_cart_scorefxn("clash")

        movemap = self._build_local_movemap(
            linker_residues=linker_residues,
            sidechain_residues=protein_interface_residues,
            backbone_residues=anchor_residues + constrained_interface_residues,
        )

        min_mover = MinMover()
        min_mover.score_function(scorefxn)
        min_mover.movemap(movemap)
        min_mover.min_type("lbfgs_armijo_nonmonotone")
        min_mover.tolerance(tolerance)
        min_mover.max_iter(max_iter)
        min_mover.cartesian(True)
        initial_score = scorefxn(pose)
        min_mover.apply(pose)
        final_score = scorefxn(pose)

        after = self._count_linker_environment_clashes(
            pose,
            linker_chain=linker_chain,
            overlap_tolerance=overlap_tolerance,
            excluded_pair_keys=excluded_pair_keys,
        )

        message = (
            "Local clash relief: "
            f"score {initial_score:.2f} -> {final_score:.2f}, "
            f"overlaps {before['count']} -> {after['count']}, "
            f"worst_overlap {before['worst_overlap']:.3f} -> {after['worst_overlap']:.3f}, "
            f"min_distance {before['min_distance']:.3f} -> {after['min_distance']:.3f}"
        )
        if (
            after["count"] > before["count"]
            or after["worst_overlap"] > before["worst_overlap"]
            or after["overlap_sum"] > before["overlap_sum"]
            or after["worst_overlap"] >= LINKER_ENVIRONMENT_CLASH_WARNING_OVERLAP
        ):
            logger.warning(message + "; review residual linker-environment overlaps")
        else:
            logger.info(message)
        return pose, before, after
