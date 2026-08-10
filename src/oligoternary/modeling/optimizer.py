"""Rosetta linker optimizer orchestrator."""
import math
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger
from pyrosetta import Pose, pose_from_pdb
from pyrosetta.rosetta.core.id import AtomID

from oligoternary.modeling.bonding_mixin import BondingMixin
from oligoternary.modeling.clash_mixin import LinkerClashMixin
from oligoternary.modeling.connect_mixin import LinkerConnectMixin
from oligoternary.modeling.constants import MIN_ATTACHMENT_ANGLE_DEGREES
from oligoternary.modeling.pose_mixin import PoseUtilsMixin
from oligoternary.modeling.params_io import ideal_bond_length_from_atom_names
from oligoternary.modeling.relax_mixin import LinkerRelaxMixin
from oligoternary.modeling.runtime import (
    build_pyrosetta_init_options,
    ensure_pyrosetta_initialized,
)
from oligoternary.modeling.types import parse_pdb_atom_label


@dataclass
class RosettaLinkerOptimizer(
    LinkerConnectMixin,
    LinkerClashMixin,
    LinkerRelaxMixin,
    BondingMixin,
    PoseUtilsMixin,
):
    linker_prefix: str
    linker_params: str
    e3l_params: str
    random_seed: int
    pdb_file: str = ""
    last_clash_stats: Optional[Dict[str, Dict[str, float]]] = field(default=None, repr=False)

    def __post_init__(self):
        if not self.linker_params or not self.e3l_params:
            raise FileNotFoundError("Linker or E3L params file not found.")

    def _initialize_pyrosetta(self) -> None:
        ensure_pyrosetta_initialized(
            build_pyrosetta_init_options(
                linker_params=self.linker_params,
                e3l_params=self.e3l_params,
                random_seed=self.random_seed,
            )
        )

    def _load_pose(self) -> Pose:
        if not os.path.exists(self.pdb_file):
            raise FileNotFoundError(f"PDB file not found: '{self.pdb_file}'")
        pose = pose_from_pdb(self.pdb_file)
        logger.success(f"Loaded PDB: {self.pdb_file} ({pose.total_residue()} residues)")
        return pose

    @staticmethod
    def _params_bond_neighbors(params_file: str) -> Dict[str, Set[str]]:
        """Return the declared BOND_TYPE neighbor set for each params atom."""
        neighbors: Dict[str, Set[str]] = {}
        if not os.path.isfile(params_file):
            return neighbors
        with open(params_file, encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("BOND_TYPE"):
                    continue
                fields = line.split()
                if len(fields) < 4:
                    continue
                atom1, atom2 = fields[1], fields[2]
                neighbors.setdefault(atom1, set()).add(atom2)
                neighbors.setdefault(atom2, set()).add(atom1)
        return neighbors

    def _validate_covalent_attachments(
        self,
        pose: Pose,
        *,
        linker_chain: str,
        linker_to_warhead_atom: str,
        warhead_anchor_label: str,
        linker_to_e3l_atom: str,
        e3l_anchor_label: str,
        distance_tolerance: float = 0.45,
        minimum_attachment_angle_degrees: float = MIN_ATTACHMENT_ANGLE_DEGREES,
    ) -> List[str]:
        """Validate final attachment topology, distance, and marked-residue valence."""
        linker_res_idx = self._get_single_residue_index_for_chain(pose, linker_chain)
        pair_specs = []
        for pair_name, linker_atom, anchor_label in (
            ("warhead-linker", linker_to_warhead_atom, warhead_anchor_label),
            ("e3l-linker", linker_to_e3l_atom, e3l_anchor_label),
        ):
            chain, resnum, icode, anchor_atom = parse_pdb_atom_label(anchor_label)
            anchor_res_idx = self._get_pdb_residue_index(pose, chain, resnum, icode)
            pair_specs.append(
                (pair_name, anchor_res_idx, anchor_atom, linker_res_idx, linker_atom)
            )

        issues: List[str] = []
        conformation = pose.conformation()
        for pair_name, anchor_res_idx, anchor_atom, lnk_res_idx, linker_atom in pair_specs:
            anchor_residue = pose.residue(anchor_res_idx)
            linker_residue = pose.residue(lnk_res_idx)
            missing = []
            if not anchor_residue.has(anchor_atom):
                missing.append(f"anchor atom {anchor_atom}")
            if not linker_residue.has(linker_atom):
                missing.append(f"linker atom {linker_atom}")
            if missing:
                issues.append(f"{pair_name}: missing " + " and ".join(missing))
                continue

            anchor_atom_idx = anchor_residue.atom_index(anchor_atom)
            linker_atom_idx = linker_residue.atom_index(linker_atom)
            anchor_atom_id = AtomID(anchor_atom_idx, anchor_res_idx)
            linker_atom_id = AtomID(linker_atom_idx, lnk_res_idx)
            if not conformation.atoms_are_bonded(anchor_atom_id, linker_atom_id):
                issues.append(
                    f"{pair_name}: Pose topology does not contain "
                    f"{anchor_atom}-{linker_atom}"
                )
                continue

            anchor_connections = list(anchor_residue.connections_to_residue(lnk_res_idx))
            linker_connections = list(linker_residue.connections_to_residue(anchor_res_idx))
            if len(anchor_connections) != 1 or len(linker_connections) != 1:
                issues.append(
                    f"{pair_name}: expected one reciprocal residue connection, got "
                    f"{len(anchor_connections)}/{len(linker_connections)}"
                )
            else:
                anchor_connect_atom = anchor_residue.residue_connect_atom_index(
                    anchor_connections[0]
                )
                linker_connect_atom = linker_residue.residue_connect_atom_index(
                    linker_connections[0]
                )
                if (
                    anchor_connect_atom != anchor_atom_idx
                    or linker_connect_atom != linker_atom_idx
                ):
                    issues.append(
                        f"{pair_name}: residue connection uses the wrong atom(s)"
                    )

            distance = float(pose.xyz(anchor_atom_id).distance(pose.xyz(linker_atom_id)))
            ideal = ideal_bond_length_from_atom_names(anchor_atom, linker_atom, 1)
            if abs(distance - ideal) > distance_tolerance:
                issues.append(
                    f"{pair_name}: bond distance {distance:.3f} A is outside "
                    f"{ideal:.2f} +/- {distance_tolerance:.2f} A"
                )

            anchor_xyz = pose.xyz(anchor_atom_id)
            linker_xyz = pose.xyz(linker_atom_id)
            anchor_vector = anchor_xyz - linker_xyz
            for neighbor_idx in linker_residue.bonded_neighbor(linker_atom_idx):
                atom_type = linker_residue.atom_type(neighbor_idx)
                if atom_type.is_hydrogen() or atom_type.is_virtual():
                    continue
                neighbor_vector = linker_residue.xyz(neighbor_idx) - linker_xyz
                cosine = anchor_vector.dot(neighbor_vector) / (
                    anchor_vector.norm() * neighbor_vector.norm()
                )
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                if angle < minimum_attachment_angle_degrees:
                    neighbor_name = linker_residue.atom_name(neighbor_idx).strip()
                    issues.append(
                        f"{pair_name}: {neighbor_name}-{linker_atom}-{anchor_atom} "
                        f"angle {angle:.1f} degrees is below "
                        f"{minimum_attachment_angle_degrees:.1f} degrees"
                    )

        # Marked params define retained hydrogens and internal neighbor counts.
        # Each declared attachment adds exactly one external neighbor.
        linker_neighbors = self._params_bond_neighbors(self.linker_params)
        e3l_neighbors = self._params_bond_neighbors(self.e3l_params)
        valence_specs: Dict[Tuple[int, str, str], Tuple[Dict[str, Set[str]], int]] = {}
        for _name, _anchor_res, _anchor_atom, lnk_res_idx, linker_atom in pair_specs:
            key = (lnk_res_idx, linker_atom, "linker")
            prior_count = valence_specs.get(key, (linker_neighbors, 0))[1]
            valence_specs[key] = (linker_neighbors, prior_count + 1)
        _name, e3l_res_idx, e3l_atom, _lnk_res_idx, _lnk_atom = pair_specs[1]
        valence_specs[(e3l_res_idx, e3l_atom, "e3l")] = (e3l_neighbors, 1)

        for (residue_index, atom_name, role), (params_neighbors, external_count) in valence_specs.items():
            declared_neighbors = params_neighbors.get(atom_name)
            if declared_neighbors is None:
                continue
            residue = pose.residue(residue_index)
            if not residue.has(atom_name):
                continue
            atom_index = residue.atom_index(atom_name)
            expected_degree = len(declared_neighbors) + external_count
            actual_degree = int(residue.n_bonded_neighbor_all_res(atom_index))
            if actual_degree != expected_degree:
                issues.append(
                    f"{role} {atom_name}: expected valence degree {expected_degree} "
                    f"from params plus attachment(s), got {actual_degree}"
                )
            for hydrogen_name in sorted(
                neighbor for neighbor in declared_neighbors if neighbor.startswith("H")
            ):
                if not residue.has(hydrogen_name):
                    issues.append(
                        f"{role} {atom_name}: params hydrogen {hydrogen_name} is missing"
                    )
                    continue
                hydrogen_id = AtomID(residue.atom_index(hydrogen_name), residue_index)
                atom_id = AtomID(atom_index, residue_index)
                if not conformation.atoms_are_bonded(atom_id, hydrogen_id):
                    issues.append(
                        f"{role} {atom_name}: params hydrogen {hydrogen_name} is not bonded"
                    )
        return issues

    @staticmethod
    def _validate_saved_attachment_conect(
        pdb_file: str,
        *,
        linker_chain: str,
        linker_to_warhead_atom: str,
        warhead_anchor_label: str,
        linker_to_e3l_atom: str,
        e3l_anchor_label: str,
    ) -> List[str]:
        """Require one reciprocal PDB CONECT entry for each attachment pair."""
        atoms_by_identity: Dict[Tuple[str, int, str, str], List[int]] = {}
        linker_serials: Dict[str, List[int]] = {
            linker_to_warhead_atom: [],
            linker_to_e3l_atom: [],
        }
        conect: Dict[int, List[int]] = {}
        with open(pdb_file, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(("ATOM  ", "HETATM")):
                    try:
                        serial = int(line[6:11])
                        chain = line[21]
                        resnum = int(line[22:26])
                    except (ValueError, IndexError):
                        continue
                    icode = line[26] if len(line) > 26 and line[26].strip() else " "
                    atom_name = line[12:16].strip()
                    atoms_by_identity.setdefault(
                        (chain, resnum, icode, atom_name), []
                    ).append(serial)
                    if chain == linker_chain and atom_name in linker_serials:
                        linker_serials[atom_name].append(serial)
                elif line.startswith("CONECT"):
                    fields = line.split()
                    try:
                        source = int(fields[1])
                        targets = [int(value) for value in fields[2:]]
                    except (ValueError, IndexError):
                        continue
                    conect.setdefault(source, []).extend(targets)

        issues: List[str] = []
        for pair_name, linker_atom, anchor_label in (
            ("warhead-linker", linker_to_warhead_atom, warhead_anchor_label),
            ("e3l-linker", linker_to_e3l_atom, e3l_anchor_label),
        ):
            anchor_key = parse_pdb_atom_label(anchor_label)
            anchor_serial_candidates = atoms_by_identity.get(anchor_key, [])
            linker_serial_candidates = linker_serials.get(linker_atom, [])
            if len(anchor_serial_candidates) != 1:
                issues.append(
                    f"{pair_name}: saved PDB contains {len(anchor_serial_candidates)} "
                    f"atom records for {anchor_label}"
                )
                continue
            if len(linker_serial_candidates) != 1:
                issues.append(
                    f"{pair_name}: saved PDB contains {len(linker_serial_candidates)} "
                    f"atom records for {linker_chain}:{linker_atom}"
                )
                continue
            anchor_serial = anchor_serial_candidates[0]
            linker_serial = linker_serial_candidates[0]
            forward_count = conect.get(anchor_serial, []).count(linker_serial)
            reverse_count = conect.get(linker_serial, []).count(anchor_serial)
            if forward_count != 1 or reverse_count != 1:
                issues.append(
                    f"{pair_name}: saved PDB requires one reciprocal CONECT entry, "
                    f"got {forward_count}/{reverse_count}"
                )
        return issues

    @staticmethod
    def _quarantine_failed_output(output_pdb: str) -> None:
        """Move an invalid model PDB out of models/ into models/_failed/."""
        if not os.path.isfile(output_pdb):
            return
        failed_dir = os.path.join(os.path.dirname(output_pdb), "_failed")
        os.makedirs(failed_dir, exist_ok=True)
        destination = os.path.join(failed_dir, os.path.basename(output_pdb))
        shutil.move(output_pdb, destination)
        logger.warning(f"Quarantined invalid output: {destination}")

    def fit(
        self,
        pdb_file: str,
        output_dir: str,
        output_prefix: str,
        warhead_anchor_label: str,
        e3l_anchor_label: str,
        linker_chain: str,
        linker_to_warhead_atom: str,
        linker_to_e3l_atom: str,
        relax_cycles: int = 10,
        minimize_steps: int = 1000,
        minimize_tolerance: float = 0.001,
        minimum_attachment_angle_degrees: float = MIN_ATTACHMENT_ANGLE_DEGREES,
    ) -> Tuple[Pose, str]:
        """Load, relax, connect, relieve clashes, save, and validate."""
        self._initialize_pyrosetta()
        self.pdb_file = pdb_file
        self.last_clash_stats = None
        pose = self._load_pose()

        e3l_chain = parse_pdb_atom_label(e3l_anchor_label)[0]

        pose = self.relax_linker_segment(
            pose, linker_chain, e3l_chain,
            linker_to_warhead_atom, linker_to_e3l_atom, relax_cycles,
            warhead_anchor_label=warhead_anchor_label,
            e3l_anchor_label=e3l_anchor_label,
        )

        pose = self.connect_linker_to_anchors(
            pose, warhead_anchor_label, e3l_anchor_label, linker_chain,
            linker_to_warhead_atom, linker_to_e3l_atom,
            minimize_steps, minimize_tolerance,
        )
        pose, clash_before, clash_after = self.relieve_local_clashes(
            pose,
            linker_chain=linker_chain,
            e3l_chain=e3l_chain,
            warhead_anchor_label=warhead_anchor_label,
            e3l_anchor_label=e3l_anchor_label,
            linker_to_warhead_atom=linker_to_warhead_atom,
            linker_to_e3l_atom=linker_to_e3l_atom,
            interface_cutoff=6.0,
            overlap_tolerance=0.0,
            max_iter=minimize_steps,
            tolerance=minimize_tolerance,
        )
        self.last_clash_stats = {
            "before": clash_before,
            "after": clash_after,
        }

        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{output_prefix}_{self.linker_prefix}_full_optimized.pdb")
        self._save_pose_pdb(pose, filename)
        try:
            attachment_issues = self._validate_covalent_attachments(
                pose,
                linker_chain=linker_chain,
                linker_to_warhead_atom=linker_to_warhead_atom,
                warhead_anchor_label=warhead_anchor_label,
                linker_to_e3l_atom=linker_to_e3l_atom,
                e3l_anchor_label=e3l_anchor_label,
                minimum_attachment_angle_degrees=minimum_attachment_angle_degrees,
            )
            if attachment_issues:
                raise RuntimeError(
                    f"Invalid covalent attachment state in Rosetta output {filename}: "
                    + "; ".join(attachment_issues[:8])
                )
            saved_conect_issues = self._validate_saved_attachment_conect(
                filename,
                linker_chain=linker_chain,
                linker_to_warhead_atom=linker_to_warhead_atom,
                warhead_anchor_label=warhead_anchor_label,
                linker_to_e3l_atom=linker_to_e3l_atom,
                e3l_anchor_label=e3l_anchor_label,
            )
            if saved_conect_issues:
                raise RuntimeError(
                    f"Invalid serialized covalent topology in Rosetta output {filename}: "
                    + "; ".join(saved_conect_issues[:8])
                )
            saved_pose = pose_from_pdb(filename)
            reloaded_attachment_issues = self._validate_covalent_attachments(
                saved_pose,
                linker_chain=linker_chain,
                linker_to_warhead_atom=linker_to_warhead_atom,
                warhead_anchor_label=warhead_anchor_label,
                linker_to_e3l_atom=linker_to_e3l_atom,
                e3l_anchor_label=e3l_anchor_label,
                minimum_attachment_angle_degrees=minimum_attachment_angle_degrees,
            )
            if reloaded_attachment_issues:
                raise RuntimeError(
                    f"Invalid reloaded covalent attachment state in {filename}: "
                    + "; ".join(reloaded_attachment_issues[:8])
                )
            environment_excluded_pairs = self._build_environment_excluded_pairs(
                pose,
                linker_chain=linker_chain,
                linker_to_warhead_atom=linker_to_warhead_atom,
                warhead_anchor_label=warhead_anchor_label,
                linker_to_e3l_atom=linker_to_e3l_atom,
                e3l_anchor_label=e3l_anchor_label,
            )
            linker_geometry_issues = self._validate_saved_linker_geometry(
                filename,
                linker_chain=linker_chain,
                environment_excluded_pairs=environment_excluded_pairs,
            )
            if linker_geometry_issues:
                preview = "; ".join(linker_geometry_issues[:5])
                raise RuntimeError(
                    f"Invalid linker geometry in Rosetta output {filename}: "
                    f"{len(linker_geometry_issues)} abnormal heavy-atom bonds or "
                    f"environment clashes. Examples: {preview}"
                )
            return pose, filename
        except Exception:
            self._quarantine_failed_output(filename)
            raise
