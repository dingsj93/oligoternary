"""FastRelax stage for linker segments."""
from typing import Optional

import pyrosetta
from loguru import logger
from pyrosetta import Pose
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.scoring.constraints import AngleConstraint, AtomPairConstraint
from pyrosetta.rosetta.core.scoring.func import CircularHarmonicFunc, HarmonicFunc

from oligoternary.modeling.scorefxn import build_cart_scorefxn
from oligoternary.modeling.types import parse_pdb_atom_label


class LinkerRelaxMixin:
    @staticmethod
    def _build_relax_movemap(pose: Pose, linker_chain: str) -> MoveMap:
        """Allow internal linker torsions only; keep every rigid-body jump frozen."""
        movemap = MoveMap()
        movemap.set_bb(False)
        movemap.set_chi(False)
        movemap.set_jump(False)
        for residue_index in range(1, pose.total_residue() + 1):
            if pose.pdb_info().chain(residue_index) != linker_chain:
                continue
            movemap.set_bb(residue_index, True)
            movemap.set_chi(residue_index, True)
        return movemap

    def relax_linker_segment(
        self,
        pose: Pose,
        linker_chain: str,
        e3l_chain: str,
        linker_to_warhead_atom: str,
        linker_to_e3l_atom: str,
        cycles: int,
        warhead_anchor_label: Optional[str] = None,
        e3l_anchor_label: Optional[str] = None,
    ) -> Pose:
        """FastRelax linker chain; freeze all other chains.

        Optional anchor labels add cross-residue AtomPairConstraints (1.6 A, sigma=0.3)
        so the linker cannot drift from warhead/E3L anchors during relax.
        """
        linker_residues = self._get_chain_residue_indices(pose, linker_chain)
        if not linker_residues:
            raise ValueError(f"Linker chain '{linker_chain}' not found in pose.")
        found_atoms = set()
        for residue_index in linker_residues:
            residue = pose.residue(residue_index)
            for atom_name in (linker_to_warhead_atom, linker_to_e3l_atom):
                if residue.has(atom_name):
                    found_atoms.add(atom_name)
        missing_atoms = {
            linker_to_warhead_atom,
            linker_to_e3l_atom,
        } - found_atoms
        if missing_atoms:
            raise ValueError(
                f"Linker chain '{linker_chain}' is missing attachment atom(s): "
                f"{sorted(missing_atoms)}"
            )

        logger.info(
            f"Starting FastRelax on linker segment: linker_chain={linker_chain}, "
            f"e3l_chain={e3l_chain}, cycles={cycles}"
        )

        movemap = self._build_relax_movemap(pose, linker_chain)

        # --- Bond-length constraints for all heavy-atom bonds in the linker ---
        # Read each bond's current length from the initial pose (RDKit-embedded
        # geometry has near-ideal bond lengths) and constrain to that value with
        # a tight Harmonic. Prevents FastRelax from compressing/stretching bonds
        # when fa_rep + strong clash environments push atoms off-ideal.
        _BOND_SIGMA = 0.05
        bond_constraints_added = 0
        for residue_index in linker_residues:
            res = pose.residue(residue_index)
            res_type = res.type()
            n_atoms = res.natoms()
            for atom_i in range(1, n_atoms + 1):
                if res.atom_is_hydrogen(atom_i):
                    continue
                for atom_j in range(atom_i + 1, n_atoms + 1):
                    if res.atom_is_hydrogen(atom_j):
                        continue
                    if not res_type.atoms_are_bonded(atom_i, atom_j):
                        continue
                    initial_dist = (res.xyz(atom_i) - res.xyz(atom_j)).norm()
                    pose.add_constraint(
                        AtomPairConstraint(
                            AtomID(atom_i, residue_index),
                            AtomID(atom_j, residue_index),
                            HarmonicFunc(initial_dist, _BOND_SIGMA),
                        )
                    )
                    bond_constraints_added += 1
        logger.info(
            f"Added {bond_constraints_added} heavy-atom bond-length constraints "
            f"(sigma={_BOND_SIGMA}) on linker residues {linker_residues}"
        )

        # Preserve cross-residue covalent-bond geometry during FastRelax.
        _ANCHOR_TARGET_DIST = 1.6
        _ANCHOR_SIGMA = 0.15
        _anchor_csts_added = 0
        for linker_atom_name, anchor_label in (
            (linker_to_warhead_atom, warhead_anchor_label),
            (linker_to_e3l_atom, e3l_anchor_label),
        ):
            if anchor_label is None:
                continue
            chain_id, res_num, icode, atom_name = parse_pdb_atom_label(anchor_label)
            anchor_res_idx = self._get_pdb_residue_index(pose, chain_id, res_num, icode)
            anchor_residue = pose.residue(anchor_res_idx)
            if not anchor_residue.has(atom_name):
                raise ValueError(
                    f"Anchor residue {anchor_label} is missing atom '{atom_name}'"
                )
            anchor_atom_idx = anchor_residue.atom_index(atom_name)
            # Use the first linker residue that has the attachment atom.
            linker_res_idx = next(
                (r for r in linker_residues if pose.residue(r).has(linker_atom_name)),
                None,
            )
            if linker_res_idx is None:
                continue
            linker_atom_idx = pose.residue(linker_res_idx).atom_index(linker_atom_name)
            pose.add_constraint(
                AtomPairConstraint(
                    AtomID(linker_atom_idx, linker_res_idx),
                    AtomID(anchor_atom_idx, anchor_res_idx),
                    HarmonicFunc(_ANCHOR_TARGET_DIST, _ANCHOR_SIGMA),
                )
            )
            _anchor_csts_added += 1
            logger.info(
                f"Added cross-residue anchor tether: "
                f"LNK.{linker_atom_name} ↔ {anchor_label} "
                f"(target={_ANCHOR_TARGET_DIST} Å, sigma={_ANCHOR_SIGMA})"
            )
        if _anchor_csts_added == 0 and (
            warhead_anchor_label is not None or e3l_anchor_label is not None
        ):
            logger.warning(
                "Anchor labels were provided but no anchor tether constraints "
                "were added — check anchor_label parsing."
            )

        # Stereochemistry-preservation constraints around the linker attachment
        # atoms. We pin the heavy-atom neighbours of ``linker_to_warhead_atom``
        # (and ``linker_to_e3l_atom``) at their **initial** distance to the
        # anchor atom, with sigma=0.5 Å. This locks the rotational orientation
        # of the linker substituents around the attachment atom, preventing the
        # Cartesian min from selecting an inverted-chirality minimum where a
        # non-bonded neighbouring atom ends up between the attachment atom and
        # the anchor, producing a duplicate-atom geometry artefact.
        _CHIRAL_SIGMA = 0.5
        _chiral_csts_added = 0
        for linker_atom_name, anchor_label in (
            (linker_to_warhead_atom, warhead_anchor_label),
            (linker_to_e3l_atom, e3l_anchor_label),
        ):
            if anchor_label is None:
                continue
            chain_id, res_num, icode, atom_name = parse_pdb_atom_label(anchor_label)
            anchor_res_idx = self._get_pdb_residue_index(pose, chain_id, res_num, icode)
            anchor_residue = pose.residue(anchor_res_idx)
            anchor_atom_idx = anchor_residue.atom_index(atom_name)
            anchor_xyz = anchor_residue.xyz(anchor_atom_idx)
            linker_res_idx = next(
                (r for r in linker_residues if pose.residue(r).has(linker_atom_name)),
                None,
            )
            if linker_res_idx is None:
                continue
            linker_residue = pose.residue(linker_res_idx)
            linker_attach_idx = linker_residue.atom_index(linker_atom_name)
            for nbr_name in self._get_heavy_neighbor_names(
                pose, linker_res_idx, linker_atom_name
            ):
                nbr_idx = linker_residue.atom_index(nbr_name)
                # Skip the connecting atom itself (already covered by anchor tether).
                if nbr_idx == linker_attach_idx:
                    continue
                initial_dist = float((linker_residue.xyz(nbr_idx) - anchor_xyz).norm())
                pose.add_constraint(
                    AtomPairConstraint(
                        AtomID(nbr_idx, linker_res_idx),
                        AtomID(anchor_atom_idx, anchor_res_idx),
                        HarmonicFunc(initial_dist, _CHIRAL_SIGMA),
                    )
                )
                _chiral_csts_added += 1
        if _chiral_csts_added:
            logger.info(
                f"Added {_chiral_csts_added} chirality-preservation constraints "
                f"on linker-attachment-atom neighbours (sigma={_CHIRAL_SIGMA})"
            )

        # Preserve initial bond angles around both linker attachment atoms.
        # Intra-residue triplets maintain local geometry; cross-residue triplets
        # maintain substituent orientation relative to each anchor.
        # CircularHarmonicFunc takes radians.
        import math as _math
        _ANGLE_SIGMA_RAD = _math.radians(5.0)
        _angle_csts_added = 0
        for linker_atom_name, anchor_label in (
            (linker_to_warhead_atom, warhead_anchor_label),
            (linker_to_e3l_atom, e3l_anchor_label),
        ):
            linker_res_idx = next(
                (r for r in linker_residues if pose.residue(r).has(linker_atom_name)),
                None,
            )
            if linker_res_idx is None:
                continue
            linker_residue = pose.residue(linker_res_idx)
            attach_idx = linker_residue.atom_index(linker_atom_name)
            attach_xyz = linker_residue.xyz(attach_idx)

            nbr_names = self._get_heavy_neighbor_names(
                pose, linker_res_idx, linker_atom_name
            )
            nbr_records = []  # list of (atom_idx, xyz)
            for nm in nbr_names:
                if not linker_residue.has(nm):
                    continue
                idx = linker_residue.atom_index(nm)
                if idx == attach_idx:
                    continue
                nbr_records.append((idx, linker_residue.xyz(idx)))

            # (a) intra-residue: every neighbour pair around attach atom.
            for i in range(len(nbr_records)):
                for j in range(i + 1, len(nbr_records)):
                    idx_i, xyz_i = nbr_records[i]
                    idx_j, xyz_j = nbr_records[j]
                    v1 = xyz_i - attach_xyz
                    v2 = xyz_j - attach_xyz
                    cos_ang = v1.dot(v2) / (v1.norm() * v2.norm())
                    cos_ang = max(-1.0, min(1.0, cos_ang))
                    target_rad = _math.acos(cos_ang)
                    pose.add_constraint(
                        AngleConstraint(
                            AtomID(idx_i, linker_res_idx),
                            AtomID(attach_idx, linker_res_idx),
                            AtomID(idx_j, linker_res_idx),
                            CircularHarmonicFunc(target_rad, _ANGLE_SIGMA_RAD),
                        )
                    )
                    _angle_csts_added += 1

            # (b) cross-residue: (linker_nbr, attach, anchor) — locks orientation
            #     of linker substituents around the attachment atom.
            # (c) cross-residue: (attach, anchor, anchor_nbr) — locks the sp2/sp3
            #     geometry on the anchor side. Without (c), an sp2 anchor angle
            #     can collapse during Cartesian minimization because the anchor
            #     residue is frozen while the linker attachment atom can rotate
            #     around the bond axis.
            if anchor_label is not None:
                chain_id, res_num, icode, atom_name = parse_pdb_atom_label(anchor_label)
                anchor_res_idx = self._get_pdb_residue_index(
                    pose, chain_id, res_num, icode
                )
                anchor_residue = pose.residue(anchor_res_idx)
                if anchor_residue.has(atom_name):
                    anchor_atom_idx = anchor_residue.atom_index(atom_name)
                    anchor_xyz = anchor_residue.xyz(anchor_atom_idx)
                    # (b) (linker_nbr, LNK.attach, anchor)
                    for idx_i, xyz_i in nbr_records:
                        v1 = xyz_i - attach_xyz
                        v2 = anchor_xyz - attach_xyz
                        cos_ang = v1.dot(v2) / (v1.norm() * v2.norm())
                        cos_ang = max(-1.0, min(1.0, cos_ang))
                        target_rad = _math.acos(cos_ang)
                        pose.add_constraint(
                            AngleConstraint(
                                AtomID(idx_i, linker_res_idx),
                                AtomID(attach_idx, linker_res_idx),
                                AtomID(anchor_atom_idx, anchor_res_idx),
                                CircularHarmonicFunc(target_rad, _ANGLE_SIGMA_RAD),
                            )
                        )
                        _angle_csts_added += 1
                    # (c) Anchor-centered 3-body angles; sp2 anchors use 120 deg targets.
                    anchor_nbr_names = self._get_heavy_neighbor_names(
                        pose, anchor_res_idx, atom_name
                    )
                    # Attach atom first, then anchor heavy-atom neighbors.
                    anchor_side_nbrs = [(attach_idx, linker_res_idx, attach_xyz)]
                    for anbr_nm in anchor_nbr_names:
                        if not anchor_residue.has(anbr_nm):
                            continue
                        anbr_idx_local = anchor_residue.atom_index(anbr_nm)
                        if anbr_idx_local == anchor_atom_idx:
                            continue
                        anchor_side_nbrs.append(
                            (anbr_idx_local, anchor_res_idx, anchor_residue.xyz(anbr_idx_local))
                        )
                    pair_initial = []
                    for ii in range(len(anchor_side_nbrs)):
                        for jj in range(ii + 1, len(anchor_side_nbrs)):
                            v1 = anchor_side_nbrs[ii][2] - anchor_xyz
                            v2 = anchor_side_nbrs[jj][2] - anchor_xyz
                            cos_ang = v1.dot(v2) / (v1.norm() * v2.norm())
                            cos_ang = max(-1.0, min(1.0, cos_ang))
                            pair_initial.append((ii, jj, _math.acos(cos_ang)))
                    # sp2: three neighbors and pairwise angle sum >= 340 deg.
                    is_sp2 = False
                    if len(anchor_side_nbrs) == 3:
                        ang_sum_deg = sum(_math.degrees(a) for _, _, a in pair_initial)
                        if ang_sum_deg >= 340.0:
                            is_sp2 = True
                            logger.info(
                                f"Anchor {anchor_label} detected as sp2 "
                                f"(Σ angles={ang_sum_deg:.1f}°) — using 120° target"
                            )
                    sp2_target_rad = _math.radians(120.0)
                    for ii, jj, init_rad in pair_initial:
                        target_rad = sp2_target_rad if is_sp2 else init_rad
                        atom_i = anchor_side_nbrs[ii]
                        atom_j = anchor_side_nbrs[jj]
                        pose.add_constraint(
                            AngleConstraint(
                                AtomID(atom_i[0], atom_i[1]),
                                AtomID(anchor_atom_idx, anchor_res_idx),
                                AtomID(atom_j[0], atom_j[1]),
                                CircularHarmonicFunc(target_rad, _ANGLE_SIGMA_RAD),
                            )
                        )
                        _angle_csts_added += 1
        if _angle_csts_added:
            logger.info(
                f"Added {_angle_csts_added} bond-angle constraints "
                f"around linker attachment atoms (sigma=5°)"
            )

        scorefxn = build_cart_scorefxn("relax")

        relax = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn, cycles)
        relax.set_movemap(movemap)
        relax.max_iter(50)
        relax.constrain_relax_to_start_coords(True)

        initial_score = scorefxn(pose)
        relax.apply(pose)
        final_score = scorefxn(pose)
        logger.success(
            f"FastRelax complete. Score: {initial_score:.2f} → {final_score:.2f}"
        )
        return pose
