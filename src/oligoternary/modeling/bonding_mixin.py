"""Covalent bond creation between linker and anchors."""
from loguru import logger
from pyrosetta import Pose
from pyrosetta.rosetta.core.pose import (
    remove_lower_terminus_type_from_pose_residue,
    remove_upper_terminus_type_from_pose_residue,
)


class BondingMixin:
    def _add_covalent_bond(
        self, pose: Pose, pos1: int, atom1: str, pos2: int, atom2: str
    ) -> Pose:
        residue1 = pose.residue(pos1)
        residue2 = pose.residue(pos2)
        if not residue1.has(atom1):
            raise ValueError(f"Residue {pos1} is missing connect atom '{atom1}'")
        if not residue2.has(atom2):
            raise ValueError(f"Residue {pos2} is missing connect atom '{atom2}'")

        # Strip terminal-end variants before declaring the bond so a terminal
        # RNA/DNA residue exposes its normal connection atom.  Marked ligand
        # params already encode the correct hydrogens on their CONNECT atoms;
        # deleting one proton from each side would turn, for example, an
        # E3L-NH--LNK-CH2 junction into E3L-N--LNK-CH.
        self._strip_terminus_variants_for_connected_residue(pose, pos1)
        self._strip_terminus_variants_for_connected_residue(pose, pos2)
        pose.conformation().declare_chemical_bond(pos1, atom1, pos2, atom2)

        return pose

    def _strip_terminus_variants_for_connected_residue(
        self, pose: Pose, residue_index: int
    ) -> None:
        residue = pose.residue(residue_index)
        if residue.is_lower_terminus():
            remove_lower_terminus_type_from_pose_residue(pose, residue_index)
            logger.info(
                f"Stripped LOWER_TERMINUS variant from residue {residue_index} "
                f"({residue.name3()}) before covalent linkage"
            )
        if pose.residue(residue_index).is_upper_terminus():
            remove_upper_terminus_type_from_pose_residue(pose, residue_index)
            logger.info(
                f"Stripped UPPER_TERMINUS variant from residue {residue_index} "
                f"({pose.residue(residue_index).name3()}) before covalent linkage"
            )
