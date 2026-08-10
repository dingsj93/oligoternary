"""Covalent connection and Cartesian minimization stage."""
from loguru import logger
from pyrosetta import Pose
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

from oligoternary.modeling.scorefxn import build_cart_scorefxn
from oligoternary.modeling.types import parse_pdb_atom_label


class LinkerConnectMixin:
    @staticmethod
    def _build_connect_scorefxn():
        return build_cart_scorefxn("connect")

    @staticmethod
    def _build_connect_movemap(pose: Pose, linker_chain: str) -> MoveMap:
        """Connect-stage MoveMap: linker bb+chi free; every other DOF frozen."""
        movemap = MoveMap()
        movemap.set_bb(False)
        movemap.set_chi(False)
        movemap.set_jump(False)
        for i in range(1, pose.total_residue() + 1):
            chain = pose.pdb_info().chain(i)
            if chain == linker_chain:
                movemap.set_bb(i, True)
                movemap.set_chi(i, True)
        return movemap

    def connect_linker_to_anchors(
        self,
        pose: Pose,
        warhead_anchor_label: str,
        e3l_anchor_label: str,
        linker_chain: str,
        linker_to_warhead_atom: str,
        linker_to_e3l_atom: str,
        max_iter: int = 5000,
        tolerance: float = 0.001,
    ) -> Pose:
        """Form warhead/E3L covalent bonds to linker; Cartesian minimize."""
        logger.info(
            f"Connecting linker: warhead_atom={linker_to_warhead_atom}, e3l_atom={linker_to_e3l_atom}"
        )

        warhead_chain, warhead_res, warhead_icode, warhead_atom = parse_pdb_atom_label(
            warhead_anchor_label
        )
        e3l_chain, e3l_res, e3l_icode, e3l_atom = parse_pdb_atom_label(
            e3l_anchor_label
        )

        warhead_res_idx = self._get_pdb_residue_index(
            pose, warhead_chain, warhead_res, warhead_icode,
        )
        e3l_res_idx = self._get_pdb_residue_index(
            pose, e3l_chain, e3l_res, e3l_icode,
        )
        linker_res_idx = self._get_single_residue_index_for_chain(pose, linker_chain)

        # Create covalent bonds
        pose = self._add_covalent_bond(
            pose, warhead_res_idx, warhead_atom, linker_res_idx, linker_to_warhead_atom
        )
        pose = self._add_covalent_bond(
            pose, e3l_res_idx, e3l_atom, linker_res_idx, linker_to_e3l_atom
        )

        # Cartesian minimization with ref2015_cart
        scorefxn = self._build_connect_scorefxn()
        movemap = self._build_connect_movemap(pose, linker_chain)

        min_mover = MinMover()
        min_mover.score_function(scorefxn)
        min_mover.movemap(movemap)
        min_mover.min_type("lbfgs_armijo_nonmonotone")
        min_mover.tolerance(tolerance)
        min_mover.max_iter(max_iter)
        min_mover.cartesian(True)
        min_mover.apply(pose)
        return pose
