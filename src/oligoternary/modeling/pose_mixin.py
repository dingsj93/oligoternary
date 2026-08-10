"""Pose residue lookup helpers."""
from typing import List

from pyrosetta import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector


class PoseUtilsMixin:
    def _get_pdb_residue_index(
        self,
        pose: Pose,
        chain_id: str,
        residue_number: int,
        insertion_code: str = " ",
    ) -> int:
        pdb_info = pose.pdb_info()
        try:
            pose_index = pdb_info.pdb2pose(chain_id, residue_number, insertion_code)
        except TypeError:
            if insertion_code != " ":
                raise ValueError(
                    "This PyRosetta build does not support insertion-code lookup, "
                    f"but anchor requested {chain_id}:{residue_number}{insertion_code}."
                )
            pose_index = pdb_info.pdb2pose(chain_id, residue_number)
        if pose_index <= 0:
            raise ValueError(
                f"Residue {chain_id}:{residue_number}{insertion_code.strip()} not found in pose."
            )
        return pose_index

    def _get_single_residue_index_for_chain(self, pose: Pose, chain_id: str) -> int:
        indices = [
            i for i in range(1, pose.total_residue() + 1)
            if pose.pdb_info().chain(i) == chain_id
        ]
        if len(indices) != 1:
            raise ValueError(
                f"Expected chain '{chain_id}' to contain exactly one residue, found {len(indices)}."
            )
        return indices[0]

    def _get_chain_residue_indices(self, pose: Pose, chain_id: str) -> List[int]:
        return [
            i for i in range(1, pose.total_residue() + 1)
            if pose.pdb_info().chain(i) == chain_id
        ]

    def _build_residue_selector(self, residue_indices: List[int]) -> ResidueIndexSelector:
        selector = ResidueIndexSelector()
        for residue_index in residue_indices:
            selector.append_index(residue_index)
        return selector
