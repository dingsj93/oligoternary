"""RDKit-based constrained linker conformer generation."""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.AllChem import EmbedMolecule, MMFFGetMoleculeForceField, MMFFGetMoleculeProperties
from rdkit.Geometry import Point3D

from oligoternary.modeling.attachments import build_attachment_endpoint_candidates
from oligoternary.modeling.constants import (
    ATTACHMENT_ENDPOINT_CANDIDATES,
    CONFORMER_RANKING,
    CONFORMER_RANKINGS,
    DEFAULT_RANDOM_SEED,
    MIN_ATTACHMENT_ANGLE_DEGREES,
    PREPARE_CONFORMER_CANDIDATES,
)
from oligoternary.modeling.geometry import score_rdkit_mol_environment_clashes
from oligoternary.modeling.pdb import PDBAtomSelector, merge_pdbs
from oligoternary.modeling.types import HeavyAtomRecord, parse_pdb_atom_label


@dataclass
class LinkerConstrainedMinimizer:
    linker_prefix: str
    smiles: str
    start_point: int
    end_point: int
    max_linker_confs: int = 200
    random_seed: int = DEFAULT_RANDOM_SEED
    endpoint_candidates: int = ATTACHMENT_ENDPOINT_CANDIDATES
    conformers_per_endpoint: int = PREPARE_CONFORMER_CANDIDATES
    minimum_attachment_angle_degrees: float = MIN_ATTACHMENT_ANGLE_DEGREES
    conformer_ranking: str = CONFORMER_RANKING
    # Hydrogenated SMILES template; each embed uses Chem.Mol(copy) for reproducibility.
    _mol_template: Chem.Mol = field(init=False, repr=False)

    def __post_init__(self):
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {self.smiles}")
        if not (0 <= self.start_point < mol.GetNumAtoms()):
            raise ValueError(f"start_point {self.start_point} out of range for {mol.GetNumAtoms()} atoms")
        if not (0 <= self.end_point < mol.GetNumAtoms()):
            raise ValueError(f"end_point {self.end_point} out of range for {mol.GetNumAtoms()} atoms")
        if self.endpoint_candidates < 1:
            raise ValueError("endpoint_candidates must be >= 1")
        if self.conformers_per_endpoint < 1:
            raise ValueError("conformers_per_endpoint must be >= 1")
        if not 0 <= self.minimum_attachment_angle_degrees <= 180:
            raise ValueError("minimum_attachment_angle_degrees must be between 0 and 180")
        if self.conformer_ranking not in CONFORMER_RANKINGS:
            raise ValueError(
                "conformer_ranking must be attachment-first or clash-first"
            )
        self._mol_template = Chem.AddHs(mol)

    def _get_molecule(self) -> Chem.Mol:
        return Chem.Mol(self._mol_template)

    def _get_length(self) -> tuple:
        """Return deterministic topological endpoint-distance bounds.

        RDKit's distance-geometry bounds matrix describes the molecule's
        topology.  A finite ETKDG sample is a search result, not a physical
        reachability bound, and can reject valid anchor separations simply
        because an extended conformer was not sampled.
        """
        mol = Chem.Mol(self._mol_template)
        bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
        low_index = max(self.start_point, self.end_point)
        high_index = min(self.start_point, self.end_point)
        # RDKit stores lower bounds below the diagonal and upper bounds above.
        # A small tolerance prevents strict downstream comparisons from
        # rejecting values at the numerical boundary.
        d_min = max(0.0, float(bounds[low_index, high_index]) - 0.25)
        d_max = float(bounds[high_index, low_index]) + 0.25
        if not np.isfinite(d_min) or not np.isfinite(d_max) or d_min >= d_max:
            raise RuntimeError(
                "Invalid linker distance-geometry bounds: "
                f"min={d_min!r}, max={d_max!r}"
            )
        return d_min, d_max

    def _build_coordmap(self, start_coords, end_coords):
        start_coords = [float(x) for x in start_coords]
        end_coords = [float(x) for x in end_coords]
        return {
            self.start_point: Point3D(*start_coords),
            self.end_point: Point3D(*end_coords),
        }

    def _minimize(
        self,
        mol: Chem.Mol,
        coordMap: Dict,
        max_iters: int = 1000,
        seed: int = DEFAULT_RANDOM_SEED,
        remove_hydrogens: bool = True,
    ):
        embed_result = EmbedMolecule(
            mol,
            coordMap=coordMap,
            useRandomCoords=True,
            randomSeed=seed,
        )
        if embed_result < 0:
            raise RuntimeError(f"RDKit embedding failed for seed={seed}")
        mp = MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        ff = MMFFGetMoleculeForceField(mol, mp, confId=0)
        for idx in [self.start_point, self.end_point]:
            ff.AddFixedPoint(idx)
        ff.Initialize()
        ff.Minimize(maxIts=max_iters)
        if remove_hydrogens:
            mol = Chem.RemoveAllHs(mol)
        return mol

    def _score_environment_clashes(
        self,
        mol: Chem.Mol,
        environment_atoms: List[HeavyAtomRecord],
        warhead_anchor_label: str,
        e3l_anchor_label: str,
    ) -> Dict[str, float]:
        """Score linker conformer against the environment (see geometry module)."""
        return score_rdkit_mol_environment_clashes(
            mol,
            environment_atoms,
            start_point=self.start_point,
            end_point=self.end_point,
            warhead_anchor_label=warhead_anchor_label,
            e3l_anchor_label=e3l_anchor_label,
            query_radius=None,
        )

    def _has_valid_attachment_angles(
        self,
        mol: Chem.Mol,
        environment_atoms: List[HeavyAtomRecord],
        warhead_anchor_label: str,
        e3l_anchor_label: str,
    ) -> bool:
        anchors = {atom.label: atom.coord for atom in environment_atoms}
        conformer = mol.GetConformer()
        for endpoint, anchor_label in (
            (self.start_point, warhead_anchor_label),
            (self.end_point, e3l_anchor_label),
        ):
            endpoint_point = conformer.GetAtomPosition(endpoint)
            endpoint_position = np.asarray(
                [endpoint_point.x, endpoint_point.y, endpoint_point.z], dtype=float
            )
            anchor_vector = anchors[anchor_label] - endpoint_position
            for neighbor in mol.GetAtomWithIdx(endpoint).GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    continue
                neighbor_point = conformer.GetAtomPosition(neighbor.GetIdx())
                neighbor_position = np.asarray(
                    [neighbor_point.x, neighbor_point.y, neighbor_point.z], dtype=float
                )
                neighbor_vector = neighbor_position - endpoint_position
                cosine = np.dot(anchor_vector, neighbor_vector) / (
                    np.linalg.norm(anchor_vector) * np.linalg.norm(neighbor_vector)
                )
                angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                if angle < self.minimum_attachment_angle_degrees:
                    return False
        return True

    def _select_best_conformer(
        self,
        coord_maps: List[Dict],
        environment_atoms: List[HeavyAtomRecord],
        warhead_anchor_label: str,
        e3l_anchor_label: str,
        minimize_steps: int,
        seed: int,
        conformers_per_coord_map: Optional[int] = None,
    ) -> Chem.Mol:
        """Joint search over (attachment endpoint, RDKit conformer).

        For every candidate ``(start_point, end_point)`` placement produced by
        :func:`build_attachment_endpoint_candidates`, embed
        ``conformers_per_coord_map`` RDKit conformers, and pick the globally
        lowest-clash result using a dictionary-ordered score.

        Candidates below ``minimum_attachment_angle_degrees`` are rejected.
        ``clash-first`` prioritizes the worst environment overlap, whereas
        ``attachment-first`` prioritizes overlap at the two anchor atoms.
        """
        if conformers_per_coord_map is None:
            conformers_per_coord_map = self.conformers_per_endpoint
        if not coord_maps:
            raise ValueError("coord_maps must contain at least one CoordMap.")
        if conformers_per_coord_map < 1:
            raise ValueError("conformers_per_coord_map must be >= 1")

        best_mol = None
        best_score = None
        best_stats = None
        best_coord_map_index = None
        rejected_angles = 0
        for coord_map_index, coord_map in enumerate(coord_maps):
            for offset in range(conformers_per_coord_map):
                candidate_seed = seed + coord_map_index * 1000 + offset
                candidate = self._minimize(
                    mol=self._get_molecule(),
                    coordMap=coord_map,
                    max_iters=minimize_steps,
                    seed=candidate_seed,
                    remove_hydrogens=False,
                )
                if not self._has_valid_attachment_angles(
                    candidate,
                    environment_atoms,
                    warhead_anchor_label,
                    e3l_anchor_label,
                ):
                    rejected_angles += 1
                    continue
                clash_stats = self._score_environment_clashes(
                    mol=candidate,
                    environment_atoms=environment_atoms,
                    warhead_anchor_label=warhead_anchor_label,
                    e3l_anchor_label=e3l_anchor_label,
                )
                if self.conformer_ranking == "attachment-first":
                    score = (
                        clash_stats["anchor_overlap"],
                        clash_stats["worst_overlap"],
                        clash_stats["overlap_sum"],
                        clash_stats["count"],
                        -clash_stats["min_distance"],
                    )
                else:
                    score = (
                        clash_stats["worst_overlap"],
                        clash_stats["overlap_sum"],
                        clash_stats["count"],
                        clash_stats["anchor_overlap"],
                        -clash_stats["min_distance"],
                    )
                if best_score is None or score < best_score:
                    best_score = score
                    best_mol = candidate
                    best_stats = clash_stats
                    best_coord_map_index = coord_map_index

        if best_mol is None or best_stats is None:
            raise RuntimeError(
                "No constrained linker conformer passed the attachment-angle check."
            )

        logger.info(
            "Selected best constrained linker conformer "
            f"(coord_map={best_coord_map_index + 1}/{len(coord_maps)}): "
            f"anchor_overlap={best_stats['anchor_overlap']:.3f}, "
            f"worst_overlap={best_stats['worst_overlap']:.3f}, "
            f"overlap_sum={best_stats['overlap_sum']:.3f}, "
            f"clashes={best_stats['count']}, "
            f"min_distance={best_stats['min_distance']:.3f}, "
            f"rejected_angles={rejected_angles}"
        )
        return Chem.RemoveAllHs(best_mol)

    def _save_molecule(self, mol: Chem.Mol, pdb_file: str):
        Chem.MolToPDBFile(mol, pdb_file)
        return pdb_file

    def get_linker_conformer(
        self,
        pdb_file: str,
        linker_chain: str,
        e3l_chain: str,
        warhead_anchor_label: str,
        e3l_anchor_label: str,
        linker_to_warhead_atom: str,
        linker_to_e3l_atom: str,
        linker_min_distance: float,
        linker_max_distance: float,
        output_dir: str,
        output_prefix: str,
        minimize_steps: int = 1000,
        seed: int = DEFAULT_RANDOM_SEED,
    ):
        """Embed a distance-constrained linker conformer and merge into the complex PDB."""
        anchor_locator = PDBAtomSelector(pdb_file)
        warhead_coord, e3l_coord, distance = anchor_locator.get_warhead_e3l_linked_coordinates(
            warhead_anchor_label, e3l_anchor_label
        )
        environment_atoms = anchor_locator.get_heavy_atoms()
        warhead_anchor_atom = parse_pdb_atom_label(warhead_anchor_label)[3]
        e3l_anchor_atom = parse_pdb_atom_label(e3l_anchor_label)[3]
        endpoint_candidates = build_attachment_endpoint_candidates(
            warhead_coord=warhead_coord,
            e3l_coord=e3l_coord,
            warhead_anchor_atom=warhead_anchor_atom,
            e3l_anchor_atom=e3l_anchor_atom,
            linker_to_warhead_atom=linker_to_warhead_atom,
            linker_to_e3l_atom=linker_to_e3l_atom,
            environment_atoms=environment_atoms,
            warhead_anchor_label=warhead_anchor_label,
            e3l_anchor_label=e3l_anchor_label,
            linker_min_distance=linker_min_distance,
            linker_max_distance=linker_max_distance,
            top_k=self.endpoint_candidates,
        )
        coord_maps = [
            self._build_coordmap(start, end)
            for start, end, _attachment_distance in endpoint_candidates
        ]
        attachment_distances = [entry[2] for entry in endpoint_candidates]
        logger.info(
            f"Anchor distance: {distance:.2f} Å, "
            f"{len(endpoint_candidates)} endpoint candidate(s), "
            f"{self.conformers_per_endpoint} conformer(s) per endpoint, "
            f"ranking={self.conformer_ranking}, "
            f"minimum attachment angle={self.minimum_attachment_angle_degrees:g}°, "
            f"attachment distances: "
            f"{[f'{d:.2f}' for d in attachment_distances]} Å, "
            f"linker range: [{linker_min_distance:.2f}, {linker_max_distance:.2f}]"
        )
        mol = self._select_best_conformer(
            coord_maps=coord_maps,
            environment_atoms=environment_atoms,
            warhead_anchor_label=warhead_anchor_label,
            e3l_anchor_label=e3l_anchor_label,
            minimize_steps=minimize_steps,
            seed=seed,
        )
        linker_file = os.path.join(output_dir, f"{output_prefix}_{self.linker_prefix}_conformer.pdb")
        self._save_molecule(mol, linker_file)
        logger.success(f"Saved linker conformer to {linker_file}")
        complex_file = os.path.join(output_dir, f"{output_prefix}_{self.linker_prefix}_complex.pdb")
        merge_pdbs(pdb_file, linker_file, linker_chain, e3l_chain, complex_file)
        logger.success(f"Saved complex to {complex_file}")
        return complex_file
