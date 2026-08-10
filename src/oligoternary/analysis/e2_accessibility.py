"""Geometry-based E2 active-site accessibility screening."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1


@dataclass(frozen=True)
class LysineDistance:
    """Distance from one target lysine N-zeta atom to the E2 active site."""

    residue_number: int
    insertion_code: str
    distance_angstrom: float
    in_range: bool


@dataclass(frozen=True)
class E2AccessibilityScreenResult:
    """E2 active-site accessibility result for one structural model."""

    input_structure: str
    reference_structure: str
    poi_chain: str
    mobile_e3_chain: str
    reference_e3_chain: str
    e2_chain: str
    e2_residue: int
    e2_atom: str
    alignment_residue_count: int
    alignment_rmsd_angstrom: float
    minimum_alignment_residues: int
    maximum_alignment_rmsd_angstrom: float
    alignment_compatible: bool
    minimum_poi_e2_distance_angstrom: float
    contacts_below_cutoff: int
    severe_contacts: int
    contact_cutoff_angstrom: float
    severe_cutoff_angstrom: float
    minimum_separation_angstrom: float
    maximum_contacts: int
    sterically_compatible: bool
    lysine_minimum_distance_angstrom: float
    lysine_maximum_distance_angstrom: float
    lysines: tuple[LysineDistance, ...]
    passed: bool

    def to_dict(self) -> dict:
        """Return a stable, human-readable result document."""

        in_range = [lysine for lysine in self.lysines if lysine.in_range]
        closest = min(self.lysines, key=lambda lysine: lysine.distance_angstrom)
        return {
            "schema_version": 1,
            "screen": "e2-active-site-accessibility",
            "input_structure": self.input_structure,
            "reference_structure": self.reference_structure,
            "alignment": {
                "mobile_e3_chain": self.mobile_e3_chain,
                "reference_e3_chain": self.reference_e3_chain,
                "residue_count": self.alignment_residue_count,
                "rmsd_angstrom": round(self.alignment_rmsd_angstrom, 3),
                "minimum_residues": self.minimum_alignment_residues,
                "maximum_rmsd_angstrom": self.maximum_alignment_rmsd_angstrom,
                "passed": self.alignment_compatible,
            },
            "e2_active_site": {
                "chain": self.e2_chain,
                "residue": self.e2_residue,
                "atom": self.e2_atom,
            },
            "steric_screen": {
                "poi_chain": self.poi_chain,
                "minimum_poi_e2_distance_angstrom": round(
                    self.minimum_poi_e2_distance_angstrom, 3
                ),
                "minimum_separation_angstrom": self.minimum_separation_angstrom,
                "contacts_below_cutoff": self.contacts_below_cutoff,
                "contact_cutoff_angstrom": self.contact_cutoff_angstrom,
                "maximum_contacts": self.maximum_contacts,
                "severe_contacts": self.severe_contacts,
                "severe_cutoff_angstrom": self.severe_cutoff_angstrom,
                "passed": self.sterically_compatible,
            },
            "lysine_screen": {
                "distance_window_angstrom": [
                    self.lysine_minimum_distance_angstrom,
                    self.lysine_maximum_distance_angstrom,
                ],
                "measured_count": len(self.lysines),
                "in_range_count": len(in_range),
                "closest_lysine": asdict(closest),
                "lysines": [asdict(lysine) for lysine in self.lysines],
                "passed": bool(in_range),
            },
            "passed": self.passed,
        }


def _parse_pdb(path: Path):
    handle = gzip.open(path, "rt") if path.suffix == ".gz" else path.open()
    with handle:
        return PDBParser(QUIET=True).get_structure(path.name, handle)[0]


def _chain(model, chain_id: str, structure_name: str):
    if chain_id not in model:
        raise ValueError(f"chain {chain_id!r} is absent from {structure_name}")
    return model[chain_id]


def _protein_residues(chain) -> list:
    return [
        residue
        for residue in chain
        if is_aa(residue, standard=False) and "CA" in residue
    ]


def _alignment_transform(
    mobile_chain, reference_chain
) -> tuple[np.ndarray, np.ndarray, int, float]:
    mobile_residues = _protein_residues(mobile_chain)
    reference_residues = _protein_residues(reference_chain)
    mobile_sequence = "".join(seq1(residue.resname) for residue in mobile_residues)
    reference_sequence = "".join(seq1(residue.resname) for residue in reference_residues)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(mobile_sequence, reference_sequence)[0]

    mobile_atoms = []
    reference_atoms = []
    for mobile_block, reference_block in zip(*alignment.aligned):
        block_length = mobile_block[1] - mobile_block[0]
        for offset in range(block_length):
            mobile_atoms.append(mobile_residues[mobile_block[0] + offset]["CA"])
            reference_atoms.append(
                reference_residues[reference_block[0] + offset]["CA"]
            )
    if len(mobile_atoms) < 3:
        raise ValueError("E3 chains do not provide at least three aligned C-alpha atoms")

    superimposer = Superimposer()
    superimposer.set_atoms(mobile_atoms, reference_atoms)
    rotation, translation = superimposer.rotran
    return rotation, translation, len(mobile_atoms), float(superimposer.rms)


def _heavy_atom_coordinates(chain) -> np.ndarray:
    coordinates = [
        atom.coord
        for atom in chain.get_atoms()
        if atom.element.upper() != "H"
    ]
    return np.asarray(coordinates, dtype=float)


def _distance_statistics(
    poi_coordinates: np.ndarray,
    e2_coordinates: np.ndarray,
    contact_cutoff: float,
    severe_cutoff: float,
) -> tuple[float, int, int]:
    minimum_squared = float("inf")
    contact_count = 0
    severe_count = 0
    for start in range(0, len(poi_coordinates), 256):
        differences = (
            poi_coordinates[start : start + 256, np.newaxis, :]
            - e2_coordinates[np.newaxis, :, :]
        )
        squared_distances = np.sum(differences * differences, axis=2)
        minimum_squared = min(minimum_squared, float(np.min(squared_distances)))
        contact_count += int(np.count_nonzero(squared_distances < contact_cutoff**2))
        severe_count += int(np.count_nonzero(squared_distances < severe_cutoff**2))
    return minimum_squared**0.5, contact_count, severe_count


def _residue_by_number(chain, residue_number: int):
    matches = [residue for residue in chain if residue.id[1] == residue_number]
    if len(matches) != 1:
        raise ValueError(
            f"expected one residue {residue_number} in chain {chain.id!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def screen_e2_accessibility(
    input_pdb: str | Path,
    reference_pdb: str | Path,
    *,
    poi_chain: str,
    mobile_e3_chain: str,
    reference_e3_chain: str,
    e2_chain: str,
    e2_residue: int,
    e2_atom: str = "SG",
    lysine_minimum_distance: float = 0.0,
    lysine_maximum_distance: float = 25.0,
    minimum_alignment_residues: int = 30,
    maximum_alignment_rmsd: float = 3.0,
    contact_cutoff: float = 2.0,
    severe_cutoff: float = 1.5,
    minimum_separation: float = 3.0,
    maximum_contacts: int = 5,
) -> E2AccessibilityScreenResult:
    """Place the reference E2 and screen target lysine geometry."""

    input_path = Path(input_pdb)
    reference_path = Path(reference_pdb)
    mobile_model = _parse_pdb(input_path)
    reference_model = _parse_pdb(reference_path)

    poi = _chain(mobile_model, poi_chain, input_path.name)
    mobile_e3 = _chain(mobile_model, mobile_e3_chain, input_path.name)
    reference_e3 = _chain(
        reference_model, reference_e3_chain, reference_path.name
    )
    e2 = _chain(reference_model, e2_chain, reference_path.name)
    rotation, translation, alignment_count, alignment_rmsd = _alignment_transform(
        mobile_e3, reference_e3
    )
    alignment_compatible = (
        alignment_count >= minimum_alignment_residues
        and alignment_rmsd <= maximum_alignment_rmsd
    )

    e2_coordinates = _heavy_atom_coordinates(e2) @ rotation + translation
    poi_coordinates = _heavy_atom_coordinates(poi)
    minimum_distance, contact_count, severe_count = _distance_statistics(
        poi_coordinates, e2_coordinates, contact_cutoff, severe_cutoff
    )
    sterically_compatible = (
        minimum_distance >= minimum_separation
        and contact_count < maximum_contacts
        and severe_count == 0
    )

    active_residue = _residue_by_number(e2, e2_residue)
    if e2_atom not in active_residue:
        raise ValueError(
            f"atom {e2_atom!r} is absent from chain {e2_chain!r} residue {e2_residue}"
        )
    active_site = active_residue[e2_atom].coord @ rotation + translation

    lysines = []
    for residue in poi:
        if residue.resname != "LYS" or "NZ" not in residue:
            continue
        distance = float(np.linalg.norm(residue["NZ"].coord - active_site))
        lysines.append(
            LysineDistance(
                residue_number=residue.id[1],
                insertion_code=residue.id[2].strip(),
                distance_angstrom=round(distance, 3),
                in_range=(
                    lysine_minimum_distance <= distance < lysine_maximum_distance
                ),
            )
        )
    if not lysines:
        raise ValueError(f"chain {poi_chain!r} contains no lysine NZ atoms")
    lysines.sort(key=lambda lysine: lysine.distance_angstrom)
    passed = (
        alignment_compatible
        and sterically_compatible
        and any(lysine.in_range for lysine in lysines)
    )

    return E2AccessibilityScreenResult(
        input_structure=input_path.name,
        reference_structure=reference_path.name,
        poi_chain=poi_chain,
        mobile_e3_chain=mobile_e3_chain,
        reference_e3_chain=reference_e3_chain,
        e2_chain=e2_chain,
        e2_residue=e2_residue,
        e2_atom=e2_atom,
        alignment_residue_count=alignment_count,
        alignment_rmsd_angstrom=alignment_rmsd,
        minimum_alignment_residues=minimum_alignment_residues,
        maximum_alignment_rmsd_angstrom=maximum_alignment_rmsd,
        alignment_compatible=alignment_compatible,
        minimum_poi_e2_distance_angstrom=minimum_distance,
        contacts_below_cutoff=contact_count,
        severe_contacts=severe_count,
        contact_cutoff_angstrom=contact_cutoff,
        severe_cutoff_angstrom=severe_cutoff,
        minimum_separation_angstrom=minimum_separation,
        maximum_contacts=maximum_contacts,
        sterically_compatible=sterically_compatible,
        lysine_minimum_distance_angstrom=lysine_minimum_distance,
        lysine_maximum_distance_angstrom=lysine_maximum_distance,
        lysines=tuple(lysines),
        passed=passed,
    )


def write_e2_accessibility_result(
    result: E2AccessibilityScreenResult, output_path: str | Path
) -> Path:
    """Write one E2 active-site accessibility result as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
