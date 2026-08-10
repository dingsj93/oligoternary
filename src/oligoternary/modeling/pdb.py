"""PDB coordinate extraction and merge utilities."""
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure

from oligoternary.modeling.constants import E3L_RESNAME, LINKER_RESNAME
from oligoternary.modeling.geometry import normalize_element_symbol
from oligoternary.modeling.types import HeavyAtomRecord, parse_pdb_atom_label

@dataclass
class PDBAtomSelector:
    pdb_file: str
    parser: PDBParser = field(init=False)
    structure: Structure = field(init=False)

    def __post_init__(self):
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure('molecule', self.pdb_file)

    def _get_atom_coordinates(self, atom_string: str):
        chain_id, res_num, insertion_code, atom_name = parse_pdb_atom_label(atom_string)
        for model in self.structure:
            for chain in model:
                if chain.id != chain_id:
                    continue
                for residue in chain:
                    if residue.id[1] == res_num and residue.id[2] == insertion_code:
                        if atom_name in residue:
                            return residue[atom_name].get_coord()
                        break
        raise ValueError(f"Atom not found in PDB: {atom_string}")

    def get_heavy_atoms(self) -> List[HeavyAtomRecord]:
        """Return all heavy-atom records in the PDB."""
        atoms = []
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    residue_number = residue.id[1]
                    insertion_code = residue.id[2]
                    for atom in residue:
                        element = normalize_element_symbol(atom.element, atom.name)
                        if element == "H":
                            continue
                        atoms.append(
                            HeavyAtomRecord(
                                label=(
                                    f"{chain.id}:{residue_number}"
                                    f"{insertion_code.strip() if insertion_code.strip() else ''}:{atom.name}"
                                ),
                                element=element,
                                coord=np.asarray(atom.get_coord(), dtype=float),
                            )
                        )
        if not atoms:
            raise ValueError(f"No heavy atoms found in {self.pdb_file}")
        return atoms

    def get_warhead_e3l_linked_coordinates(
        self, warhead_anchor_label: str, e3l_anchor_label: str
    ) -> Tuple[List[float], List[float], float]:
        """Return warhead/E3L anchor coordinates and their distance."""
        warhead_coord = self._get_atom_coordinates(warhead_anchor_label)
        e3l_coord = self._get_atom_coordinates(e3l_anchor_label)
        distance = np.linalg.norm(warhead_coord - e3l_coord)
        return warhead_coord.tolist(), e3l_coord.tolist(), float(distance)


def _merged_atom_records(
    pdb_file: str,
    *,
    first_serial: int,
    chain_id: str | None = None,
    residue_name: str | None = None,
    residue_number: int | None = None,
    rename_chain: str | None = None,
) -> tuple[list[str], list[str], int]:
    records = []
    conect_lines = []
    serial_map = {}
    serial = first_serial
    with open(pdb_file, encoding="utf-8") as handle:
        lines = handle.readlines()
    for line in lines:
        if line.startswith("CONECT"):
            conect_lines.append(line)
            continue
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = normalize_element_symbol(line[76:78], line[12:16])
        if element == "H":
            continue
        old_serial = int(line[6:11])
        serial_map[old_serial] = serial
        fields = list(line.rstrip("\n").ljust(80))
        fields[6:11] = f"{serial:5d}"
        if chain_id is not None:
            fields[21] = chain_id
        if residue_name is not None and (
            rename_chain is None or fields[21] == rename_chain
        ):
            fields[17:20] = f"{residue_name:>3}"
        if residue_number is not None:
            fields[22:26] = f"{residue_number:4d}"
        records.append("".join(fields) + "\n")
        serial += 1

    remapped_conect = []
    for line in conect_lines:
        old_serials = [int(value) for value in line[6:].split()]
        if not old_serials or old_serials[0] not in serial_map:
            continue
        mapped = [serial_map[value] for value in old_serials if value in serial_map]
        if len(mapped) > 1:
            remapped_conect.append("CONECT" + "".join(f"{value:5d}" for value in mapped) + "\n")
    return records, remapped_conect, serial


def merge_pdbs(
    complex_pdb: str,
    linker_pdb: str,
    linker_chain: str,
    e3l_chain: str,
    filename: str,
) -> str:
    """Merge a hydrogen-free complex and linker into one numbered PDB."""
    if not (os.path.isfile(complex_pdb) and os.path.isfile(linker_pdb)):
        raise FileNotFoundError(f"Input PDB files not found: {complex_pdb}, {linker_pdb}")
    complex_records, complex_conect, next_serial = _merged_atom_records(
        complex_pdb,
        first_serial=1,
        residue_name=E3L_RESNAME,
        rename_chain=e3l_chain,
    )
    linker_records, linker_conect, _ = _merged_atom_records(
        linker_pdb,
        first_serial=next_serial,
        chain_id=linker_chain,
        residue_name=LINKER_RESNAME,
        residue_number=1,
    )
    with open(filename, "w", encoding="utf-8") as handle:
        handle.writelines(complex_records)
        handle.writelines(linker_records)
        handle.writelines(complex_conect)
        handle.writelines(linker_conect)
        handle.write("END\n")
    return filename
