"""Rosetta params file parsing helpers."""
from typing import List, Tuple

def parse_heavy_bonds_from_params(params_file: str) -> List[Tuple[str, str, int]]:
    """Extract heavy-atom BOND_TYPE records from a params file."""
    bonds = []
    with open(params_file) as handle:
        for line in handle:
            if not line.startswith("BOND_TYPE"):
                continue
            fields = line.split()
            atom1, atom2, order = fields[1], fields[2], int(fields[3])
            if atom1.startswith("H") or atom2.startswith("H"):
                continue
            bonds.append((atom1, atom2, order))
    return bonds


def ideal_bond_length_from_atom_names(atom1: str, atom2: str, order: int) -> float:
    """Ideal bond length (A) from atom names and bond order."""
    elem1 = atom1[0]
    elem2 = atom2[0]
    single_bond_lengths = {
        ("C", "C"): 1.54,
        ("C", "N"): 1.47,
        ("C", "O"): 1.43,
        ("C", "S"): 1.82,
        ("C", "P"): 1.85,
        ("N", "N"): 1.45,
        ("N", "O"): 1.40,
        ("N", "S"): 1.75,
        ("O", "P"): 1.61,
        ("P", "O"): 1.61,
        ("P", "S"): 2.12,
        ("S", "S"): 2.05,
    }
    pair = (elem1, elem2)
    if pair not in single_bond_lengths:
        pair = (elem2, elem1)
    base_length = single_bond_lengths.get(pair, 1.50)
    if order == 2:
        base_length -= 0.13
    return base_length
