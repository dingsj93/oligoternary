"""Align residue atom names to the Rosetta params reference topology."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from rdkit import Chem


def _load_residue_lines(
    pdb_file: Path,
    chain_id: str,
    resname: Optional[str] = None,
    residue_number: Optional[int] = None,
    insertion_code: Optional[str] = None,
) -> tuple[List[str], List[str]]:
    all_lines = pdb_file.read_text().splitlines(True)
    residue_lines = []
    for line in all_lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) <= 26 or line[21] != chain_id:
            continue
        if resname is not None and line[17:20].strip() != resname:
            continue
        if residue_number is not None:
            try:
                line_residue_number = int(line[22:26].strip())
            except ValueError:
                continue
            if line_residue_number != residue_number:
                continue
        if insertion_code is not None and line[26] != insertion_code:
            continue
        residue_lines.append(line)
    if not residue_lines:
        selector = [f"chain={chain_id}"]
        if resname is not None:
            selector.append(f"resname={resname!r}")
        if residue_number is not None:
            selector.append(f"residue_number={residue_number}")
        if insertion_code is not None:
            selector.append(f"insertion_code={insertion_code!r}")
        raise ValueError(f"No residue found in {pdb_file} matching {', '.join(selector)}")
    return all_lines, residue_lines


def _mol_from_residue_lines(residue_lines: List[str], label: str) -> Chem.Mol:
    pdb_block = "".join(residue_lines) + "END\n"
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True, sanitize=True)
    if mol is None:
        raise RuntimeError(f"RDKit failed to parse {label} residue block")
    return mol


def _mol_from_params_pdb(params_pdb_file: Path) -> Chem.Mol:
    mol = Chem.MolFromPDBFile(str(params_pdb_file), removeHs=True, sanitize=True)
    if mol is None:
        raise RuntimeError(f"RDKit failed to parse params PDB: {params_pdb_file}")
    return mol


def _atom_name_map(mol: Chem.Mol, label: str) -> Dict[int, str]:
    name_map: Dict[int, str] = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            raise RuntimeError(f"{label} atom idx {atom.GetIdx()} is missing PDB residue info")
        name_map[atom.GetIdx()] = info.GetName().strip()

    if len(name_map) != mol.GetNumAtoms():
        raise RuntimeError(
            f"{label} atom-name map incomplete: {len(name_map)} / {mol.GetNumAtoms()}"
        )
    if len(set(name_map.values())) != len(name_map):
        raise RuntimeError(f"{label} contains duplicate atom names: {name_map}")
    return name_map


def _build_rename_map(input_mol: Chem.Mol, params_mol: Chem.Mol) -> Dict[str, str]:
    if input_mol.GetNumAtoms() != params_mol.GetNumAtoms():
        raise RuntimeError(
            "Residue atom count does not match params reference: "
            f"{input_mol.GetNumAtoms()} vs {params_mol.GetNumAtoms()}"
        )

    match = input_mol.GetSubstructMatch(params_mol)
    if len(match) != params_mol.GetNumAtoms():
        raise RuntimeError(
            "Input residue graph is not isomorphic to params reference. "
            f"input_atoms={input_mol.GetNumAtoms()} input_bonds={input_mol.GetNumBonds()} "
            f"params_atoms={params_mol.GetNumAtoms()} params_bonds={params_mol.GetNumBonds()}"
        )

    reverse_match = params_mol.GetSubstructMatch(input_mol)
    if len(reverse_match) != input_mol.GetNumAtoms():
        raise RuntimeError("Params reference does not fully match the input residue graph")

    input_names = _atom_name_map(input_mol, "input")
    params_names = _atom_name_map(params_mol, "params")

    rename: Dict[str, str] = {}
    for params_idx, input_idx in enumerate(match):
        old_name = input_names[input_idx]
        new_name = params_names[params_idx]
        if old_name == new_name:
            continue
        existing = rename.get(old_name)
        if existing is not None and existing != new_name:
            raise RuntimeError(f"Conflicting rename for atom {old_name}: {existing} vs {new_name}")
        rename[old_name] = new_name
    return rename


def build_residue_atom_rename_map(
    pdb_file: str,
    chain_id: str,
    params_pdb_file: str,
    resname: Optional[str] = None,
    residue_number: Optional[int] = None,
    insertion_code: Optional[str] = None,
    residue_lines: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Return the input-name -> params-name mapping without rewriting the PDB."""
    pdb_path = Path(pdb_file)
    params_path = Path(params_pdb_file)

    if not params_path.is_file():
        raise FileNotFoundError(f"Params PDB not found: {params_path}")

    if residue_lines is None:
        _all_lines, residue_lines = _load_residue_lines(
            pdb_path,
            chain_id,
            resname=resname,
            residue_number=residue_number,
            insertion_code=insertion_code,
        )
    input_mol = _mol_from_residue_lines(residue_lines, f"residue in {pdb_path}")
    params_mol = _mol_from_params_pdb(params_path)
    return _build_rename_map(input_mol, params_mol)


def _format_atom_name(atom_name: str) -> str:
    return f" {atom_name:<3s}" if len(atom_name) <= 3 else f"{atom_name:<4s}"


def fix_residue_atom_names(
    pdb_file: str,
    chain_id: str,
    resname: str,
    params_pdb_file: str,
) -> Dict[str, str]:
    """
    Rewrite residue atom names in-place so they exactly match the Rosetta params PDB.

    The params-generated PDB is the only naming source of truth because Rosetta maps
    coordinates to non-canonical residues by atom name.
    """
    pdb_path = Path(pdb_file)
    params_path = Path(params_pdb_file)

    if not params_path.is_file():
        raise FileNotFoundError(f"Params PDB not found: {params_path}")

    all_lines, residue_lines = _load_residue_lines(pdb_path, chain_id, resname=resname)
    rename = build_residue_atom_rename_map(
        pdb_file=str(pdb_path),
        chain_id=chain_id,
        resname=resname,
        params_pdb_file=str(params_path),
        residue_lines=residue_lines,
    )

    if not rename:
        return rename

    rewritten_lines: List[str] = []
    for line in all_lines:
        if (
            line.startswith(("ATOM", "HETATM"))
            and len(line) > 21
            and line[21] == chain_id
            and line[17:20].strip() == resname
        ):
            atom_name = line[12:16].strip()
            new_name = rename.get(atom_name)
            if new_name is not None:
                line = line[:12] + _format_atom_name(new_name) + line[16:]
        rewritten_lines.append(line)

    pdb_path.write_text("".join(rewritten_lines))
    return rename


def remap_atom_label(atom_label: str, rename_map: Dict[str, str]) -> str:
    """Rewrite the atom-name part of a `chain:resnum:atom` label via `rename_map`."""
    chain_id, resnum, atom_name = atom_label.split(":")
    return f"{chain_id}:{resnum}:{rename_map.get(atom_name, atom_name)}"
