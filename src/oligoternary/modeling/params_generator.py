"""
Rosetta params generation helpers.

This module is the canonical implementation for converting a marked SMILES
containing `[*]` connection markers into Rosetta `.params`/`.pdb` files.

On the Rosetta build used in this repo, `molfile_to_params.py` preserves the
radical valence state written in the input molfile, but does not emit CONNECT
records automatically. We therefore patch CONNECT/CONN records deterministically
after generation and validate that the connect atoms keep the expected hydrogen
counts from the radical source molecule.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem


_EMBED_SEEDS = (42, 123, 987, 555, 789)


@dataclass(frozen=True)
class PreparedMolSpec:
    mol: Chem.Mol
    prepared_smiles: str
    connect_atom_indices: Tuple[int, ...]


@dataclass(frozen=True)
class ParamsResult:
    params_path: str
    pdb_path: str
    mol_path: str
    connect_atoms: Tuple[str, ...]
    connect_atom_indices: Tuple[int, ...]
    prepared_smiles: str


def _index_map_after_atom_removals(
    original_atom_count: int,
    removed_atom_indices: Sequence[int],
) -> Dict[int, int]:
    removed = sorted(removed_atom_indices)
    mapping: Dict[int, int] = {}
    for old_index in range(original_atom_count):
        if old_index in removed:
            continue
        shift = sum(1 for removed_index in removed if removed_index < old_index)
        mapping[old_index] = old_index - shift
    return mapping


def prepare_marked_molecule(
    smiles: str,
) -> PreparedMolSpec:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if not dummy_indices:
        raise ValueError(
            "SMILES must contain one or two `[*]` connect markers. "
            f"Got: {smiles}"
        )

    if len(dummy_indices) > 2:
        raise ValueError(
            "At most two `[*]` connect markers are supported. "
            f"Got {len(dummy_indices)} in: {smiles}"
        )

    neighbor_old_indices: List[int] = []
    for dummy_index in dummy_indices:
        dummy_atom = mol.GetAtomWithIdx(dummy_index)
        neighbors = list(dummy_atom.GetNeighbors())
        if len(neighbors) != 1:
            raise ValueError(
                "Each `[*]` connect marker must have exactly one neighbor. "
                f"Dummy atom {dummy_index} has {len(neighbors)} neighbors in: {smiles}"
            )
        neighbor = neighbors[0]
        if neighbor.GetAtomicNum() == 0:
            raise ValueError(
                "A `[*]` connect marker cannot be attached to another dummy atom. "
                f"Got `[*][*]` around atom indices {dummy_index}-{neighbor.GetIdx()} in: {smiles}"
            )
        neighbor_old_indices.append(neighbor.GetIdx())

    if len(set(neighbor_old_indices)) != len(neighbor_old_indices):
        raise ValueError(
            "Different `[*]` markers map to the same connect atom. "
            f"Neighbor atom indices: {neighbor_old_indices} in: {smiles}"
        )

    rw_mol = Chem.RWMol(Chem.Mol(mol))
    for dummy_index in sorted(dummy_indices, reverse=True):
        rw_mol.RemoveAtom(dummy_index)
    prepared = rw_mol.GetMol()
    index_map = _index_map_after_atom_removals(mol.GetNumAtoms(), dummy_indices)
    connect_atom_indices = tuple(index_map[index] for index in neighbor_old_indices)

    for connect_atom_index in connect_atom_indices:
        prepared.GetAtomWithIdx(connect_atom_index).SetNumRadicalElectrons(1)

    Chem.SanitizeMol(prepared)
    prepared_smiles = Chem.MolToSmiles(prepared, isomericSmiles=True, canonical=False)
    return PreparedMolSpec(
        mol=prepared,
        prepared_smiles=prepared_smiles,
        connect_atom_indices=connect_atom_indices,
    )


def _generated_pdb_atom_name_map(pdb_path: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            serial = int(line[6:11]) - 1
            atom_name = line[12:16].strip()
            mapping[serial] = atom_name
    if not mapping:
        raise ValueError(f"No ATOM/HETATM records found in generated PDB: {pdb_path}")
    return mapping


def _params_atom_names(params_path: str) -> List[str]:
    atom_names: List[str] = []
    with open(params_path) as handle:
        for line in handle:
            if line.startswith("ATOM "):
                atom_names.append(line.split()[1])
    if not atom_names:
        raise ValueError(f"No ATOM records found in params file: {params_path}")
    return atom_names


def _params_hydrogen_neighbor_count(params_path: str, atom_name: str) -> int:
    count = 0
    with open(params_path) as handle:
        for line in handle:
            if not line.startswith("BOND_TYPE"):
                continue
            fields = line.split()
            if len(fields) < 4:
                continue
            left_atom, right_atom = fields[1], fields[2]
            if left_atom == atom_name and right_atom.startswith("H"):
                count += 1
            elif right_atom == atom_name and left_atom.startswith("H"):
                count += 1
    return count


def _mol_hydrogen_neighbor_count(mol: Chem.Mol, atom_index: int) -> int:
    with_hydrogens = Chem.AddHs(Chem.Mol(mol))
    atom = with_hydrogens.GetAtomWithIdx(atom_index)
    return sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1)


def _patch_params_add_connect(params_path: str, connect_atoms: Sequence[str]) -> None:
    requested_atoms = tuple(atom.strip() for atom in connect_atoms if atom and atom.strip())
    if not requested_atoms:
        return
    if len(set(requested_atoms)) != len(requested_atoms):
        raise ValueError(
            f"Duplicate CONNECT atoms requested for {params_path}: {requested_atoms}"
        )

    with open(params_path) as handle:
        lines = handle.readlines()

    atom_names = [
        line.split()[1]
        for line in lines
        if line.startswith("ATOM ")
    ]
    missing_atoms = [atom for atom in requested_atoms if atom not in atom_names]
    if missing_atoms:
        raise ValueError(
            f"Cannot add CONNECT record(s) {missing_atoms} to {params_path}; "
            f"available params atoms: {atom_names}"
        )

    existing_connects = [
        line.split()[1]
        for line in lines
        if line.startswith("CONNECT ")
    ]
    existing_conn_icoors = [
        line.split()[1]
        for line in lines
        if line.startswith("ICOOR_INTERNAL")
        and len(line.split()) > 1
        and line.split()[1].startswith("CONN")
    ]
    if existing_connects or existing_conn_icoors:
        expected_conn_names = [f"CONN{i}" for i in range(1, len(requested_atoms) + 1)]
        if existing_connects != list(requested_atoms) or existing_conn_icoors != expected_conn_names:
            raise ValueError(
                f"{params_path} already defines CONNECT/CONN records "
                f"(CONNECT={existing_connects}, CONN={existing_conn_icoors}), "
                f"expected CONNECT={list(requested_atoms)}, CONN={expected_conn_names}"
            )
        return

    insert_index = next(
        (index for index, line in enumerate(lines) if line.startswith("NBR_ATOM")),
        None,
    )
    if insert_index is None:
        raise ValueError(f"No NBR_ATOM record found in params file: {params_path}")

    icoor_map: Dict[str, List[str]] = {}
    for line in lines:
        if not line.startswith("ICOOR_INTERNAL"):
            continue
        fields = line.split()
        if len(fields) < 8:
            continue
        icoor_map[fields[1]] = fields[2:8]

    last_icoor_index = max(
        index for index, line in enumerate(lines)
        if line.startswith("ICOOR_INTERNAL")
    ) + 1

    connect_lines: List[str] = []
    conn_icoor_lines: List[str] = []
    for connect_index, atom_name in enumerate(requested_atoms, start=1):
        if atom_name not in icoor_map:
            raise ValueError(
                f"Cannot derive CONN{connect_index} ICOOR for atom '{atom_name}' in {params_path}; "
                "missing ICOOR_INTERNAL record."
            )
        phi, theta, distance, stub1, stub2, _stub3 = icoor_map[atom_name]
        connect_lines.append(f"CONNECT {atom_name:<4s} CAN_ROTATE #CONN{connect_index}\n")
        conn_icoor_lines.append(
            "ICOOR_INTERNAL  "
            f"CONN{connect_index:<1d} "
            f"{float(phi):11.6f} {float(theta):11.6f} {float(distance):11.6f}  "
            f"{atom_name:<4s}  {stub1:<4s}  {stub2:<4s}\n"
        )

    updated_lines = (
        lines[:insert_index]
        + connect_lines
        + lines[insert_index:last_icoor_index]
        + conn_icoor_lines
        + lines[last_icoor_index:]
    )
    with open(params_path, "w") as handle:
        handle.writelines(updated_lines)


@dataclass
class ParamsGenerator:
    """Generate Rosetta params from marked or plain SMILES strings."""

    mol_to_params_path: str
    output_dir: str = field(default_factory=os.getcwd)
    clobber: bool = True

    def __post_init__(self) -> None:
        if not self.mol_to_params_path:
            raise ValueError("mol_to_params_path is required")
        os.makedirs(self.output_dir, exist_ok=True)

    def _embed_molecule(self, mol: Chem.Mol, mol_name: str) -> Chem.Mol:
        mol_with_h = Chem.AddHs(Chem.Mol(mol))
        mol_with_h.SetProp("_Name", mol_name)

        params = AllChem.ETKDGv3()
        params.randomSeed = _EMBED_SEEDS[0]
        embed_status = AllChem.EmbedMolecule(mol_with_h, params)
        if embed_status != 0:
            for seed in _EMBED_SEEDS[1:]:
                retry_params = AllChem.ETKDGv3()
                retry_params.randomSeed = seed
                if AllChem.EmbedMolecule(mol_with_h, retry_params) == 0:
                    embed_status = 0
                    break
        if embed_status != 0:
            raise ValueError(f"Failed to generate 3D coordinates for: {mol_name}")

        try:
            AllChem.MMFFOptimizeMolecule(mol_with_h)
        except (ValueError, RuntimeError):
            # MMFF may fail on radicals; ETKDG coords are enough for molfile_to_params.
            pass
        return mol_with_h

    def _write_mol_file(self, mol: Chem.Mol, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Chem.MolToMolFile(mol, output_path)
        return output_path

    def _run_molfile_to_params(
        self,
        mol_file: str,
        resname: str,
        output_dir: str,
    ) -> Tuple[str, str]:
        if not os.path.exists(mol_file):
            raise FileNotFoundError(f"MOL file not found: {mol_file}")

        params_base = os.path.join(output_dir, resname)
        command = [
            sys.executable,
            self.mol_to_params_path,
            mol_file,
            "-p",
            params_base,
            "-n",
            resname,
        ]
        if self.clobber:
            command.append("--clobber")

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "molfile_to_params.py failed.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        params_path = f"{params_base}.params"
        pdb_path = f"{params_base}_0001.pdb"
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Params file not generated: {params_path}")
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"Reference PDB not generated: {pdb_path}")
        return params_path, pdb_path

    def _validate_generated_params(
        self,
        params_path: str,
        prepared_spec: PreparedMolSpec,
        connect_atoms: Sequence[str],
    ) -> None:
        params_atom_names = _params_atom_names(params_path)
        params_heavy_atom_count = sum(1 for atom_name in params_atom_names if not atom_name.startswith("H"))
        expected_heavy_atom_count = sum(
            1 for atom in prepared_spec.mol.GetAtoms() if atom.GetAtomicNum() > 1
        )
        if params_heavy_atom_count != expected_heavy_atom_count:
            raise ValueError(
                f"Heavy atom count mismatch for {params_path}: "
                f"expected {expected_heavy_atom_count}, got {params_heavy_atom_count}"
            )

        existing_connects = [
            line.split()[1]
            for line in Path(params_path).read_text().splitlines()
            if line.startswith("CONNECT ")
        ]
        if tuple(existing_connects) != tuple(connect_atoms):
            raise ValueError(
                f"CONNECT mismatch for {params_path}: "
                f"expected {tuple(connect_atoms)}, got {tuple(existing_connects)}"
            )

        for atom_index, atom_name in zip(prepared_spec.connect_atom_indices, connect_atoms):
            expected_hydrogens = _mol_hydrogen_neighbor_count(prepared_spec.mol, atom_index)
            actual_hydrogens = _params_hydrogen_neighbor_count(params_path, atom_name)
            if actual_hydrogens != expected_hydrogens:
                raise ValueError(
                    f"Hydrogen count mismatch at connect atom {atom_name} in {params_path}: "
                    f"expected {expected_hydrogens}, got {actual_hydrogens}"
                )

    def generate_params(
        self,
        smiles: str,
        resname: str,
        out_dir: Optional[str] = None,
    ) -> ParamsResult:
        output_dir = out_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        prepared_spec = prepare_marked_molecule(smiles)
        embedded_mol = self._embed_molecule(prepared_spec.mol, mol_name=resname)
        mol_path = os.path.join(output_dir, f"{resname}.mol")
        self._write_mol_file(embedded_mol, mol_path)

        params_path, pdb_path = self._run_molfile_to_params(
            mol_file=mol_path,
            resname=resname,
            output_dir=output_dir,
        )

        generated_names = _generated_pdb_atom_name_map(pdb_path)
        connect_atom_names = tuple(
            generated_names[connect_atom_index]
            for connect_atom_index in prepared_spec.connect_atom_indices
        )

        _patch_params_add_connect(params_path, connect_atom_names)
        self._validate_generated_params(
            params_path=params_path,
            prepared_spec=prepared_spec,
            connect_atoms=connect_atom_names,
        )
        return ParamsResult(
            params_path=params_path,
            pdb_path=pdb_path,
            mol_path=mol_path,
            connect_atoms=connect_atom_names,
            connect_atom_indices=prepared_spec.connect_atom_indices,
            prepared_smiles=prepared_spec.prepared_smiles,
        )
