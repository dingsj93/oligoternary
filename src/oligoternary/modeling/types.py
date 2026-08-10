"""Dataclasses and PDB label parsing."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oligoternary.modeling.constants import ATOM_LABEL_PATTERN


def parse_pdb_atom_label(label: str) -> Tuple[str, int, str, str]:
    """Parse ``chain:resnum[:icode]:atomname`` atom labels."""
    match = ATOM_LABEL_PATTERN.match(label)
    if match is None:
        raise ValueError(
            f"Invalid atom label format: {label}. Expected chain:resnum[:icode]:atomname"
        )
    chain_id = match.group("chain")
    residue_number = int(match.group("resnum"))
    insertion_code = match.group("icode") or " "
    atom_name = match.group("atom")
    return chain_id, residue_number, insertion_code, atom_name


@dataclass(frozen=True)
class PreparedComplex:
    base_name: str
    filename: str
    initial_complex: str
    input_e3l_anchor_label: str
    rosetta_e3l_anchor_label: str


@dataclass(frozen=True)
class HeavyAtomRecord:
    label: str
    element: str
    coord: np.ndarray


@dataclass(frozen=True)
class PrepareBatchJob:
    pdb_file: str
    params_dir: str
    tmp_dir: str
    linker_min: float
    linker_max: float
    minimize_steps: int
    random_seed: int


@dataclass(frozen=True)
class RosettaBatchJob:
    prepared_item: PreparedComplex
    params_dir: str
    model_dir: str
    relax_cycles: int
    minimize_steps: int
    minimize_tolerance: float
    random_seed: int
    collect_metrics: bool


@dataclass(frozen=True)
class PrepareWorkerContext:
    linker_prefix: str
    linker_prepared_smiles: str
    linker_to_warhead_index: int
    linker_to_e3l_index: int
    linker_to_warhead_atom: str
    linker_to_e3l_atom: str
    linker_chain: str
    e3l_chain: str
    e3l_connect_atom: str
    warhead_anchor_label: str
    e3l_anchor_label: str
    endpoint_candidates: int
    conformers_per_endpoint: int
    minimum_attachment_angle_degrees: float
    conformer_ranking: str


@dataclass(frozen=True)
class OptimizeWorkerContext:
    linker_prefix: str
    linker_chain: str
    warhead_anchor_label: str
    linker_to_warhead_atom: str
    linker_to_e3l_atom: str
    poi_chain: str
    warhead_chain: str
    e3_chain: str
    e3l_chain: str
    minimum_attachment_angle_degrees: float


@dataclass(frozen=True)
class BatchFailure:
    """One failed input in a batch linker-refiner run."""

    input_path: str
    stage: str
    message: str


@dataclass(frozen=True)
class BatchRunSummary:
    """Batch linker-refiner run counts and output path."""

    metrics_csv: Optional[str]
    total_inputs: int
    prepared_count: int
    succeeded_count: int
    failed_inputs_tsv: Optional[str] = None
    random_seed: int = 42

    def exit_ok(self, *, strict: bool = True) -> bool:
        if self.succeeded_count < 1 or self.metrics_csv is None:
            return False
        if strict and self.succeeded_count != self.total_inputs:
            return False
        return True


def sort_batch_results_by_input_order(
    results: List[Tuple[str, Dict[str, Any]]],
    input_pdb_files: List[str],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Order batch metrics rows to match the expanded input PDB list."""
    import os

    order = {os.path.basename(path): index for index, path in enumerate(input_pdb_files)}
    fallback = len(input_pdb_files)
    return sorted(
        results,
        key=lambda item: order.get(item[1].get("filename", ""), fallback),
    )


def write_failed_inputs_tsv(failures: List[BatchFailure], path: str) -> Optional[str]:
    """Write batch failure records as TSV; return path when written."""
    if not failures:
        return None
    import os

    import pandas as pd

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(
        {
            "input": [os.path.basename(item.input_path) for item in failures],
            "stage": [item.stage for item in failures],
            "message": [item.message for item in failures],
        }
    )
    df.to_csv(path, sep="\t", index=False)
    return path
