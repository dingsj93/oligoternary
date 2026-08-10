"""
Metrics module for PROTAC ternary complex analysis.
This module provides classes for calculating various metrics such as energy and SASA.
"""

import concurrent.futures
import os
import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple

import pyrosetta
from pyrosetta import Pose
from pyrosetta.rosetta.core.scoring import calc_total_sasa
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector

from loguru import logger
from rich.console import Console
from rich.table import Table
import glob
import pandas as pd
from tqdm import tqdm

from oligoternary.analysis.linker_geometry import compute_linker_geometry_metrics
from oligoternary.modeling.pyrosetta_runtime import (
    build_metrics_init_options,
    ensure_pyrosetta_initialized,
)


# ---------------------------------------------------------------------------
# Chain-specifier helpers
# ---------------------------------------------------------------------------

def _expand_chain_ids(chain_spec: Any) -> List[str]:
    """Normalize a chain specifier into a flat list of single-char chain IDs.

    Accepts:
      - single char: ``"A"`` -> ``["A"]``
      - concatenated multi-char: ``"BC"`` -> ``["B", "C"]`` (multi-chain
        warhead)
      - iterable of (possibly multi-char) strings:
        ``["BC", "X", "L"]`` -> ``["B", "C", "X", "L"]``

    ``ChainSelector`` receives comma-separated single-character IDs, while
    ``InterfaceAnalyzerMover`` receives concatenated IDs on each side of the
    interface (for example ``"AXL_B"``).
    """
    def _is_chain_char(ch: str) -> bool:
        # Skip commas/whitespace so that "B,L,X" and "B L X" are normalized
        # the same way as the concatenated form "BLX".
        return bool(ch) and not ch.isspace() and ch != ","

    if chain_spec is None:
        return []
    if isinstance(chain_spec, str):
        return [c for c in chain_spec if _is_chain_char(c)]
    out: List[str] = []
    for item in chain_spec:
        if item is None:
            continue
        out.extend(c for c in str(item) if _is_chain_char(c))
    return out


def _validate_chain_roles(
    pose: Pose,
    poi_chain: Any,
    warhead_chain: Any,
    e3_chain: Any,
    protac_chains: Any,
) -> Dict[str, List[str]]:
    """Validate an exhaustive, non-overlapping top-level chain-role map.

    ``warhead`` is a named subset of ``protac``.  The three physical groups
    used by IAM and BSA (POI, E3, PROTAC) must otherwise be disjoint and must
    identify every chain present in the pose so all metrics in one output row
    operate on the same physical system.
    """
    if pose is None or pose.total_residue() == 0:
        raise ValueError("Pose is empty or invalid.")
    pdb_info = pose.pdb_info()
    if pdb_info is None:
        raise ValueError("Pose lacks PDBInfo; cannot validate chain roles.")

    roles = {
        "poi": list(dict.fromkeys(_expand_chain_ids(poi_chain))),
        "warhead": list(dict.fromkeys(_expand_chain_ids(warhead_chain))),
        "e3": list(dict.fromkeys(_expand_chain_ids(e3_chain))),
        "protac": list(dict.fromkeys(_expand_chain_ids(protac_chains))),
    }
    empty_roles = [name for name, chains in roles.items() if not chains]
    if empty_roles:
        raise ValueError(f"Chain role(s) cannot be empty: {', '.join(empty_roles)}")

    warhead_not_protac = set(roles["warhead"]) - set(roles["protac"])
    if warhead_not_protac:
        raise ValueError(
            "Warhead chain(s) must be included in PROTAC chains: "
            f"{','.join(sorted(warhead_not_protac))}"
        )

    top_level = {name: set(roles[name]) for name in ("poi", "e3", "protac")}
    overlaps = []
    top_names = list(top_level)
    for index, left in enumerate(top_names):
        for right in top_names[index + 1:]:
            shared = top_level[left] & top_level[right]
            if shared:
                overlaps.append(f"{left}/{right}={','.join(sorted(shared))}")
    if overlaps:
        raise ValueError(f"Chain role overlap is not allowed: {'; '.join(overlaps)}")

    available = list(dict.fromkeys(
        pdb_info.chain(index) for index in range(1, pose.total_residue() + 1)
    ))
    assigned = set().union(*top_level.values())
    missing = assigned - set(available)
    if missing:
        raise ValueError(
            f"Assigned chain(s) missing from pose: {','.join(sorted(missing))}. "
            f"Available chains: {','.join(available)}"
        )
    unassigned = set(available) - assigned
    if unassigned:
        raise ValueError(
            "Pose contains unassigned chain(s), so complex metrics would be "
            f"ambiguous: {','.join(sorted(unassigned))}"
        )
    return roles


def _process_single_pdb_worker(
    pdb_index: int,
    pdb_file: str,
    poi_chain: str,
    warhead_chain: str,
    e3_chain: str,
    protac_chains: List[str],
    score_function_name: str,
    probe_radius: float,
    init_options: Tuple[str, ...],
    linker_geometry_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[str, float]]:
    """Worker-safe single-PDB analysis entrypoint for spawn-based multiprocessing.

    When ``linker_geometry_kwargs`` is set, also compute linker geometry metrics.
    """
    ensure_pyrosetta_initialized(init_options)
    pose = pyrosetta.pose_from_file(pdb_file)
    if pose.total_residue() == 0:
        raise ValueError(f"Empty pose for {pdb_file}")

    analyzer = ComplexAnalyzer(
        score_function_name=score_function_name,
        probe_radius=probe_radius,
    )
    results = analyzer.analyze_complex(
        pose=pose,
        poi_chain=poi_chain,
        warhead_chain=warhead_chain,
        e3_chain=e3_chain,
        protac_chains=protac_chains,
    )
    if linker_geometry_kwargs:
        geometry = compute_linker_geometry_metrics(pose, **linker_geometry_kwargs)
        results.update(geometry)
    results["pdb_file"] = os.path.basename(pdb_file)
    results["pdb_path"] = pdb_file
    results["output_file"] = os.path.basename(pdb_file)
    return pdb_index, results


@dataclass
class EnergyCalculator:
    """Class for calculating energy-related metrics for PROTAC ternary complexes."""
    score_name: str = "ref2015"

    def __post_init__(self):
        self.scorefxn = None

    def _ensure_scorefxn(self):
        """Lazy initialization of score function."""
        if self.scorefxn is None:
            self.scorefxn = pyrosetta.create_score_function(self.score_name)

    def calculate_total_energy(self, pose: Pose) -> float:
        self._ensure_scorefxn()
        self.scorefxn(pose)
        total_energy = pose.scores["total_score"]
        return total_energy

    def calculate_interface_energies(
        self,
        pose: Pose,
        poi_selector_str: str,
        e3_selector_str: str,
        warhead_selector_str: str,
        protac_selector_str: Union[str, List[str]],
    ):
        """Compute total energy and per-interface ΔG values.

        Returns five Rosetta energy fields:
          - ``total_energy``                : full-pose Rosetta total score (REU)
          - ``dG_poi_protac``               : POI vs full PROTAC interface ΔG
          - ``dG_e3_protac``                : E3 vs full PROTAC interface ΔG
          - ``dG_poi_e3``                   : POI vs E3 protein-protein interface ΔG
          - ``dG_poi_complex_e3``           : (POI + warhead) vs E3 interface ΔG

        Chain-spec arguments accept single chars (``"A"``), concatenated
        multi-char (``"BC"`` for a multi-chain warhead) or iterables.
        IAM interface strings use the *concatenated* form (``"AXL_B"``).
        """
        self._ensure_scorefxn()
        total_energy = self.calculate_total_energy(pose)

        poi = "".join(_expand_chain_ids(poi_selector_str))
        e3 = "".join(_expand_chain_ids(e3_selector_str))
        warhead = "".join(_expand_chain_ids(warhead_selector_str))
        protac = "".join(_expand_chain_ids(protac_selector_str))

        # 1. POI-PROTAC interface (QC: warhead engagement of POI)
        iam_poi_protac = InterfaceAnalyzerMover(f"{poi}_{protac}")
        iam_poi_protac.set_scorefunction(self.scorefxn)
        iam_poi_protac.set_pack_separated(False)
        iam_poi_protac.apply(pose)
        dG_poi_protac = iam_poi_protac.get_interface_dG()

        # 2. E3-PROTAC interface (QC: E3-ligand engagement of E3)
        iam_e3_protac = InterfaceAnalyzerMover(f"{e3}_{protac}")
        iam_e3_protac.set_scorefunction(self.scorefxn)
        iam_e3_protac.set_pack_separated(False)
        iam_e3_protac.apply(pose)
        dG_e3_protac = iam_e3_protac.get_interface_dG()

        # 3. POI-E3 protein-protein interface (primary mechanistic signal)
        iam_poi_e3 = InterfaceAnalyzerMover(f"{poi}_{e3}")
        iam_poi_e3.set_scorefunction(self.scorefxn)
        iam_poi_e3.set_pack_separated(False)
        iam_poi_e3.apply(pose)
        dG_poi_e3 = iam_poi_e3.get_interface_dG()

        # 4. (POI + warhead) vs E3 interface (extended ternary interface)
        iam_poi_complex_e3 = InterfaceAnalyzerMover(f"{poi}{warhead}_{e3}")
        iam_poi_complex_e3.set_scorefunction(self.scorefxn)
        iam_poi_complex_e3.set_pack_separated(False)
        iam_poi_complex_e3.apply(pose)
        dG_poi_complex_e3 = iam_poi_complex_e3.get_interface_dG()

        return {
            "total_energy": total_energy,
            "dG_poi_protac": dG_poi_protac,
            "dG_e3_protac": dG_e3_protac,
            "dG_poi_e3": dG_poi_e3,
            "dG_poi_complex_e3": dG_poi_complex_e3,
        }


@dataclass
class SASACalculator:
    """Class for calculating SASA-related metrics for PROTAC ternary complexes."""

    probe_radius: float = 1.4

    def _get_available_chain_ids(self, pose: Pose) -> List[str]:
        pdb_info = pose.pdb_info()
        if pdb_info is None:
            raise ValueError("Pose lacks PDBInfo; cannot resolve chain identifiers.")

        chain_ids: List[str] = []
        seen = set()
        for residue_index in range(1, pose.total_residue() + 1):
            chain_id = pdb_info.chain(residue_index)
            if chain_id not in seen:
                seen.add(chain_id)
                chain_ids.append(chain_id)
        return chain_ids

    def get_pose_from_chains(self, pose: Pose, chain_ids_to_extract: List[str]) -> Pose:
        if pose is None or pose.total_residue() == 0:
            raise ValueError("Pose is empty or invalid.")

        single_char_chains = list(dict.fromkeys(_expand_chain_ids(chain_ids_to_extract)))
        if not single_char_chains:
            raise ValueError("No chain IDs were requested for SASA extraction.")

        available_chains = self._get_available_chain_ids(pose)
        missing_chains = [c for c in single_char_chains if c not in available_chains]
        if missing_chains:
            raise ValueError(
                "Missing requested chain(s) in pose: "
                f"{','.join(missing_chains)}. Available chains: {','.join(available_chains)}"
            )

        chain_string = ",".join(single_char_chains)
        selector = ChainSelector(chain_string)
        selection_vector = selector.apply(pose)

        if not any(selection_vector):
            raise ValueError(
                f"Chain selection '{chain_string}' produced an empty sub-pose unexpectedly."
            )

        # Clone the original pose and delete residues that are NOT part of the selection
        sub_pose = pose.clone()

        # Collect indices of residues to delete (1-based)
        residues_to_delete = []
        for i, is_selected in enumerate(selection_vector):
            if not is_selected:
                residues_to_delete.append(i + 1)

        # Delete residues in reverse order to avoid renumbering issues
        for res_num in sorted(residues_to_delete, reverse=True):
            sub_pose.delete_residue_slow(res_num)

        return sub_pose

    def calculate_protac_bsa(
        self,
        pose: Pose,
        poi_chain: str,
        e3_chain: str,
        protac_chains: List[str],
    ) -> Dict[str, float]:
        sub_pose = lambda chains: self.get_pose_from_chains(pose, chains)
        sasa_total_complex = calc_total_sasa(pose, self.probe_radius)
        sasa_poi = calc_total_sasa(sub_pose([poi_chain]), self.probe_radius)
        sasa_e3 = calc_total_sasa(sub_pose([e3_chain]), self.probe_radius)
        sasa_protac = calc_total_sasa(sub_pose(protac_chains), self.probe_radius)
        sasa_poi_protac = calc_total_sasa(sub_pose([poi_chain] + protac_chains), self.probe_radius)
        sasa_e3_protac = calc_total_sasa(sub_pose([e3_chain] + protac_chains), self.probe_radius)
        sasa_poi_e3 = calc_total_sasa(sub_pose([poi_chain, e3_chain]), self.probe_radius)

        bsa_poi_protac = sasa_poi + sasa_protac - sasa_poi_protac
        bsa_e3_protac = sasa_e3 + sasa_protac - sasa_e3_protac
        bsa_poi_e3 = sasa_poi + sasa_e3 - sasa_poi_e3
        bsa_total = (sasa_poi + sasa_e3 + sasa_protac) - sasa_total_complex

        return {
            "bsa_total_buried_in_complex": bsa_total,
            "bsa_poi_protac_interface": bsa_poi_protac,
            "bsa_e3_protac_interface": bsa_e3_protac,
            "bsa_poi_e3_interface": bsa_poi_e3,
        }


@dataclass
class ComplexAnalyzer:
    """Class for comprehensive analysis of PROTAC ternary complexes."""
    score_function_name: str = "ref2015"
    probe_radius: float = 1.4

    def __post_init__(self):
        self.energy_calculator = EnergyCalculator(score_name=self.score_function_name)
        self.sasa_calculator = SASACalculator(probe_radius=self.probe_radius)

    def analyze_complex(self, pose: Pose, poi_chain: str, warhead_chain: str, e3_chain: str,
                       protac_chains: List[str]):
        roles = _validate_chain_roles(
            pose,
            poi_chain=poi_chain,
            warhead_chain=warhead_chain,
            e3_chain=e3_chain,
            protac_chains=protac_chains,
        )
        # Calculate energy metrics
        energy_results = self.energy_calculator.calculate_interface_energies(
            pose = pose,
            poi_selector_str = roles["poi"],
            e3_selector_str = roles["e3"],
            warhead_selector_str = roles["warhead"],
            protac_selector_str = roles["protac"]
        )

        # Calculate SASA metrics
        sasa_results = self.sasa_calculator.calculate_protac_bsa(
            pose = pose,
            poi_chain = "".join(roles["poi"]),
            e3_chain = "".join(roles["e3"]),
            protac_chains = roles["protac"]
        )

        results = {**energy_results, **sasa_results}
        return results


class BatchAnalyzer:
    """Class for batch analysis of multiple PDB files."""

    def __init__(self, score_function_name: str = "ref2015", probe_radius: float = 1.4):
        self.score_function_name = score_function_name
        self.probe_radius = probe_radius
        self.last_total_count = 0
        self.last_succeeded_count = 0
        self.last_failed_inputs: List[str] = []

    def process_pdb_folder(
        self,
        input_folder: str,
        output_csv: str,
        poi_chain: str,
        warhead_chain: str,
        e3_chain: str,
        protac_chains: List[str],
        params_files: List[str] = None,
        num_processes: int = 1,
        linker_geometry_kwargs: Optional[Dict[str, Any]] = None,
        strict: bool = True,
        random_seed: int = 42,
    ) -> int:
        """
        Process all PDB files in a folder and save results to CSV.

        Returns:
            0 on success, 1 on failure. With ``strict=True``, every input PDB
            must succeed; otherwise at least one successful row is enough.
        """
        if params_files is None:
            params_files = []

        if num_processes < 1:
            raise ValueError(f"num_processes must be >= 1, got {num_processes}")

        init_options = build_metrics_init_options(
            params_files, random_seed=random_seed
        )
        if num_processes == 1:
            ensure_pyrosetta_initialized(init_options)

        # Find all PDB files
        pdb_files = sorted(glob.glob(os.path.join(input_folder, "*.pdb")))
        self.last_total_count = len(pdb_files)
        self.last_succeeded_count = 0
        self.last_failed_inputs = []
        if not pdb_files:
            logger.error(f"No PDB files found in {input_folder}")
            return 1

        logger.info(f"Found {len(pdb_files)} PDB files to process")

        indexed_results: List[Tuple[int, Dict[str, float]]] = []

        if num_processes == 1:
            for pdb_index, pdb_file in enumerate(tqdm(pdb_files, desc="Processing PDB files", unit="file")):
                try:
                    indexed_results.append(
                        _process_single_pdb_worker(
                            pdb_index=pdb_index,
                            pdb_file=pdb_file,
                            poi_chain=poi_chain,
                            warhead_chain=warhead_chain,
                            e3_chain=e3_chain,
                            protac_chains=protac_chains,
                            score_function_name=self.score_function_name,
                            probe_radius=self.probe_radius,
                            init_options=init_options,
                            linker_geometry_kwargs=linker_geometry_kwargs,
                        )
                    )
                    logger.info(f"Successfully processed: {os.path.basename(pdb_file)}")
                except Exception as e:  # per-file worker boundary
                    logger.error(f"Error processing {os.path.basename(pdb_file)}: {str(e)}")
                    self.last_failed_inputs.append(os.path.basename(pdb_file))
        else:
            logger.info(f"Using spawn-based multiprocessing with {num_processes} workers")
            spawn_context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_processes,
                mp_context=spawn_context,
            ) as executor:
                future_map = {
                    executor.submit(
                        _process_single_pdb_worker,
                        pdb_index,
                        pdb_file,
                        poi_chain,
                        warhead_chain,
                        e3_chain,
                        protac_chains,
                        self.score_function_name,
                        self.probe_radius,
                        init_options,
                        linker_geometry_kwargs,
                    ): (pdb_index, pdb_file)
                    for pdb_index, pdb_file in enumerate(pdb_files)
                }

                with tqdm(total=len(pdb_files), desc="Processing PDB files", unit="file") as pbar:
                    for future in concurrent.futures.as_completed(future_map):
                        _pdb_index, pdb_file = future_map[future]
                        try:
                            indexed_results.append(future.result())
                            logger.info(f"Successfully processed: {os.path.basename(pdb_file)}")
                        except Exception as e:  # per-file worker boundary
                            logger.error(f"Error processing {os.path.basename(pdb_file)}: {str(e)}")
                            self.last_failed_inputs.append(os.path.basename(pdb_file))
                        pbar.update(1)

        successful_indices = {index for index, _ in indexed_results}
        self.last_succeeded_count = len(successful_indices)
        self.last_failed_inputs = [
            os.path.basename(pdb_file)
            for index, pdb_file in enumerate(pdb_files)
            if index not in successful_indices
        ]

        # Save results to CSV
        if indexed_results:
            indexed_results.sort(key=lambda item: item[0])
            results_list = [result for _, result in indexed_results]
            df = pd.DataFrame(results_list)
            # Reorder columns to put file info first
            cols = ['pdb_file', 'pdb_path'] + [col for col in df.columns if col not in ['pdb_file', 'pdb_path']]
            df = df[cols]

            os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
            df.to_csv(output_csv, index=False)
            logger.info(f"Results saved to {output_csv}")
            logger.info(f"Successfully processed {len(results_list)} out of {len(pdb_files)} PDB files")

            # Display summary statistics
            self._display_summary(df)
            if strict and len(results_list) < len(pdb_files):
                logger.error(
                    f"Strict mode: only {len(results_list)}/{len(pdb_files)} PDB files succeeded"
                )
                return 1
            return 0
        else:
            logger.error("No results to save. All PDB files failed to process.")
            return 1

    def _display_summary(self, df: pd.DataFrame):
        """Display summary statistics of the results."""
        console = Console()

        # Create summary table
        table = Table(title="Batch Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="green")
        table.add_column("Std", style="yellow")
        table.add_column("Min", style="red")
        table.add_column("Max", style="magenta")

        # Select numeric columns for summary
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        for col in numeric_cols:
            if col not in ['pdb_file', 'pdb_path']:
                stats = df[col].describe()
                table.add_row(
                    col,
                    f"{stats['mean']:.4f}",
                    f"{stats['std']:.4f}",
                    f"{stats['min']:.4f}",
                    f"{stats['max']:.4f}"
                )

        console.print(table)
