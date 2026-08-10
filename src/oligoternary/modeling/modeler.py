"""Batch orchestration for linker refinement.

Heavy scientific implementations are imported at the point of use.  Importing
this module therefore remains safe for CLI discovery on machines without a
licensed PyRosetta installation.
"""
import multiprocessing
import os
import re
from dataclasses import dataclass, field
from glob import glob
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from oligoternary.modeling.constants import (
    ATTACHMENT_ENDPOINT_CANDIDATES,
    CONFORMER_RANKING,
    DEFAULT_RANDOM_SEED,
    E3L_RESNAME,
    LINKER_RESNAME,
    MIN_ATTACHMENT_ANGLE_DEGREES,
    PREPARE_CONFORMER_CANDIDATES,
)
from oligoternary.modeling.types import (
    BatchFailure,
    BatchRunSummary,
    OptimizeWorkerContext,
    PreparedComplex,
    PrepareBatchJob,
    PrepareWorkerContext,
    RosettaBatchJob,
    parse_pdb_atom_label,
    sort_batch_results_by_input_order,
    write_failed_inputs_tsv,
)

if TYPE_CHECKING:
    from oligoternary.modeling.params_generator import ParamsResult
else:
    ParamsResult = object


def prepare_batch_worker(
    ctx: PrepareWorkerContext,
    job: PrepareBatchJob,
) -> PreparedComplex:
    """Build and prepare one constrained linker complex in a worker process."""
    from oligoternary.modeling.stages import (
        build_conformer_generator,
        prepare_initial_complex,
    )

    conformer_gen = build_conformer_generator(ctx, job)
    return prepare_initial_complex(ctx, job, conformer_gen)


def optimize_batch_worker(
    ctx: OptimizeWorkerContext,
    job: RosettaBatchJob,
) -> Tuple[str, Dict[str, object]]:
    """Optimize one prepared linker complex in a worker process."""
    from oligoternary.modeling.stages import optimize_initial_complex

    return optimize_initial_complex(ctx, job)


@dataclass
class LinkerModeler:
    molfile2params: str
    linker_prefix: str
    linker_smiles: str
    linker_chain: str
    e3l_smiles: str
    e3l_chain: str
    warhead_chain: str
    poi_chain: str
    e3_chain: str
    warhead_anchor_label: str
    e3l_anchor_label: str
    endpoint_candidates: int = ATTACHMENT_ENDPOINT_CANDIDATES
    conformers_per_endpoint: int = PREPARE_CONFORMER_CANDIDATES
    minimum_attachment_angle_degrees: float = MIN_ATTACHMENT_ANGLE_DEGREES
    conformer_ranking: str = CONFORMER_RANKING
    linker_to_warhead_index: Optional[int] = None
    linker_to_e3l_index: Optional[int] = None
    linker_to_warhead_atom: Optional[str] = None
    linker_to_e3l_atom: Optional[str] = None
    linker_prepared_smiles: Optional[str] = field(init=False, default=None)
    e3l_connect_atom: Optional[str] = field(init=False, default=None)
    linker_params_result: Optional[ParamsResult] = field(init=False, default=None)
    e3l_params_result: Optional[ParamsResult] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Reject identities that could mislabel chemistry or escape output paths."""

        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.linker_prefix) is None:
            raise ValueError(
                "linker_prefix must be a safe identifier using letters, digits, '.', '_', or '-'"
            )
        chain_fields = (
            "poi_chain",
            "warhead_chain",
            "e3_chain",
            "linker_chain",
            "e3l_chain",
        )
        chains = {field_name: getattr(self, field_name) for field_name in chain_fields}
        invalid_chains = [
            field_name
            for field_name, chain in chains.items()
            if re.fullmatch(r"[A-Za-z0-9]", chain) is None
        ]
        if invalid_chains:
            raise ValueError(
                "chain identities must be one alphanumeric PDB character; invalid: "
                + ", ".join(invalid_chains)
            )
        if len(set(chains.values())) != len(chains):
            raise ValueError("poi, warhead, E3, linker, and E3L chains must be distinct")
        if self.linker_smiles.count("[*]") != 2:
            raise ValueError("linker_smiles must contain exactly two [*] markers")
        if self.e3l_smiles.count("[*]") != 1:
            raise ValueError("e3l_smiles must contain exactly one [*] marker")
        warhead_anchor_chain = parse_pdb_atom_label(self.warhead_anchor_label)[0]
        e3l_anchor_chain = parse_pdb_atom_label(self.e3l_anchor_label)[0]
        if warhead_anchor_chain != self.warhead_chain:
            raise ValueError("warhead anchor chain must match warhead_chain")
        if e3l_anchor_chain != self.e3l_chain:
            raise ValueError("E3L anchor chain must match e3l_chain")

    def generate_rosetta_params(self, output_dir: str):
        """Generate linker and E3L Rosetta params; set connect-atom metadata."""
        from loguru import logger
        from oligoternary.modeling.params_generator import (
            ParamsGenerator as RosettaParamsGenerator,
        )

        generator = RosettaParamsGenerator(
            mol_to_params_path=self.molfile2params,
            output_dir=output_dir,
            clobber=True,
        )
        linker_result = generator.generate_params(
            smiles=self.linker_smiles,
            resname=LINKER_RESNAME,
            out_dir=output_dir,
        )
        e3l_result = generator.generate_params(
            smiles=self.e3l_smiles,
            resname=E3L_RESNAME,
            out_dir=output_dir,
        )
        if len(linker_result.connect_atom_indices) != 2 or len(linker_result.connect_atoms) != 2:
            raise ValueError(
                "Linker params generation must resolve exactly two connect atoms from "
                f"`[*]` markers. Got indices={linker_result.connect_atom_indices}, "
                f"atoms={linker_result.connect_atoms} for {self.linker_smiles}"
            )
        if len(e3l_result.connect_atoms) != 1:
            raise ValueError(
                "E3L params generation must resolve exactly one connect atom from "
                f"`[*]` markers. Got atoms={e3l_result.connect_atoms} for {self.e3l_smiles}"
            )

        derived_linker_indices = linker_result.connect_atom_indices
        derived_linker_atoms = linker_result.connect_atoms
        if self.linker_to_warhead_atom is not None or self.linker_to_e3l_atom is not None:
            if self.linker_to_warhead_atom is None or self.linker_to_e3l_atom is None:
                raise ValueError(
                    "linker_to_warhead_atom and linker_to_e3l_atom must be provided together "
                    "when explicitly binding linker connect roles."
                )
            if self.linker_to_warhead_atom == self.linker_to_e3l_atom:
                raise ValueError(
                    "linker_to_warhead_atom and linker_to_e3l_atom must refer to different "
                    f"connect atoms. Got {self.linker_to_warhead_atom}."
                )
            connect_index_by_atom = dict(
                zip(linker_result.connect_atoms, linker_result.connect_atom_indices)
            )
            missing_role_atoms = [
                atom_name
                for atom_name in (self.linker_to_warhead_atom, self.linker_to_e3l_atom)
                if atom_name not in connect_index_by_atom
            ]
            if missing_role_atoms:
                raise ValueError(
                    "Explicit linker role atom(s) are not present in the generated connect set: "
                    f"missing={missing_role_atoms}, generated={list(linker_result.connect_atoms)}"
                )
            derived_linker_atoms = (
                self.linker_to_warhead_atom,
                self.linker_to_e3l_atom,
            )
            derived_linker_indices = (
                connect_index_by_atom[self.linker_to_warhead_atom],
                connect_index_by_atom[self.linker_to_e3l_atom],
            )
        if self.linker_to_warhead_index is not None and self.linker_to_warhead_index != derived_linker_indices[0]:
            raise ValueError(
                f"linker_to_warhead_index mismatch: caller={self.linker_to_warhead_index}, "
                f"derived={derived_linker_indices[0]}"
            )
        if self.linker_to_e3l_index is not None and self.linker_to_e3l_index != derived_linker_indices[1]:
            raise ValueError(
                f"linker_to_e3l_index mismatch: caller={self.linker_to_e3l_index}, "
                f"derived={derived_linker_indices[1]}"
            )

        self.linker_params_result = linker_result
        self.e3l_params_result = e3l_result
        self.linker_prepared_smiles = linker_result.prepared_smiles
        self.linker_to_warhead_index, self.linker_to_e3l_index = derived_linker_indices
        self.linker_to_warhead_atom, self.linker_to_e3l_atom = derived_linker_atoms
        self.e3l_connect_atom = e3l_result.connect_atoms[0]
        logger.info(
            "Derived linker connect metadata from `[*]`-marked SMILES: "
            f"indices={derived_linker_indices}, atoms={derived_linker_atoms}, "
            f"prepared_smiles={self.linker_prepared_smiles}"
        )

    def _require_linker_connect_metadata(self, *, stage: str = "") -> None:
        during = f" during {stage}" if stage else ""
        if self.linker_prepared_smiles is None:
            raise RuntimeError(f"Linker prepared SMILES was not initialized{during}.")
        if self.linker_to_warhead_index is None or self.linker_to_e3l_index is None:
            raise RuntimeError(f"Linker connect atom indices were not initialized{during}.")
        if self.linker_to_warhead_atom is None or self.linker_to_e3l_atom is None:
            raise RuntimeError(f"Linker connect atom names were not initialized{during}.")

    def _prepare_worker_context(self) -> PrepareWorkerContext:
        self._require_linker_connect_metadata()
        if self.e3l_connect_atom is None:
            raise RuntimeError("E3L connect atom was not initialized.")
        return PrepareWorkerContext(
            linker_prefix=self.linker_prefix,
            linker_prepared_smiles=self.linker_prepared_smiles,
            linker_to_warhead_index=self.linker_to_warhead_index,
            linker_to_e3l_index=self.linker_to_e3l_index,
            linker_to_warhead_atom=self.linker_to_warhead_atom,
            linker_to_e3l_atom=self.linker_to_e3l_atom,
            linker_chain=self.linker_chain,
            e3l_chain=self.e3l_chain,
            e3l_connect_atom=self.e3l_connect_atom,
            warhead_anchor_label=self.warhead_anchor_label,
            e3l_anchor_label=self.e3l_anchor_label,
            endpoint_candidates=self.endpoint_candidates,
            conformers_per_endpoint=self.conformers_per_endpoint,
            minimum_attachment_angle_degrees=self.minimum_attachment_angle_degrees,
            conformer_ranking=self.conformer_ranking,
        )

    def _optimize_worker_context(self) -> OptimizeWorkerContext:
        if self.linker_to_warhead_atom is None or self.linker_to_e3l_atom is None:
            raise RuntimeError("Linker connect atom names were not initialized.")
        return OptimizeWorkerContext(
            linker_prefix=self.linker_prefix,
            linker_chain=self.linker_chain,
            warhead_anchor_label=self.warhead_anchor_label,
            linker_to_warhead_atom=self.linker_to_warhead_atom,
            linker_to_e3l_atom=self.linker_to_e3l_atom,
            poi_chain=self.poi_chain,
            warhead_chain=self.warhead_chain,
            e3_chain=self.e3_chain,
            e3l_chain=self.e3l_chain,
            minimum_attachment_angle_degrees=self.minimum_attachment_angle_degrees,
        )

    def _prepare_initial_complex(
        self,
        pdb_file: str,
        params_dir: str,
        tmp_dir: str,
        linker_min: float,
        linker_max: float,
        conformer_gen,
        minimize_steps: int,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> PreparedComplex:
        from oligoternary.modeling.stages import prepare_initial_complex

        ctx = self._prepare_worker_context()
        job = PrepareBatchJob(
            pdb_file=pdb_file,
            params_dir=params_dir,
            tmp_dir=tmp_dir,
            linker_min=linker_min,
            linker_max=linker_max,
            minimize_steps=minimize_steps,
            random_seed=seed,
        )
        return prepare_initial_complex(ctx, job, conformer_gen)

    def _optimize_initial_complex(
        self,
        prepared_item: PreparedComplex,
        params_dir: str,
        model_dir: str,
        relax_cycles: int,
        minimize_steps: int,
        minimize_tolerance: float,
        random_seed: int,
    ):
        from oligoternary.modeling.stages import optimize_initial_complex

        job = RosettaBatchJob(
            prepared_item=prepared_item,
            params_dir=params_dir,
            model_dir=model_dir,
            relax_cycles=relax_cycles,
            minimize_steps=minimize_steps,
            minimize_tolerance=minimize_tolerance,
            random_seed=random_seed,
            collect_metrics=False,
        )
        return optimize_initial_complex(self._optimize_worker_context(), job)

    def _ensure_params_ready(self, params_dir: str) -> None:
        """Generate Rosetta params once and validate linker connect metadata."""
        from loguru import logger

        self.generate_rosetta_params(params_dir)
        logger.success("Rosetta params generation complete.")
        self._require_linker_connect_metadata(stage="params generation")

    @staticmethod
    def _require_empty_managed_directories(
        output_dir: str, directory_names: Tuple[str, ...]
    ) -> None:
        """Refuse to mix a new run with pre-existing managed Artifacts."""

        occupied = []
        for name in directory_names:
            directory = os.path.join(output_dir, name)
            if os.path.exists(directory) and (
                not os.path.isdir(directory) or any(os.scandir(directory))
            ):
                occupied.append(directory)
        if occupied:
            raise FileExistsError(
                "Managed output directories must be absent or empty for a new run: "
                + ", ".join(occupied)
            )

    def run(
        self,
        pdb_file: str,
        output_dir: str,
        relax_cycles: int = 5,
        minimize_steps: int = 1000,
        minimize_tolerance: float = 0.001,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> bool:
        """Single-structure mode: params, conformer prep, Rosetta refine, output."""
        from loguru import logger

        from oligoternary.modeling.minimizer import LinkerConstrainedMinimizer
        if random_seed < 0:
            raise ValueError("random_seed must be >= 0")
        logger.info("Starting OligoTernary linker refinement")
        logger.info("Random seed: {}", random_seed)
        params_dir = os.path.join(output_dir, "params")
        tmp_dir = os.path.join(output_dir, "tmp")
        model_dir = os.path.join(output_dir, "models")
        input_stem = os.path.splitext(os.path.basename(pdb_file))[0]
        expected_model = os.path.abspath(
            os.path.join(model_dir, f"{input_stem}_{self.linker_prefix}_full_optimized.pdb")
        )
        existing_models = [
            os.path.abspath(path) for path in glob(os.path.join(model_dir, "*.pdb"))
        ]
        unexpected_models = [path for path in existing_models if path != expected_model]
        if unexpected_models:
            raise FileExistsError(
                "Output models directory contains PDBs from another run: "
                + ", ".join(unexpected_models)
            )
        if expected_model in existing_models:
            logger.warning("Re-running and replacing existing model: {}", expected_model)
        for d in [params_dir, tmp_dir, model_dir]:
            os.makedirs(d, exist_ok=True)

        # Generate params and validate
        self._ensure_params_ready(params_dir)

        # Generate constrained linker conformer
        conformer_gen = LinkerConstrainedMinimizer(
            linker_prefix=self.linker_prefix,
            smiles=self.linker_prepared_smiles,
            start_point=self.linker_to_warhead_index,
            end_point=self.linker_to_e3l_index,
            random_seed=random_seed,
            endpoint_candidates=self.endpoint_candidates,
            conformers_per_endpoint=self.conformers_per_endpoint,
            minimum_attachment_angle_degrees=self.minimum_attachment_angle_degrees,
            conformer_ranking=self.conformer_ranking,
        )
        linker_min, linker_max = conformer_gen._get_length()
        prepared_item = self._prepare_initial_complex(
            pdb_file=pdb_file,
            params_dir=params_dir,
            tmp_dir=tmp_dir,
            linker_min=linker_min,
            linker_max=linker_max,
            conformer_gen=conformer_gen,
            minimize_steps=minimize_steps,
            seed=random_seed,
        )
        self._optimize_initial_complex(
            prepared_item=prepared_item,
            params_dir=params_dir,
            model_dir=model_dir,
            relax_cycles=relax_cycles,
            minimize_steps=minimize_steps,
            minimize_tolerance=minimize_tolerance,
            random_seed=random_seed,
        )
        logger.success("OligoTernary linker refinement completed successfully.")
        return True

    @staticmethod
    def _expand_pdb_paths(pdb_files: List[str]) -> List[str]:
        """Expand directories to absolute PDB paths."""
        expanded = []
        for path in pdb_files:
            if os.path.isdir(path):
                expanded.extend(sorted(glob(os.path.join(path, "*.pdb"))))
            else:
                expanded.append(path)
        return [os.path.abspath(path) for path in expanded]

    @staticmethod
    def _validate_unique_input_basenames(pdb_files: List[str]) -> None:
        """Reject inputs that would share tmp/model/metrics identities."""

        paths_by_basename: Dict[str, List[str]] = {}
        for path in pdb_files:
            paths_by_basename.setdefault(os.path.basename(path), []).append(path)
        collisions = {
            name: paths for name, paths in paths_by_basename.items() if len(paths) > 1
        }
        if collisions:
            details = "; ".join(
                f"{name}: {', '.join(paths)}"
                for name, paths in sorted(collisions.items())
            )
            raise ValueError(
                "Batch inputs must have unique basenames because generated tmp/model "
                f"paths use the basename identity; collisions: {details}"
            )

    def _validate_batch_params(self, params_dir: str) -> None:
        """Generate Rosetta params and validate linker connect atom metadata."""
        self._ensure_params_ready(params_dir)

    @staticmethod
    def _build_prepare_jobs(
        pdb_files: List[str],
        *,
        params_dir: str,
        tmp_dir: str,
        linker_min: float,
        linker_max: float,
        minimize_steps: int,
        random_seed: int,
    ) -> List[PrepareBatchJob]:
        """Build deterministic per-input jobs from the public batch seed."""

        return [
            PrepareBatchJob(
                pdb_file=pdb_file,
                params_dir=params_dir,
                tmp_dir=tmp_dir,
                linker_min=linker_min,
                linker_max=linker_max,
                minimize_steps=minimize_steps,
                random_seed=random_seed,
            )
            for pdb_file in pdb_files
        ]

    def _prepare_all_complexes(
        self,
        prepare_jobs: List[PrepareBatchJob],
        num_processes: int,
        spawn_context,
    ) -> Tuple[List[PreparedComplex], List[BatchFailure]]:
        """Run prepare stage (parallel or serial); return successes and failures."""
        import tqdm
        import concurrent.futures
        from concurrent.futures import ProcessPoolExecutor
        from loguru import logger

        prepared_by_file: Dict[str, PreparedComplex] = {}
        failures: List[BatchFailure] = []
        prepare_ctx = self._prepare_worker_context()
        if num_processes == 1:
            for job in tqdm.tqdm(prepare_jobs, desc="Preparing"):
                try:
                    prepared = prepare_batch_worker(prepare_ctx, job)
                except Exception as exc:  # worker boundary: log and continue batch
                    logger.exception(f"Preparation worker failed for {job.pdb_file}")
                    failures.append(
                        BatchFailure(
                            job.pdb_file,
                            "prepare",
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                prepared_by_file[job.pdb_file] = prepared
        else:
            with ProcessPoolExecutor(
                max_workers=num_processes,
                mp_context=spawn_context,
            ) as executor:
                future_map = {
                    executor.submit(prepare_batch_worker, prepare_ctx, job): job
                    for job in prepare_jobs
                }
                with tqdm.tqdm(total=len(prepare_jobs), desc="Preparing") as pbar:
                    for future in concurrent.futures.as_completed(future_map):
                        job = future_map[future]
                        try:
                            prepared_by_file[job.pdb_file] = future.result()
                        except Exception as exc:  # worker boundary: log and continue batch
                            logger.exception(f"Preparation worker failed for {job.pdb_file}")
                            failures.append(
                                BatchFailure(
                                    job.pdb_file,
                                    "prepare",
                                    f"{type(exc).__name__}: {exc}",
                                )
                            )
                        pbar.update(1)

        prepared_items = [
            prepared_by_file[job.pdb_file]
            for job in prepare_jobs
            if job.pdb_file in prepared_by_file
        ]
        return prepared_items, failures

    def _run_rosetta_jobs(
        self,
        rosetta_jobs: List[RosettaBatchJob],
        rosetta_num_processes: int,
        spawn_context,
    ) -> Tuple[List[Tuple[str, Dict]], List[BatchFailure]]:
        """Run Rosetta optimization (parallel or serial)."""
        import tqdm
        import concurrent.futures
        from concurrent.futures import ProcessPoolExecutor
        from loguru import logger

        results: List[Tuple[str, Dict]] = []
        failures: List[BatchFailure] = []
        optimize_ctx = self._optimize_worker_context()
        if rosetta_num_processes == 1:
            for job in tqdm.tqdm(rosetta_jobs, desc="Rosetta"):
                try:
                    result = optimize_batch_worker(optimize_ctx, job)
                except Exception as exc:  # worker boundary: log and continue batch
                    logger.exception(
                        f"Rosetta worker failed for {job.prepared_item.filename}"
                    )
                    failures.append(
                        BatchFailure(
                            job.prepared_item.filename,
                            "rosetta",
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                results.append(result)
        else:
            with ProcessPoolExecutor(
                max_workers=rosetta_num_processes,
                mp_context=spawn_context,
            ) as executor:
                future_map = {
                    executor.submit(optimize_batch_worker, optimize_ctx, job): job
                    for job in rosetta_jobs
                }
                with tqdm.tqdm(total=len(rosetta_jobs), desc="Rosetta") as pbar:
                    for future in concurrent.futures.as_completed(future_map):
                        job = future_map[future]
                        try:
                            results.append(future.result())
                        except Exception as exc:  # worker boundary: log and continue batch
                            logger.exception(
                                f"Rosetta worker failed for {job.prepared_item.filename}"
                            )
                            failures.append(
                                BatchFailure(
                                    job.prepared_item.filename,
                                    "rosetta",
                                    f"{type(exc).__name__}: {exc}",
                                )
                            )
                        pbar.update(1)
        return results, failures

    @staticmethod
    def _write_metrics_csv(
        results: List[Tuple[str, Dict]],
        metrics_csv: str,
        total_files: int,
    ) -> str:
        """Write batch metrics CSV; return output path."""
        import pandas as pd
        from loguru import logger

        data = [m for _, m in results]
        df = pd.DataFrame(data)
        if "model_name" not in df.columns:
            df["model_name"] = [name for name, _ in results]
        df.to_csv(metrics_csv, index=False)
        logger.success(f"Processed {len(results)}/{total_files} files → {metrics_csv}")
        return metrics_csv

    def run_batch(
        self,
        pdb_files: List[str],
        output_dir: str,
        num_processes: Optional[int] = None,
        rosetta_num_processes: int = 24,
        relax_cycles: int = 5,
        minimize_steps: int = 1000,
        minimize_tolerance: float = 0.001,
        strict: bool = True,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> BatchRunSummary:
        """Batch mode: parallel prepare + Rosetta refine; write metrics CSV."""
        from loguru import logger

        from oligoternary.modeling.minimizer import LinkerConstrainedMinimizer
        from oligoternary.modeling.runtime import default_prepare_process_count

        pdb_files = self._expand_pdb_paths(pdb_files)
        self._validate_unique_input_basenames(pdb_files)
        total_inputs = len(pdb_files)
        if num_processes is None:
            num_processes = default_prepare_process_count()
        if num_processes < 1:
            raise ValueError("num_processes must be >= 1")
        if rosetta_num_processes < 1:
            raise ValueError("rosetta_num_processes must be >= 1")
        if random_seed < 0:
            raise ValueError("random_seed must be >= 0")

        self._require_empty_managed_directories(
            output_dir, ("params", "tmp", "models", "metrics")
        )

        # PyRosetta C++ globals are not fork-safe; spawn lets each worker init independently.
        spawn_context = multiprocessing.get_context("spawn")
        logger.info(
            f"Batch: {len(pdb_files)} PDB files, "
            f"prepare_processes={num_processes}, rosetta_processes={rosetta_num_processes}, "
            f"mp_start_method=spawn, random_seed={random_seed}"
        )

        # Directories
        params_dir = os.path.join(output_dir, "params")
        tmp_dir = os.path.join(output_dir, "tmp")
        model_dir = os.path.join(output_dir, "models")
        metrics_dir = os.path.join(output_dir, "metrics")
        for d in [params_dir, tmp_dir, model_dir, metrics_dir]:
            os.makedirs(d, exist_ok=True)

        # Generate params (once) and validate
        self._validate_batch_params(params_dir)

        # Linker constraints
        conformer_gen = LinkerConstrainedMinimizer(
            linker_prefix=self.linker_prefix,
            smiles=self.linker_prepared_smiles,
            start_point=self.linker_to_warhead_index,
            end_point=self.linker_to_e3l_index,
            random_seed=random_seed,
            endpoint_candidates=self.endpoint_candidates,
            conformers_per_endpoint=self.conformers_per_endpoint,
            minimum_attachment_angle_degrees=self.minimum_attachment_angle_degrees,
            conformer_ranking=self.conformer_ranking,
        )
        linker_min, linker_max = conformer_gen._get_length()

        # Prepare stage
        prepare_jobs = self._build_prepare_jobs(
            pdb_files,
            params_dir=params_dir,
            tmp_dir=tmp_dir,
            linker_min=linker_min,
            linker_max=linker_max,
            minimize_steps=minimize_steps,
            random_seed=random_seed,
        )
        prepared_items, batch_failures = self._prepare_all_complexes(
            prepare_jobs, num_processes, spawn_context
        )
        prepared_count = len(prepared_items)
        failed_inputs_path = os.path.join(metrics_dir, "failed_inputs.tsv")
        logger.info(f"Prepared {prepared_count}/{total_inputs} complexes for Rosetta")
        if not prepared_items:
            logger.error("No initial complexes generated")
            failed_tsv = write_failed_inputs_tsv(batch_failures, failed_inputs_path)
            if failed_tsv:
                logger.warning(f"Wrote {len(batch_failures)} batch failure(s) to {failed_tsv}")
            return BatchRunSummary(
                None,
                total_inputs,
                0,
                0,
                failed_inputs_tsv=failed_tsv,
                random_seed=random_seed,
            )

        # Rosetta stage
        rosetta_jobs = [
            RosettaBatchJob(
                prepared_item=prepared_item,
                params_dir=params_dir,
                model_dir=model_dir,
                relax_cycles=relax_cycles,
                minimize_steps=minimize_steps,
                minimize_tolerance=minimize_tolerance,
                random_seed=random_seed,
                collect_metrics=True,
            )
            for prepared_item in prepared_items
        ]
        results, rosetta_failures = self._run_rosetta_jobs(
            rosetta_jobs, rosetta_num_processes, spawn_context
        )
        batch_failures.extend(rosetta_failures)
        succeeded_count = len(results)

        failed_tsv = write_failed_inputs_tsv(batch_failures, failed_inputs_path)
        if failed_tsv:
            logger.warning(f"Wrote {len(batch_failures)} batch failure(s) to {failed_tsv}")

        metrics_csv = os.path.join(metrics_dir, "linker_metrics.csv")
        if not results:
            logger.error("No results generated")
            return BatchRunSummary(
                None,
                total_inputs,
                prepared_count,
                0,
                failed_inputs_tsv=failed_tsv,
                random_seed=random_seed,
            )

        sorted_results = sort_batch_results_by_input_order(results, pdb_files)
        csv_path = self._write_metrics_csv(sorted_results, metrics_csv, total_inputs)
        if strict and succeeded_count != total_inputs:
            logger.error(
                f"Strict batch mode: only {succeeded_count}/{total_inputs} inputs produced metrics"
            )
        return BatchRunSummary(
            csv_path,
            total_inputs,
            prepared_count,
            succeeded_count,
            failed_inputs_tsv=failed_tsv,
            random_seed=random_seed,
        )
