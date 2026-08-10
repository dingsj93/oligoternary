"""Prepare and optimize stage implementations for batch workers."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, Tuple

from loguru import logger

if TYPE_CHECKING:
    from pyrosetta import Pose

from oligoternary.modeling.constants import E3L_RESNAME, LINKER_RESNAME
from oligoternary.modeling.minimizer import LinkerConstrainedMinimizer
from oligoternary.modeling.optimizer import RosettaLinkerOptimizer
from oligoternary.modeling.types import (
    OptimizeWorkerContext,
    PreparedComplex,
    PrepareBatchJob,
    PrepareWorkerContext,
    RosettaBatchJob,
    parse_pdb_atom_label,
)
from oligoternary.modeling.atom_names import (
    fix_residue_atom_names,
    remap_atom_label,
)


def prepare_initial_complex(
    ctx: PrepareWorkerContext,
    job: PrepareBatchJob,
    conformer_gen: LinkerConstrainedMinimizer,
) -> PreparedComplex:
    base_name = os.path.splitext(os.path.basename(job.pdb_file))[0]
    initial_complex = conformer_gen.get_linker_conformer(
        pdb_file=job.pdb_file,
        warhead_anchor_label=ctx.warhead_anchor_label,
        e3l_anchor_label=ctx.e3l_anchor_label,
        linker_to_warhead_atom=ctx.linker_to_warhead_atom,
        linker_to_e3l_atom=ctx.linker_to_e3l_atom,
        output_dir=job.tmp_dir,
        output_prefix=base_name,
        linker_chain=ctx.linker_chain,
        e3l_chain=ctx.e3l_chain,
        linker_max_distance=job.linker_max,
        linker_min_distance=job.linker_min,
        minimize_steps=job.minimize_steps,
        seed=job.random_seed,
    )
    rename_map = fix_residue_atom_names(
        pdb_file=initial_complex,
        chain_id=ctx.e3l_chain,
        resname=E3L_RESNAME,
        params_pdb_file=os.path.join(job.params_dir, f"{E3L_RESNAME}_0001.pdb"),
    )
    logger.info(
        f"Aligned {E3L_RESNAME} atom names to params reference: "
        f"{len(rename_map)} atom(s) renamed for {base_name}"
    )
    rosetta_e3l_anchor_label = remap_atom_label(ctx.e3l_anchor_label, rename_map)
    logger.info(
        f"E3L anchor label remapped for Rosetta: "
        f"{ctx.e3l_anchor_label} -> {rosetta_e3l_anchor_label} for {base_name}"
    )
    rosetta_e3l_anchor_atom = parse_pdb_atom_label(rosetta_e3l_anchor_label)[3]
    if rosetta_e3l_anchor_atom != ctx.e3l_connect_atom:
        raise ValueError(
            "Remapped E3L anchor does not match the connect atom declared in generated params: "
            f"anchor={rosetta_e3l_anchor_atom}, connect={ctx.e3l_connect_atom}, "
            f"label={rosetta_e3l_anchor_label}"
        )
    return PreparedComplex(
        base_name=base_name,
        filename=os.path.basename(job.pdb_file),
        initial_complex=initial_complex,
        input_e3l_anchor_label=ctx.e3l_anchor_label,
        rosetta_e3l_anchor_label=rosetta_e3l_anchor_label,
    )


def optimize_initial_complex(
    ctx: OptimizeWorkerContext,
    job: RosettaBatchJob,
) -> Tuple[str, Dict]:
    prepared_item = job.prepared_item
    base_name = prepared_item.base_name
    optimizer = RosettaLinkerOptimizer(
        linker_prefix=ctx.linker_prefix,
        linker_params=os.path.join(job.params_dir, f"{LINKER_RESNAME}.params"),
        e3l_params=os.path.join(job.params_dir, f"{E3L_RESNAME}.params"),
        random_seed=job.random_seed,
    )
    pose, output_file = optimizer.fit(
        pdb_file=prepared_item.initial_complex,
        output_dir=job.model_dir,
        output_prefix=base_name,
        warhead_anchor_label=ctx.warhead_anchor_label,
        e3l_anchor_label=prepared_item.rosetta_e3l_anchor_label,
        linker_chain=ctx.linker_chain,
        linker_to_warhead_atom=ctx.linker_to_warhead_atom,
        linker_to_e3l_atom=ctx.linker_to_e3l_atom,
        relax_cycles=job.relax_cycles,
        minimize_steps=job.minimize_steps,
        minimize_tolerance=job.minimize_tolerance,
        minimum_attachment_angle_degrees=ctx.minimum_attachment_angle_degrees,
    )
    metrics = _analyze_pose(ctx, pose) if job.collect_metrics else {}
    metrics["filename"] = prepared_item.filename
    metrics["model_name"] = base_name
    metrics["output_file"] = os.path.basename(output_file)
    metrics["input_e3l_anchor_label"] = prepared_item.input_e3l_anchor_label
    metrics["rosetta_e3l_anchor_label"] = prepared_item.rosetta_e3l_anchor_label
    metrics["warhead_anchor_label"] = ctx.warhead_anchor_label
    clash_stats = optimizer.last_clash_stats or {}
    clash_before = clash_stats.get("before", {})
    clash_after = clash_stats.get("after", {})
    metrics["linker_env_clashes_before_relief"] = clash_before.get("count")
    metrics["linker_env_clashes_after_relief"] = clash_after.get("count")
    metrics["linker_env_min_distance_before_relief"] = clash_before.get("min_distance")
    metrics["linker_env_min_distance_after_relief"] = clash_after.get("min_distance")
    metrics["linker_env_overlap_sum_before_relief"] = clash_before.get("overlap_sum")
    metrics["linker_env_overlap_sum_after_relief"] = clash_after.get("overlap_sum")
    metrics["linker_env_worst_overlap_before_relief"] = clash_before.get("worst_overlap")
    metrics["linker_env_worst_overlap_after_relief"] = clash_after.get("worst_overlap")
    return base_name, metrics


def _analyze_pose(ctx: OptimizeWorkerContext, pose: "Pose") -> Dict[str, object]:
    from oligoternary.analysis.metrics import ComplexAnalyzer

    protac_chains = [ctx.warhead_chain, ctx.linker_chain, ctx.e3l_chain]
    analyzer = ComplexAnalyzer()
    return analyzer.analyze_complex(
        pose,
        poi_chain=ctx.poi_chain,
        warhead_chain=ctx.warhead_chain,
        e3_chain=ctx.e3_chain,
        protac_chains=protac_chains,
    )


def build_conformer_generator(
    ctx: PrepareWorkerContext,
    job: PrepareBatchJob,
) -> LinkerConstrainedMinimizer:
    return LinkerConstrainedMinimizer(
        linker_prefix=ctx.linker_prefix,
        smiles=ctx.linker_prepared_smiles,
        start_point=ctx.linker_to_warhead_index,
        end_point=ctx.linker_to_e3l_index,
        random_seed=job.random_seed,
        endpoint_candidates=ctx.endpoint_candidates,
        conformers_per_endpoint=ctx.conformers_per_endpoint,
        minimum_attachment_angle_degrees=ctx.minimum_attachment_angle_degrees,
        conformer_ranking=ctx.conformer_ranking,
    )
