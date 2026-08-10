"""Attachment endpoint placement for linker embedding."""
from typing import List, Tuple

import numpy as np
from loguru import logger

from oligoternary.modeling.constants import (
    ATTACHMENT_DIRECTION_SAMPLES,
    ATTACHMENT_ENDPOINT_CANDIDATES,
    ATTACHMENT_TOP_CANDIDATES,
)
from oligoternary.modeling.geometry import (
    generate_attachment_directions,
    score_attachment_position,
)
from oligoternary.modeling.params_io import ideal_bond_length_from_atom_names
from oligoternary.modeling.types import HeavyAtomRecord

def build_attachment_endpoint_candidates(
    warhead_coord,
    e3l_coord,
    warhead_anchor_atom: str,
    e3l_anchor_atom: str,
    linker_to_warhead_atom: str,
    linker_to_e3l_atom: str,
    environment_atoms: List[HeavyAtomRecord],
    warhead_anchor_label: str,
    e3l_anchor_label: str,
    linker_min_distance: float,
    linker_max_distance: float,
    top_k: int = ATTACHMENT_ENDPOINT_CANDIDATES,
) -> List[Tuple[List[float], List[float], float]]:
    """Return up to ``top_k`` feasible ``(warhead_target, e3l_target, distance)``
    candidate pairs, ranked by the same local-clearance score.

    Because the local (4 Å) clearance score is blind to the rest of the linker
    body, a single best pair often collapsed the linker into the anchor
    neighborhood. Returning multiple feasible seeds lets the downstream RDKit
    constrained-embed stage pick the conformer whose full body scores best.
    """
    warhead_xyz = np.asarray(warhead_coord, dtype=float)
    e3l_xyz = np.asarray(e3l_coord, dtype=float)
    bridge_vector = e3l_xyz - warhead_xyz
    anchor_distance = float(np.linalg.norm(bridge_vector))
    if anchor_distance <= 0.0:
        raise ValueError("Anchor atoms occupy the same coordinates; cannot place linker endpoints.")

    warhead_bond_length = ideal_bond_length_from_atom_names(
        warhead_anchor_atom,
        linker_to_warhead_atom,
        1,
    )
    e3l_bond_length = ideal_bond_length_from_atom_names(
        e3l_anchor_atom,
        linker_to_e3l_atom,
        1,
    )
    if anchor_distance <= warhead_bond_length + e3l_bond_length:
        raise ValueError(
            "Anchor atoms are too close to place linker attachment atoms at ideal bond lengths: "
            f"anchor_distance={anchor_distance:.2f} Å, "
            f"required>{warhead_bond_length + e3l_bond_length:.2f} Å"
        )
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    bridge_unit = bridge_vector / anchor_distance
    directions = [bridge_unit, -bridge_unit]
    directions.extend(generate_attachment_directions(ATTACHMENT_DIRECTION_SAMPLES))

    def _rank(candidates):
        candidates.sort(
            key=lambda candidate: (
                candidate["overlap_sum"],
                -candidate["min_clearance"],
                -candidate["toward_partner"],
            )
        )

    warhead_candidates = []
    for direction in directions:
        toward_partner = float(np.dot(direction, bridge_unit))
        if toward_partner <= 0.0:
            continue
        position = warhead_xyz + direction * warhead_bond_length
        overlap_sum, min_clearance = score_attachment_position(
            position=position,
            linker_element=linker_to_warhead_atom[0],
            environment_atoms=environment_atoms,
            excluded_labels=[warhead_anchor_label],
            anchor_center=warhead_xyz,
        )
        warhead_candidates.append(
            {
                "position": position,
                "overlap_sum": overlap_sum,
                "min_clearance": min_clearance,
                "toward_partner": toward_partner,
            }
        )
    _rank(warhead_candidates)

    e3l_candidates = []
    for direction in directions:
        toward_partner = float(np.dot(-direction, bridge_unit))
        if toward_partner <= 0.0:
            continue
        position = e3l_xyz + direction * e3l_bond_length
        overlap_sum, min_clearance = score_attachment_position(
            position=position,
            linker_element=linker_to_e3l_atom[0],
            environment_atoms=environment_atoms,
            excluded_labels=[e3l_anchor_label],
            anchor_center=e3l_xyz,
        )
        e3l_candidates.append(
            {
                "position": position,
                "overlap_sum": overlap_sum,
                "min_clearance": min_clearance,
                "toward_partner": toward_partner,
            }
        )
    _rank(e3l_candidates)

    if not warhead_candidates or not e3l_candidates:
        raise ValueError("No forward-facing attachment directions available for steric search.")

    scored_pairs = []
    for warhead_candidate in warhead_candidates[:ATTACHMENT_TOP_CANDIDATES]:
        for e3l_candidate in e3l_candidates[:ATTACHMENT_TOP_CANDIDATES]:
            attachment_distance = float(
                np.linalg.norm(
                    e3l_candidate["position"] - warhead_candidate["position"]
                )
            )
            if not (linker_min_distance < attachment_distance < linker_max_distance):
                continue
            attachment_bridge = (
                e3l_candidate["position"] - warhead_candidate["position"]
            ) / attachment_distance
            score = (
                warhead_candidate["overlap_sum"] + e3l_candidate["overlap_sum"],
                -min(
                    warhead_candidate["min_clearance"],
                    e3l_candidate["min_clearance"],
                ),
                -(warhead_candidate["toward_partner"] + e3l_candidate["toward_partner"]),
                -float(np.dot(attachment_bridge, bridge_unit)),
            )
            scored_pairs.append(
                (
                    score,
                    warhead_candidate["position"],
                    e3l_candidate["position"],
                    attachment_distance,
                    warhead_candidate,
                    e3l_candidate,
                )
            )

    if not scored_pairs:
        raise ValueError(
            "No sterically feasible attachment placement found within linker distance range: "
            f"range={linker_min_distance:.2f}-{linker_max_distance:.2f} Å, "
            f"anchor_distance={anchor_distance:.2f} Å"
        )

    scored_pairs.sort(key=lambda entry: entry[0])
    top_pairs = scored_pairs[:top_k]

    logger.info(
        f"Attachment placement: {len(scored_pairs)} feasible pairs; "
        f"keeping top {len(top_pairs)}. Best pair: "
        f"warhead_clearance={top_pairs[0][4]['min_clearance']:.3f} Å, "
        f"e3l_clearance={top_pairs[0][5]['min_clearance']:.3f} Å, "
        f"attachment_distance={top_pairs[0][3]:.2f} Å"
    )

    return [
        (
            start.tolist(),
            end.tolist(),
            attachment_distance,
        )
        for (_score, start, end, attachment_distance, _w, _e) in top_pairs
    ]
