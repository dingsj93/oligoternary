"""Cartesian score-function factories for linker refinement stages."""
from __future__ import annotations

from typing import Literal

import pyrosetta
from pyrosetta import create_score_function
from pyrosetta.rosetta.core.scoring import ScoreType

CartScoreStage = Literal["relax", "connect", "clash"]


def build_cart_scorefxn(stage: CartScoreStage):
    scorefxn = create_score_function("ref2015_cart")
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_elec, 0.8)
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_sol, 0.7)
    scorefxn.set_weight(ScoreType.angle_constraint, 20.0)

    if stage == "relax":
        scorefxn.set_weight(ScoreType.atom_pair_constraint, 10.0)
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 0.5)
    elif stage == "connect":
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 1.5)
        scorefxn.set_weight(ScoreType.atom_pair_constraint, 10.0)
    elif stage == "clash":
        scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 1.5)
        scorefxn.set_weight(ScoreType.atom_pair_constraint, 50.0)
        scorefxn.set_weight(ScoreType.coordinate_constraint, 10.0)
    else:
        raise ValueError(f"Unknown cart score stage: {stage}")

    return scorefxn
