"""PyRosetta runtime initialization helpers for linker refinement."""
import multiprocessing
import os
from typing import Tuple

from oligoternary.modeling.pyrosetta_runtime import (
    build_pyrosetta_init_options as _build_pyrosetta_init_options,
    ensure_pyrosetta_initialized,
)

__all__ = [
    "build_pyrosetta_init_options",
    "ensure_pyrosetta_initialized",
    "default_prepare_process_count",
]


def build_pyrosetta_init_options(
    linker_params: str,
    e3l_params: str,
    random_seed: int,
) -> Tuple[str, ...]:
    linker_params = os.path.abspath(linker_params)
    e3l_params = os.path.abspath(e3l_params)
    return _build_pyrosetta_init_options(
        [e3l_params, linker_params],
        random_seed=random_seed,
    )


def default_prepare_process_count() -> int:
    """Default batch prepare worker count: reserve one core, minimum one worker."""
    cpu_count = multiprocessing.cpu_count() or 1
    return max(1, cpu_count - 1)
