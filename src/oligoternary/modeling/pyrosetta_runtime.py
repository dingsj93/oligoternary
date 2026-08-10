"""Shared PyRosetta initialization for refinement and structural metrics."""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

_PYROSETTA_INIT_SIGNATURE: Optional[Tuple[str, ...]] = None


def build_pyrosetta_init_options(
    extra_res_fa: Sequence[str],
    *,
    ignore_unrecognized_res: bool = False,
    sort_extra_res_fa: bool = False,
    random_seed: Optional[int] = None,
) -> Tuple[str, ...]:
    """Build a deterministic PyRosetta init token tuple."""
    init_tokens = [
        "-mute", "all",
        "-ex1", "-ex2",
        "-use_input_sc",
        "-flip_HNQ",
        "-no_optH", "false",
        "-load_PDB_components", "false",
    ]
    if ignore_unrecognized_res:
        init_tokens.append("-ignore_unrecognized_res")
    if random_seed is not None:
        if random_seed < 0:
            raise ValueError("random_seed must be >= 0")
        init_tokens.extend(["-constant_seed", "-jran", str(random_seed)])

    normalized_params: List[str] = []
    for params_file in extra_res_fa:
        abs_path = os.path.abspath(params_file)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Params file not found: {params_file}")
        normalized_params.append(abs_path)

    if sort_extra_res_fa:
        normalized_params = sorted(dict.fromkeys(normalized_params))
    else:
        normalized_params = list(dict.fromkeys(normalized_params))

    for params_file in normalized_params:
        init_tokens.extend(["-extra_res_fa", params_file])

    return tuple(init_tokens)


def build_metrics_init_options(
    extra_params_list: List[str], *, random_seed: int = 42
) -> Tuple[str, ...]:
    return build_pyrosetta_init_options(
        extra_params_list,
        ignore_unrecognized_res=True,
        sort_extra_res_fa=True,
        random_seed=random_seed,
    )


def _pyrosetta_runtime_is_initialized() -> bool:
    import pyrosetta

    rosetta_basic = getattr(getattr(pyrosetta, "rosetta", None), "basic", None)
    was_init_called = getattr(rosetta_basic, "was_init_called", None)
    if callable(was_init_called):
        return bool(was_init_called())
    is_initialized = getattr(pyrosetta, "is_initialized", None)
    if callable(is_initialized):
        return bool(is_initialized())
    return _PYROSETTA_INIT_SIGNATURE is not None


def ensure_pyrosetta_initialized(init_options: Tuple[str, ...]) -> None:
    """Initialize PyRosetta once per process with a stable option signature."""
    global _PYROSETTA_INIT_SIGNATURE
    import pyrosetta
    from loguru import logger

    if _PYROSETTA_INIT_SIGNATURE is None:
        if _pyrosetta_runtime_is_initialized():
            raise RuntimeError(
                "PyRosetta was already initialized before the expected init options "
                "were applied. Restart the process and initialize PyRosetta only once "
                "with the required -extra_res_fa files."
            )
        pyrosetta.init(" ".join(init_options))
        _PYROSETTA_INIT_SIGNATURE = init_options
        logger.info("PyRosetta initialized successfully")
        return

    if _PYROSETTA_INIT_SIGNATURE != init_options:
        raise RuntimeError(
            "PyRosetta runtime already initialized with a different configuration.\n"
            f"Existing: {_PYROSETTA_INIT_SIGNATURE}\n"
            f"Requested: {init_options}"
        )
