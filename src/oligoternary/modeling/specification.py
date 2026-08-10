"""Linker-refinement specification parsing and CLI adapter compilation."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from oligoternary.modeling.constants import (
    ATTACHMENT_ENDPOINT_CANDIDATES,
    CONFORMER_RANKING,
    CONFORMER_RANKINGS,
    E3L_RESNAME,
    LINKER_RESNAME,
    MIN_ATTACHMENT_ANGLE_DEGREES,
    PREPARE_CONFORMER_CANDIDATES,
)


class ScientificSpecError(ValueError):
    """Raised when a scientific Stage specification is inconsistent."""


@dataclass(frozen=True)
class RefinementChemistry:
    linker_prefix: str
    linker_smiles: str
    e3_ligand_smiles: str
    linker_to_warhead_atom: str
    linker_to_e3_ligand_atom: str


@dataclass(frozen=True)
class ChainMapping:
    poi: str
    warhead: str
    e3: str
    linker: str
    e3_ligand: str


@dataclass(frozen=True)
class AnchorMapping:
    warhead: str
    e3_ligand: str


@dataclass(frozen=True)
class RefinementProtocol:
    relax_cycles: int = 5
    minimize_steps: int = 1000
    minimize_tolerance: float = 0.001
    random_seed: int = 42
    endpoint_candidates: int = ATTACHMENT_ENDPOINT_CANDIDATES
    conformers_per_endpoint: int = PREPARE_CONFORMER_CANDIDATES
    minimum_attachment_angle_degrees: float = MIN_ATTACHMENT_ANGLE_DEGREES
    conformer_ranking: str = CONFORMER_RANKING


@dataclass(frozen=True)
class RefinementTools:
    molfile_to_params: Optional[Path] = None
    rosetta_root: Optional[Path] = None


@dataclass(frozen=True)
class RefinementSpec:
    input_pdb: Path
    output_dir: Path
    result_summary: Optional[Path]
    chemistry: RefinementChemistry
    chains: ChainMapping
    anchors: AnchorMapping
    protocol: RefinementProtocol
    tools: RefinementTools

    @property
    def artifact(self) -> Path:
        filename = (
            f"{self.input_pdb.stem}_{self.chemistry.linker_prefix}"
            "_full_optimized.pdb"
        )
        return self.output_dir / "models" / filename

    @property
    def linker_params(self) -> Path:
        return self.output_dir / "params" / f"{LINKER_RESNAME}.params"

    @property
    def e3_ligand_params(self) -> Path:
        return self.output_dir / "params" / f"{E3L_RESNAME}.params"


_ROOT_KEYS = {
    "type",
    "input",
    "result_summary",
    "output_dir",
    "chemistry",
    "chains",
    "anchors",
    "protocol",
    "tools",
}
_CHEMISTRY_KEYS = {
    "linker_prefix",
    "linker_smiles",
    "e3_ligand_smiles",
    "linker_to_warhead_atom",
    "linker_to_e3_ligand_atom",
}
_CHAIN_KEYS = {"poi", "warhead", "e3", "linker", "e3_ligand"}
_ANCHOR_KEYS = {"warhead", "e3_ligand"}
_PROTOCOL_KEYS = {
    "relax_cycles",
    "minimize_steps",
    "minimize_tolerance",
    "random_seed",
    "endpoint_candidates",
    "conformers_per_endpoint",
    "minimum_attachment_angle_degrees",
    "conformer_ranking",
}
_TOOL_KEYS = {"molfile_to_params", "rosetta_root"}
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_PDB_CHAIN_ID = re.compile(r"^[A-Za-z0-9]$")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScientificSpecError(f"{context} must be a mapping")
    return value


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ScientificSpecError(
            f"{context} has unknown field(s): {', '.join(str(item) for item in unknown)}"
        )


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScientificSpecError(f"{context} must be a non-empty string")
    return value.strip()


def _path(value: Any, config_dir: Path, context: str) -> Path:
    path = Path(_text(value, context)).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _positive_int(value: Any, context: str, default: int) -> int:
    if value is None:
        return default
    if type(value) is not int or value < 1:
        raise ScientificSpecError(f"{context} must be a positive integer")
    return value


def _positive_float(value: Any, context: str, default: float) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ScientificSpecError(f"{context} must be a positive number")
    return float(value)


def _nonnegative_int(value: Any, context: str, default: int) -> int:
    if value is None:
        return default
    if type(value) is not int or value < 0:
        raise ScientificSpecError(f"{context} must be a non-negative integer")
    return value


def _angle(value: Any, context: str, default: float) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScientificSpecError(f"{context} must be a number between 0 and 180")
    angle = float(value)
    if not 0 <= angle <= 180:
        raise ScientificSpecError(f"{context} must be a number between 0 and 180")
    return angle


def _anchor_chain(label: str, context: str) -> str:
    parts = label.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ScientificSpecError(
            f"{context} must use chain:residue-number:atom-name"
        )
    try:
        int(parts[1])
    except ValueError as exc:
        raise ScientificSpecError(f"{context} residue number must be an integer") from exc
    return parts[0]


def parse_refinement_spec(
    raw_value: Mapping[str, Any], config_dir: Path
) -> RefinementSpec:
    """Validate scientific intent while hiding the CLI implementation."""

    raw = _mapping(raw_value, "linker-refinement adapter")
    _reject_unknown(raw, _ROOT_KEYS, "linker-refinement adapter")
    if raw.get("type") != "linker-refinement":
        raise ScientificSpecError("adapter.type must be linker-refinement")

    chemistry_raw = _mapping(raw.get("chemistry"), "chemistry")
    chains_raw = _mapping(raw.get("chains"), "chains")
    anchors_raw = _mapping(raw.get("anchors"), "anchors")
    protocol_raw = _mapping(raw.get("protocol", {}), "protocol")
    tools_raw = _mapping(raw.get("tools", {}), "tools")
    _reject_unknown(chemistry_raw, _CHEMISTRY_KEYS, "chemistry")
    _reject_unknown(chains_raw, _CHAIN_KEYS, "chains")
    _reject_unknown(anchors_raw, _ANCHOR_KEYS, "anchors")
    _reject_unknown(protocol_raw, _PROTOCOL_KEYS, "protocol")
    _reject_unknown(tools_raw, _TOOL_KEYS, "tools")

    linker_prefix = _text(
        chemistry_raw.get("linker_prefix"), "chemistry.linker_prefix"
    )
    if _SAFE_IDENTIFIER.fullmatch(linker_prefix) is None:
        raise ScientificSpecError(
            "chemistry.linker_prefix must be a safe identifier using letters, "
            "digits, '.', '_', or '-'"
        )
    chemistry = RefinementChemistry(
        linker_prefix=linker_prefix,
        linker_smiles=_text(chemistry_raw.get("linker_smiles"), "chemistry.linker_smiles"),
        e3_ligand_smiles=_text(
            chemistry_raw.get("e3_ligand_smiles"), "chemistry.e3_ligand_smiles"
        ),
        linker_to_warhead_atom=_text(
            chemistry_raw.get("linker_to_warhead_atom"),
            "chemistry.linker_to_warhead_atom",
        ),
        linker_to_e3_ligand_atom=_text(
            chemistry_raw.get("linker_to_e3_ligand_atom"),
            "chemistry.linker_to_e3_ligand_atom",
        ),
    )
    if chemistry.linker_smiles.count("[*]") != 2:
        raise ScientificSpecError("chemistry.linker_smiles must contain exactly two [*] markers")
    if chemistry.e3_ligand_smiles.count("[*]") != 1:
        raise ScientificSpecError(
            "chemistry.e3_ligand_smiles must contain exactly one [*] marker"
        )

    chains = ChainMapping(
        poi=_text(chains_raw.get("poi"), "chains.poi"),
        warhead=_text(chains_raw.get("warhead"), "chains.warhead"),
        e3=_text(chains_raw.get("e3"), "chains.e3"),
        linker=_text(chains_raw.get("linker", "L"), "chains.linker"),
        e3_ligand=_text(chains_raw.get("e3_ligand", "X"), "chains.e3_ligand"),
    )
    invalid_chains = [
        name
        for name, value in vars(chains).items()
        if _PDB_CHAIN_ID.fullmatch(value) is None
    ]
    if invalid_chains:
        raise ScientificSpecError(
            "chains must use one alphanumeric PDB chain character; invalid: "
            + ", ".join(invalid_chains)
        )
    if len({chains.poi, chains.warhead, chains.e3, chains.linker, chains.e3_ligand}) != 5:
        raise ScientificSpecError("chains must assign distinct identifiers")

    anchors = AnchorMapping(
        warhead=_text(anchors_raw.get("warhead"), "anchors.warhead"),
        e3_ligand=_text(anchors_raw.get("e3_ligand"), "anchors.e3_ligand"),
    )
    if _anchor_chain(anchors.warhead, "anchors.warhead") != chains.warhead:
        raise ScientificSpecError("warhead anchor chain must match chains.warhead")
    if _anchor_chain(anchors.e3_ligand, "anchors.e3_ligand") != chains.e3_ligand:
        raise ScientificSpecError("E3 ligand anchor chain must match chains.e3_ligand")

    conformer_ranking = _text(
        protocol_raw.get("conformer_ranking", CONFORMER_RANKING),
        "protocol.conformer_ranking",
    )
    if conformer_ranking not in CONFORMER_RANKINGS:
        raise ScientificSpecError(
            "protocol.conformer_ranking must be attachment-first or clash-first"
        )
    protocol = RefinementProtocol(
        relax_cycles=_positive_int(
            protocol_raw.get("relax_cycles"), "protocol.relax_cycles", 5
        ),
        minimize_steps=_positive_int(
            protocol_raw.get("minimize_steps"), "protocol.minimize_steps", 1000
        ),
        minimize_tolerance=_positive_float(
            protocol_raw.get("minimize_tolerance"),
            "protocol.minimize_tolerance",
            0.001,
        ),
        random_seed=_nonnegative_int(
            protocol_raw.get("random_seed"), "protocol.random_seed", 42
        ),
        endpoint_candidates=_positive_int(
            protocol_raw.get("endpoint_candidates"),
            "protocol.endpoint_candidates",
            ATTACHMENT_ENDPOINT_CANDIDATES,
        ),
        conformers_per_endpoint=_positive_int(
            protocol_raw.get("conformers_per_endpoint"),
            "protocol.conformers_per_endpoint",
            PREPARE_CONFORMER_CANDIDATES,
        ),
        minimum_attachment_angle_degrees=_angle(
            protocol_raw.get("minimum_attachment_angle_degrees"),
            "protocol.minimum_attachment_angle_degrees",
            MIN_ATTACHMENT_ANGLE_DEGREES,
        ),
        conformer_ranking=conformer_ranking,
    )
    tools = RefinementTools(
        molfile_to_params=(
            _path(tools_raw["molfile_to_params"], config_dir, "tools.molfile_to_params")
            if "molfile_to_params" in tools_raw
            else None
        ),
        rosetta_root=(
            _path(tools_raw["rosetta_root"], config_dir, "tools.rosetta_root")
            if "rosetta_root" in tools_raw
            else None
        ),
    )

    input_pdb = _path(raw.get("input"), config_dir, "input")
    output_dir = _path(raw.get("output_dir"), config_dir, "output_dir")
    if input_pdb == output_dir or output_dir in input_pdb.parents:
        raise ScientificSpecError(
            "input must be outside refinement output_dir so source and generated "
            "Artifacts cannot be mixed"
        )
    result_summary = (
        _path(raw["result_summary"], config_dir, "result_summary")
        if "result_summary" in raw
        else None
    )
    if result_summary is not None and not result_summary.is_relative_to(output_dir):
        raise ScientificSpecError("result_summary must be inside refinement output_dir")

    return RefinementSpec(
        input_pdb=input_pdb,
        output_dir=output_dir,
        result_summary=result_summary,
        chemistry=chemistry,
        chains=chains,
        anchors=anchors,
        protocol=protocol,
        tools=tools,
    )


def compile_refinement_command(specification: RefinementSpec) -> Tuple[str, ...]:
    """Compile one scientific specification to the existing CLI adapter."""

    command = [
        sys.executable,
        "-m",
        "oligoternary.cli.refine",
        "--mode",
        "single",
        "--linker-prefix",
        specification.chemistry.linker_prefix,
        "--linker-smiles",
        specification.chemistry.linker_smiles,
        "--linker-to-warhead-atom",
        specification.chemistry.linker_to_warhead_atom,
        "--linker-to-e3l-atom",
        specification.chemistry.linker_to_e3_ligand_atom,
        "--e3l-smiles",
        specification.chemistry.e3_ligand_smiles,
        "--warhead-chain",
        specification.chains.warhead,
        "--e3l-chain",
        specification.chains.e3_ligand,
        "--e3-chain",
        specification.chains.e3,
        "--poi-chain",
        specification.chains.poi,
        "--linker-chain",
        specification.chains.linker,
        "--warhead-anchor-label",
        specification.anchors.warhead,
        "--e3l-anchor-label",
        specification.anchors.e3_ligand,
        "--pdb-file",
        str(specification.input_pdb),
        "--output-dir",
        str(specification.output_dir),
        "--relax-cycles",
        str(specification.protocol.relax_cycles),
        "--minimize-steps",
        str(specification.protocol.minimize_steps),
        "--minimize-tolerance",
        str(specification.protocol.minimize_tolerance),
        "--seed",
        str(specification.protocol.random_seed),
        "--endpoint-candidates",
        str(specification.protocol.endpoint_candidates),
        "--conformers-per-endpoint",
        str(specification.protocol.conformers_per_endpoint),
        "--minimum-attachment-angle-degrees",
        str(specification.protocol.minimum_attachment_angle_degrees),
        "--conformer-ranking",
        specification.protocol.conformer_ranking,
    ]
    if specification.tools.molfile_to_params is not None:
        command.extend(["--molfile2params", str(specification.tools.molfile_to_params)])
    if specification.tools.rosetta_root is not None:
        command.extend(["--rosetta-root", str(specification.tools.rosetta_root)])
    if specification.result_summary is not None:
        command.extend(["--result-summary", str(specification.result_summary)])
        for role, path in (
            ("refined_structure", specification.artifact),
            ("linker_params", specification.linker_params),
            ("e3_ligand_params", specification.e3_ligand_params),
        ):
            command.extend(["--result-artifact", f"{role}={path}"])
    return tuple(command)
