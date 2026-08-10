"""Locate and check Rosetta's ``molfile_to_params.py`` script."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ToolConfig:
    molfile_to_params: Optional[str] = None
    rosetta_root: Optional[str] = None


@dataclass(frozen=True)
class ToolResolution:
    name: str
    resolved_path: Optional[str]
    source: str
    exists: bool
    executable: bool
    ready: bool
    version_probe: Optional[str] = None
    diagnostic: str = ""


_SCRIPT_PATHS = (
    "main/source/scripts/python/public/molfile_to_params.py",
    "source/scripts/python/public/molfile_to_params.py",
    "scripts/python/public/molfile_to_params.py",
)


def _report(path: Optional[str], source: str) -> ToolResolution:
    exists = bool(path and Path(path).is_file())
    readable = bool(exists and os.access(path, os.R_OK))
    return ToolResolution(
        name="molfile_to_params",
        resolved_path=path,
        source=source,
        exists=exists,
        executable=bool(exists and os.access(path, os.X_OK)),
        ready=readable,
        diagnostic="ready" if readable else f"script not found or unreadable: {path or '-'}",
    )


def resolve_tools(
    config: Optional[ToolConfig] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    search_path: Optional[str] = None,
) -> dict[str, ToolResolution]:
    """Resolve ``molfile_to_params.py`` from option, environment, root, or PATH."""

    config = config or ToolConfig()
    env = os.environ if environ is None else environ
    path_env = env.get("PATH") if search_path is None else search_path

    if config.molfile_to_params:
        value = os.path.expanduser(os.path.expandvars(config.molfile_to_params))
        path = shutil.which(value, path=path_env) if os.sep not in value else os.path.abspath(value)
        report = _report(path or os.path.abspath(value), "explicit")
    else:
        env_script = next(
            (env[key] for key in ("MOLFILE_TO_PARAMS", "ROSETTA_MOLFILE_TO_PARAMS") if env.get(key)),
            None,
        )
        if env_script:
            report = _report(os.path.abspath(os.path.expanduser(env_script)), "environment")
        else:
            root = config.rosetta_root or next(
                (env[key] for key in ("ROSETTA_ROOT", "ROSETTA_HOME", "ROSETTA3") if env.get(key)),
                None,
            )
            if root:
                candidates = [Path(root).expanduser().resolve() / relative for relative in _SCRIPT_PATHS]
                selected = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
                report = _report(str(selected), "rosetta-root")
            else:
                report = _report(shutil.which("molfile_to_params.py", path=path_env), "PATH")
    return {"molfile_to_params": report}


def require_tools(
    resolutions: Mapping[str, ToolResolution], names: Sequence[str]
) -> dict[str, str]:
    failed = [resolutions[name] for name in names if not resolutions[name].ready]
    if failed:
        detail = "; ".join(f"{item.name}: {item.diagnostic}" for item in failed)
        raise FileNotFoundError(f"External-tool check failed: {detail}")
    return {name: str(resolutions[name].resolved_path) for name in names}


def preflight_tools(
    config: Optional[ToolConfig] = None,
    *,
    names: Sequence[str] = ("molfile_to_params",),
    environ: Optional[Mapping[str, str]] = None,
    search_path: Optional[str] = None,
    probe: bool = True,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[ToolResolution, ...]:
    reports = tuple(
        resolve_tools(config, environ=environ, search_path=search_path)[name] for name in names
    )
    if not probe:
        return reports
    checked = []
    for report in reports:
        if not report.ready:
            checked.append(report)
            continue
        try:
            result = runner(
                (sys.executable, str(report.resolved_path), "--help"),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            output = (result.stdout or result.stderr or "").strip().splitlines()
            checked.append(
                replace(
                    report,
                    ready=result.returncode == 0,
                    version_probe=f"exit={result.returncode}: {output[0][:240] if output else 'no output'}",
                    diagnostic="ready" if result.returncode == 0 else f"probe exited {result.returncode}",
                )
            )
        except (OSError, subprocess.SubprocessError) as exc:
            checked.append(replace(report, ready=False, diagnostic=f"probe failed: {exc}"))
    return tuple(checked)
