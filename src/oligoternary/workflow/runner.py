"""Sequential workflow runner for modeling and analysis stages."""

from __future__ import annotations

import os
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Optional, Sequence

from oligoternary.workflow.tools import ToolConfig, preflight_tools

from .artifacts import ArtifactRecord, read_result_summary, record_artifact
from .config import StageConfig, WorkflowConfig
from .provenance import collect_provenance
from .state import RunManifest, StageState, utc_now, write_manifest


class WorkflowError(RuntimeError):
    """Raised when a validated workflow cannot be executed safely."""


def _record_artifacts(stage: StageConfig) -> Optional[list[ArtifactRecord]]:
    records = []
    for artifact in stage.adapter.artifacts:
        record = record_artifact(
            artifact.path,
            role=artifact.role,
            media_type=artifact.media_type,
        )
        if record is None:
            return None
        records.append(record)
    return records


class WorkflowRunner:
    """Orchestrate stages while keeping tool implementations behind adapters."""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.manifest_path = config.output_dir / "run_manifest.json"
        self.manifest_dir = config.output_dir / "run_manifests"
        self.lock_path = config.output_dir / ".run.lock"

    def _acquire_run_lock(self, manifest: RunManifest) -> tuple[str, int]:
        """Acquire exclusive ownership of this Workflow state directory."""

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        token = f"run_id={manifest.run_id}\npid={os.getpid()}\n"
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError as exc:
            try:
                owner = self.lock_path.read_text(encoding="utf-8")[:500].strip()
            except OSError:
                owner = "owner details unavailable"
            raise WorkflowError(
                f"workflow output is locked by another run ({self.lock_path}): {owner}; "
                "verify that run before removing a stale lock"
            ) from exc
        try:
            os.write(descriptor, token.encode("utf-8"))
            inode = os.fstat(descriptor).st_ino
        except BaseException:
            os.close(descriptor)
            try:
                self.lock_path.unlink()
            except OSError:
                pass
            raise
        os.close(descriptor)
        return token, inode

    def _release_run_lock(self, owner: tuple[str, int]) -> None:
        """Release only the exact lock file created by this runner."""

        token, inode = owner
        try:
            status = self.lock_path.lstat()
            contents = self.lock_path.read_text(encoding="utf-8")
        except OSError:
            return
        if self.lock_path.is_symlink() or status.st_ino != inode or contents != token:
            return
        try:
            self.lock_path.unlink()
        except OSError:
            return

    def _write_manifest(self, manifest: RunManifest) -> None:
        """Persist both the latest pointer and the immutable per-run identity."""

        archive_path = self.manifest_dir / f"{manifest.run_id}.json"
        manifest.updated_at = utc_now()
        write_manifest(archive_path, manifest, update_timestamp=False)
        write_manifest(self.manifest_path, manifest, update_timestamp=False)

    def _new_manifest(self, dry_run: bool) -> RunManifest:
        now = utc_now()
        stages = [
            StageState(
                name=stage.name,
                status="pending",
                adapter=stage.adapter.type,
                artifact=str(stage.adapter.artifact),
                depends_on=list(stage.depends_on),
                command=list(stage.adapter.command),
                cwd=str(stage.adapter.cwd) if stage.adapter.cwd else None,
            )
            for stage in self.config.stages
        ]
        return RunManifest(
            schema_version=1,
            project=self.config.project,
            config_path=str(self.config.config_path),
            output_dir=str(self.config.output_dir),
            overall_status="pending",
            started_at=now,
            updated_at=now,
            dry_run=dry_run,
            stages=stages,
            provenance=collect_provenance(
                self.config.config_path,
                tool_reports=self._external_tool_reports(),
            ),
            run_id=str(uuid.uuid4()),
        )

    def _external_tool_reports(self):
        reports = []
        seen = set()
        for stage in self.config.stages:
            refinement = stage.adapter.refinement
            if refinement is None:
                continue
            config = ToolConfig(
                molfile_to_params=(
                    str(refinement.tools.molfile_to_params)
                    if refinement.tools.molfile_to_params is not None
                    else None
                ),
                rosetta_root=(
                    str(refinement.tools.rosetta_root)
                    if refinement.tools.rosetta_root is not None
                    else None
                ),
            )
            for report in preflight_tools(
                config,
                names=("molfile_to_params",),
                probe=True,
            ):
                identity = (report.name, report.resolved_path, report.source)
                if identity not in seen:
                    reports.append(report)
                    seen.add(identity)
        return tuple(reports)

    @staticmethod
    def _state_map(manifest: RunManifest) -> Dict[str, StageState]:
        return {stage.name: stage for stage in manifest.stages}

    def _dependency_is_satisfied(
        self, dependency: str, states: Dict[str, StageState]
    ) -> bool:
        return states[dependency].status == "succeeded"

    def _log_paths(
        self, stage: StageConfig, manifest: RunManifest
    ) -> tuple[Path, Path]:
        log_dir = self.config.output_dir / "logs" / manifest.run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{stage.name}.stdout.log"
        stderr_path = log_dir / f"{stage.name}.stderr.log"
        return stdout_path, stderr_path

    @staticmethod
    def _modification_token(path: Path) -> Optional[tuple[tuple[object, ...], ...]]:
        """Return metadata that proves a command touched a declared path."""

        if path.is_symlink() or not path.exists():
            return None
        paths = [path]
        if path.is_dir():
            paths.extend(
                sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix())
            )
        token = []
        for item in paths:
            if item.is_symlink():
                return None
            try:
                status = item.stat()
            except OSError:
                return None
            relative = "." if item == path else item.relative_to(path).as_posix()
            token.append(
                (
                    relative,
                    status.st_mode,
                    status.st_ino,
                    status.st_size,
                    status.st_mtime_ns,
                    status.st_ctime_ns,
                )
            )
        return tuple(token)

    def _read_stage_summary(
        self,
        stage: StageConfig,
        records: Sequence[ArtifactRecord],
    ):
        assert stage.adapter.result_summary is not None
        summary = read_result_summary(stage.adapter.result_summary)
        if summary.stage != stage.name:
            raise ValueError(
                f"result summary {summary.path} stage does not match "
                f"declared stage {stage.name!r}"
            )
        if summary.run_specification != str(self.config.config_path):
            raise ValueError(
                f"result summary {summary.path} run specification path does not match"
            )
        expected_artifacts = {
            record.role: (record.path, record.media_type, record.size)
            for record in records
        }
        actual_artifacts = {
            artifact.role: (artifact.path, artifact.media_type, artifact.size)
            for artifact in summary.artifacts
        }
        if actual_artifacts != expected_artifacts:
            raise ValueError(
                f"result summary {summary.path} artifact record does not match "
                "the declared Stage artifacts"
            )
        return summary

    def _apply_result_summary(
        self,
        stage: StageConfig,
        state: StageState,
    ) -> bool:
        if stage.adapter.result_summary is None:
            return True
        try:
            summary = self._read_stage_summary(stage, state.artifacts)
        except ValueError as exc:
            state.status = "failed"
            state.message = str(exc)
            return False
        state.result_summary = summary
        if summary.failed_count:
            state.status = "failed"
            state.message = (
                f"stage result reported {summary.failed_count} failed "
                f"of {summary.total_count} inputs"
            )
            return False
        return True

    def _run_command_adapter(
        self, stage: StageConfig, state: StageState, manifest: RunManifest
    ) -> None:
        assert stage.adapter.cwd is not None
        before_artifacts = [
            self._modification_token(artifact.path)
            for artifact in stage.adapter.artifacts
        ]
        before_summary = (
            self._modification_token(stage.adapter.result_summary)
            if stage.adapter.result_summary is not None
            else None
        )
        stdout_path, stderr_path = self._log_paths(stage, manifest)
        state.stdout_log = str(stdout_path)
        state.stderr_log = str(stderr_path)
        environment = os.environ.copy()
        environment.update(
            {
                "OLIGOTERNARY_STAGE_NAME": stage.name,
                "OLIGOTERNARY_RUN_SPEC_PATH": str(self.config.config_path),
            }
        )
        process: Optional[subprocess.Popen] = None
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_handle:
                process = subprocess.Popen(
                    list(stage.adapter.command),
                    cwd=stage.adapter.cwd,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    shell=False,
                    env=environment,
                    start_new_session=os.name == "posix",
                )
                return_code = process.wait(timeout=stage.adapter.timeout_seconds)
        except subprocess.TimeoutExpired:
            assert process is not None
            self._terminate_process_group(process)
            state.return_code = 124
            state.status = "failed"
            state.message = (
                f"command timed out after {stage.adapter.timeout_seconds:g} seconds"
            )
            return
        except KeyboardInterrupt:
            if process is not None:
                self._terminate_process_group(process)
            raise
        except OSError as exc:
            state.status = "failed"
            state.message = f"command could not start: {exc}"
            return

        state.return_code = return_code
        if return_code != 0:
            state.status = "failed"
            state.message = f"command exited with code {return_code}"
        elif (records := _record_artifacts(stage)) is None:
            state.status = "failed"
            state.message = (
                "command exited successfully but the declared artifact is missing or empty"
            )
        else:
            state.artifacts = records
            after_artifacts = [
                self._modification_token(artifact.path)
                for artifact in stage.adapter.artifacts
            ]
            unchanged = [
                artifact.role
                for artifact, before, after in zip(
                    stage.adapter.artifacts, before_artifacts, after_artifacts
                )
                if before is not None and before == after
            ]
            if unchanged:
                state.status = "failed"
                state.message = (
                    "command exited successfully but declared artifact(s) were not updated: "
                    + ", ".join(unchanged)
                )
                return
            if stage.adapter.result_summary is not None:
                after_summary = self._modification_token(stage.adapter.result_summary)
                if before_summary is not None and before_summary == after_summary:
                    state.status = "failed"
                    state.message = (
                        "command exited successfully but the result summary was not updated"
                    )
                    return
            if self._apply_result_summary(stage, state):
                state.status = "succeeded"
                state.message = "command completed and declared artifacts are ready"

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen) -> None:
        """Terminate the command and descendants created in its process group."""

        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover - Windows scientific runtime is not in CI
                process.terminate()
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover - Windows scientific runtime is not in CI
                process.kill()
        except ProcessLookupError:
            return
        process.wait()

    def _run_existing_artifact_adapter(
        self, stage: StageConfig, state: StageState
    ) -> None:
        records = _record_artifacts(stage)
        if records is not None:
            state.artifacts = records
            if self._apply_result_summary(stage, state):
                state.status = "succeeded"
                state.message = "declared existing artifacts are ready"
        else:
            state.status = "failed"
            state.message = "declared existing artifact is missing, empty, or unsafe"

    def run(
        self,
        *,
        dry_run: bool = False,
    ) -> RunManifest:
        """Execute the workflow and persist state before and after every stage."""

        manifest = self._new_manifest(dry_run)
        if dry_run:
            for state in manifest.stages:
                state.status = "skipped"
                state.finished_at = utc_now()
                state.message = "dry-run: adapter was not executed"
            manifest.overall_status = "incomplete"
            manifest.updated_at = utc_now()
            return manifest
        owner = self._acquire_run_lock(manifest)
        try:
            return self._execute(manifest)
        finally:
            self._release_run_lock(owner)

    def _execute(self, manifest: RunManifest) -> RunManifest:
        states = self._state_map(manifest)
        manifest.overall_status = "running"
        self._write_manifest(manifest)

        for stage in self.config.stages:
            state = states[stage.name]
            unsatisfied = [
                dependency
                for dependency in stage.depends_on
                if not self._dependency_is_satisfied(dependency, states)
            ]
            if unsatisfied:
                state.status = "skipped"
                state.finished_at = utc_now()
                state.message = "unsatisfied dependencies: " + ", ".join(unsatisfied)
                self._write_manifest(manifest)
                continue

            state.status = "running"
            state.started_at = utc_now()
            self._write_manifest(manifest)
            try:
                if stage.adapter.type in {"command", "linker-refinement"}:
                    self._run_command_adapter(stage, state, manifest)
                elif stage.adapter.type == "existing-artifact":
                    self._run_existing_artifact_adapter(stage, state)
                else:  # validated configs make this unreachable
                    state.status = "failed"
                    state.message = f"unsupported adapter type: {stage.adapter.type}"
            except KeyboardInterrupt as exc:
                state.status = "failed"
                state.message = "command was interrupted"
                state.finished_at = utc_now()
                manifest.overall_status = "failed"
                self._write_manifest(manifest)
                raise WorkflowError("workflow command was interrupted") from exc
            state.finished_at = utc_now()
            self._write_manifest(manifest)

        statuses = {stage.status for stage in manifest.stages}
        if "failed" in statuses:
            manifest.overall_status = "failed"
        elif statuses == {"succeeded"}:
            manifest.overall_status = "succeeded"
        else:
            manifest.overall_status = "incomplete"
        self._write_manifest(manifest)
        return manifest
