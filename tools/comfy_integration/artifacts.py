"""Own durable artifacts for one managed Comfy integration execution."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from tools.comfy_api import ImageReference, JsonObject


class IntegrationArtifacts:
    """Persist one run's state, logs, history, and downloaded output."""

    def __init__(self, root: Path) -> None:
        """Create one collision-resistant external run directory."""

        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        self.run_id = run_id
        self.root = root.resolve() / run_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.stdout_path = self.root / "comfy.stdout.log"
        self.stderr_path = self.root / "comfy.stderr.log"
        self._state: JsonObject = {"run_id": run_id, "status": "starting"}
        self._persist()

    def record_started(
        self,
        *,
        command: tuple[str, ...],
        environment: JsonObject,
        required_node_ids: frozenset[str],
        port: int,
        pid: int,
    ) -> None:
        """Record the exact created process identity and command."""

        self._state.update(
            {
                "status": "running",
                "command": list(command),
                "environment": environment,
                "required_node_ids": sorted(required_node_ids),
                "port": port,
                "parent_pid": pid,
                "started_at_utc": _utc_now(),
            }
        )
        self._persist()

    def record_success(
        self,
        *,
        system_stats: JsonObject,
        workflow: dict[str, JsonObject],
        history: JsonObject,
        prompt_id: str,
        image_reference: ImageReference,
        image_bytes: bytes,
    ) -> None:
        """Persist successful HTTP and image evidence before cleanup."""

        image_path = self.root / "baseline.png"
        image_path.write_bytes(image_bytes)
        (self.root / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._state.update(
            {
                "status": "workflow_completed",
                "system_stats": system_stats,
                "workflow": workflow,
                "prompt_id": prompt_id,
                "image_reference": {
                    "filename": image_reference.filename,
                    "subfolder": image_reference.subfolder,
                    "type": image_reference.output_type,
                },
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_size_bytes": len(image_bytes),
            }
        )
        self._persist()

    def record_cleanup(self, *, process_running: bool, port_available: bool) -> None:
        """Finalize only after the owned parent and listener are gone."""

        if process_running:
            raise RuntimeError("Managed Comfy parent remains alive after cleanup.")
        if not port_available:
            raise RuntimeError(
                "Managed Comfy loopback port remains occupied after cleanup."
            )
        self._state.update(
            {
                "status": "completed",
                "cleanup": {"parent_running": False, "port_available": True},
                "completed_at_utc": _utc_now(),
            }
        )
        self._persist()

    def record_failure(self, error: BaseException) -> None:
        """Persist an actionable failure without hiding prior evidence."""

        self._state.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "completed_at_utc": _utc_now(),
            }
        )
        self._persist()

    def _persist(self) -> None:
        """Atomically replace the authoritative run record."""

        path = self.root / "run.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)


def _utc_now() -> str:
    """Return a durable UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
