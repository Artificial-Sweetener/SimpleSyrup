"""Persist complete labeled P9.1 managed evidence and sidecars."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import cast

from PIL import Image

from tools.anima_attention_coupling_workflow import (
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.comfy_api import ImageReference, JsonObject
from tools.prompt_control_characterization.source_identity import (
    PromptControlSourceIdentity,
)

from .baseline import PromptControlBaseline
from .history import PromptControlAttentionOutputs
from .matrix import HEIGHT, WIDTH, PromptControlAttentionCase
from .validation import validate_case_evidence


class PromptControlAttentionResultRecorder:
    """Own P9.1 validation, journaling, artifacts, and finalization."""

    def __init__(
        self,
        root: Path,
        baseline: PromptControlBaseline,
        source: PromptControlSourceIdentity,
    ) -> None:
        """Retain one managed run and its exact external evidence identities."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("P9.1 result root must already exist.")
        self._baseline = baseline
        self._source = source
        self._started_at = _utc_now()
        self._observations: list[JsonObject] = []
        self._journal = self._root / "p9.1-result.inprogress.json"
        self._result = self._root / "p9.1-result.json"

    def record_metadata(self, metadata: JsonObject) -> Path:
        """Persist exact validated live public-node metadata."""

        path = self._root / "public-node-metadata.json"
        self._write_json(path, metadata)
        return path

    def record_masks(
        self,
        names: tuple[str, ...],
        *,
        input_root: Path,
    ) -> JsonObject:
        """Copy submitted masks and retain exact content digests."""

        destination = self._root / "masks"
        destination.mkdir(exist_ok=False)
        entries: list[object] = []
        for name in names:
            source = (input_root / name).resolve()
            target = destination / name
            shutil.copyfile(source, target)
            data = target.read_bytes()
            entries.append(
                {
                    "file": str(Path("masks") / name),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        evidence: JsonObject = {"vertical-hard-50-50": entries}
        self._write_json(self._root / "mask-evidence.json", evidence)
        return evidence

    def record_case(
        self,
        case: PromptControlAttentionCase,
        workflow: BuiltAnimaAttentionCouplingWorkflow,
        outputs: PromptControlAttentionOutputs,
        *,
        history: JsonObject,
        prompt_id: str,
        image_reference: ImageReference,
        image_bytes: bytes,
        wall_runtime_ms: float,
    ) -> Path:
        """Validate and durably journal one labeled image plus exact sidecars."""

        if any(item.get("case_id") == case.case_id for item in self._observations):
            raise ValueError(f"P9.1 case was already recorded: {case.case_id}.")
        if wall_runtime_ms <= 0.0:
            raise ValueError("P9.1 wall runtime must be positive.")
        self._validate_image(image_bytes)
        accepted = validate_case_evidence(
            case,
            workflow,
            outputs,
            self._baseline.observation(case.case_id),
        )
        image_path = self._root / f"{case.case_id}.png"
        history_path = self._root / f"{case.case_id}.history.json"
        workflow_path = self._root / f"{case.case_id}.workflow.json"
        image_path.write_bytes(image_bytes)
        self._write_json(history_path, history)
        self._write_json(workflow_path, workflow.prompt)
        self._observations.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "characterization": asdict(case.characterization),
                "cfg": case.cfg,
                "region_mask_feather": case.feather,
                "prompt_id": prompt_id,
                "metrics": outputs.metrics,
                "wall_runtime_ms": wall_runtime_ms,
                "diagnostic_record_count": accepted.diagnostic_record_count,
                "sampling_sigmas": list(accepted.sampling_sigmas),
                "conditioning_uuid_count": accepted.conditioning_uuid_count,
                "adapter_tokens": list(accepted.adapter_tokens),
                "image_file": image_path.name,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_size_bytes": len(image_bytes),
                "image_reference": {
                    "filename": image_reference.filename,
                    "subfolder": image_reference.subfolder,
                    "type": image_reference.output_type,
                },
                "workflow_file": workflow_path.name,
                "history_file": history_path.name,
            }
        )
        self._persist(self._journal, status="running", completed_at=None)
        return image_path

    def finalize(
        self,
        cases: tuple[PromptControlAttentionCase, ...],
        *,
        system_stats: JsonObject,
        cleanup_verified: bool,
    ) -> Path:
        """Write the terminal result only after matrix and cleanup completion."""

        if tuple(item.get("case_id") for item in self._observations) != tuple(
            case.case_id for case in cases
        ):
            raise ValueError("P9.1 result matrix is incomplete or out of order.")
        if not cleanup_verified:
            raise ValueError("P9.1 managed process, port, or mask cleanup failed.")
        self._persist(
            self._result,
            status="completed",
            completed_at=_utc_now(),
            system_stats=system_stats,
        )
        self._journal.unlink(missing_ok=True)
        return self._result

    def _persist(
        self,
        path: Path,
        *,
        status: str,
        completed_at: str | None,
        system_stats: JsonObject | None = None,
    ) -> None:
        """Atomically persist current evidence in authoritative matrix order."""

        payload = {
            "schema_version": 1,
            "phase": "p9.1",
            "status": status,
            "started_at_utc": self._started_at,
            "completed_at_utc": completed_at,
            "prompt_control_source": asdict(self._source),
            "p0_8_baseline": {
                "sha256": self._baseline.sha256,
                "observation_count": len(self._baseline.observations),
            },
            "system_stats": {} if system_stats is None else system_stats,
            "observations": self._observations,
        }
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _validate_image(image_bytes: bytes) -> None:
        """Require one non-flat exact-size decoded PNG artifact."""

        if not image_bytes:
            raise ValueError("P9.1 image must not be empty.")
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format != "PNG" or image.size != (WIDTH, HEIGHT):
                raise ValueError(f"P9.1 image must be a {WIDTH}x{HEIGHT} PNG.")
            extrema = cast(
                tuple[tuple[int, int], ...], image.convert("RGB").getextrema()
            )
        if not any(high > low for low, high in extrema):
            raise ValueError("P9.1 image must contain non-flat visual output.")

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        """Write one stable UTF-8 JSON sidecar."""

        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _utc_now() -> str:
    """Return one durable UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
