"""Validate and persist complete P9.4 managed evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from tools.anima_attention_coupling_workflow import (
    SAMPLER,
    SCHEDULER,
    SEED,
    STEPS,
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.comfy_api import ImageReference, JsonObject

from .fixture import TextEncoderLoraFixtureIdentity
from .history import extract_text_encoder_lora_history
from .image_validation import TEXT_ENCODER_LORA_IMAGE_VALIDATOR
from .matrix import TextEncoderLoraCase


class TextEncoderLoraResultRecorder:
    """Own ordered artifacts, acceptance, and terminal P9.4 publication."""

    def __init__(self, root: Path) -> None:
        """Retain one already-created collision-resistant artifact root."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("P9.4 result root must already exist.")
        self._observations: list[JsonObject] = []
        self._image_paths: dict[str, Path] = {}
        self._public_metadata: JsonObject = {}
        self._fixture_identity: JsonObject | None = None

    def record_fixture(self, identity: TextEncoderLoraFixtureIdentity) -> None:
        """Retain one already-validated external fixture identity."""

        if not isinstance(identity, TextEncoderLoraFixtureIdentity):
            raise TypeError("P9.4 result fixture requires a fixture identity.")
        if self._fixture_identity is not None:
            raise ValueError("P9.4 text-encoder LoRA fixture was already recorded.")
        self._fixture_identity = identity.as_record()

    def record_metadata(self, node_id: str, metadata: JsonObject) -> None:
        """Retain validated live public-node metadata by exact node ID."""

        if not node_id or node_id in self._public_metadata:
            raise ValueError(
                "P9.4 public-node metadata identity is invalid or repeated."
            )
        self._public_metadata[node_id] = metadata
        self._write_json(
            self._root / "public-node-metadata.json", self._public_metadata
        )

    def record_case(
        self,
        case: TextEncoderLoraCase,
        workflow: BuiltAnimaAttentionCouplingWorkflow,
        *,
        prompt_id: str,
        history: JsonObject,
        reference: ImageReference,
        image_bytes: bytes,
        wall_runtime_ms: float,
    ) -> Path:
        """Validate one history and persist its labeled original image."""

        if any(item.get("case_id") == case.case_id for item in self._observations):
            raise ValueError(f"P9.4 case was already recorded: {case.case_id}")
        if wall_runtime_ms <= 0:
            raise ValueError("P9.4 wall runtime must be positive.")
        evidence = extract_text_encoder_lora_history(history, workflow)
        self._validate_adapter_uses(case, evidence.diagnostics)
        image_path = self._root / f"{case.case_id}.png"
        image_path.write_bytes(image_bytes)
        self._image_paths[case.case_id] = image_path
        workflow_path = self._root / f"{case.case_id}.workflow.json"
        history_path = self._root / f"{case.case_id}.history.json"
        self._write_json(workflow_path, workflow.prompt)
        self._write_json(history_path, history)
        self._observations.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "spatial_mode": case.spatial_mode.value,
                "global_text_lora": case.global_text_lora,
                "regional_text_lora": case.regional_text_lora,
                "regional_model_lora": case.regional_model_lora,
                "comparison_case_id": case.comparison_case_id,
                "prompt_id": prompt_id,
                "metrics": evidence.metrics,
                "conditioning_batch_snapshot": evidence.conditioning_batch_snapshot,
                "diagnostic_record_count": evidence.diagnostics["record_count"],
                "wall_runtime_ms": wall_runtime_ms,
                "image_file": image_path.name,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_size_bytes": len(image_bytes),
                "image_reference": {
                    "filename": reference.filename,
                    "subfolder": reference.subfolder,
                    "type": reference.output_type,
                },
                "workflow_file": workflow_path.name,
                "history_file": history_path.name,
            }
        )
        self._persist("in_progress", {})
        return image_path

    def finalize(
        self,
        definitions: tuple[TextEncoderLoraCase, ...],
        *,
        system_stats: JsonObject,
        cleanup_verified: bool,
        masks_removed: bool,
    ) -> Path:
        """Publish completion after calls, pixels, metadata, and cleanup pass."""

        expected = tuple(case.case_id for case in definitions)
        observed = tuple(str(item["case_id"]) for item in self._observations)
        if observed != expected:
            raise ValueError("P9.4 observations are incomplete or out of order.")
        if len(self._public_metadata) != 3:
            raise ValueError("P9.4 requires metadata for all three public samplers.")
        if self._fixture_identity is None:
            raise ValueError(
                "P9.4 requires the exact text-encoder LoRA fixture identity."
            )
        if not cleanup_verified or not masks_removed:
            raise ValueError("P9.4 managed process, port, or mask cleanup failed.")
        self._validate_call_counts(definitions)
        comparisons = TEXT_ENCODER_LORA_IMAGE_VALIDATOR.validate(
            definitions,
            self._image_paths,
        )
        return self._persist(
            "completed",
            {
                "image_comparisons": comparisons,
                "system_stats": system_stats,
                "cleanup": {
                    "process_stopped": True,
                    "port_available": True,
                    "masks_removed": True,
                },
            },
        )

    @staticmethod
    def _validate_adapter_uses(
        case: TextEncoderLoraCase,
        diagnostics: JsonObject,
    ) -> None:
        """Require text-only hooks to stay out of diffusion diagnostics."""

        snapshots = cast(list[JsonObject], diagnostics["snapshots"])
        uses = [
            use
            for snapshot in snapshots
            for use in cast(list[object], snapshot.get("adapter_uses", []))
            if isinstance(use, dict)
        ]
        conditioning_entries = [
            snapshot.get("regional_conditioning_entries") for snapshot in snapshots
        ]
        if any(
            not isinstance(entries, list) or len(entries) != 2
            for entries in conditioning_entries
        ):
            raise ValueError("P9.4 diagnostics must retain both regional contexts.")
        if not case.regional_model_lora:
            if uses:
                raise ValueError(
                    f"P9.4 text-only case {case.case_id!r} entered model execution."
                )
            return
        if not uses:
            raise ValueError(
                f"P9.4 model-LoRA case {case.case_id!r} has no adapter uses."
            )
        if any(use.get("target_count") != 448 for use in uses):
            raise ValueError("P9.4 ADAPTER_A model execution must retain all 448 targets.")

    def _validate_call_counts(
        self,
        definitions: tuple[TextEncoderLoraCase, ...],
    ) -> None:
        """Require every LoRA comparison to preserve complete denoiser calls."""

        observations = {str(item["case_id"]): item for item in self._observations}
        for case in definitions:
            if case.comparison_case_id is None:
                continue
            if self._model_calls(observations[case.case_id]) != self._model_calls(
                observations[case.comparison_case_id]
            ):
                raise ValueError(
                    f"P9.4 case {case.case_id!r} changed complete model-call count."
                )

    @staticmethod
    def _model_calls(observation: JsonObject) -> int:
        """Return one narrowed measured complete model-call count."""

        metrics = observation.get("metrics")
        if not isinstance(metrics, dict):
            raise TypeError("P9.4 observation metrics must be an object.")
        value = metrics.get("model_call_count")
        if not isinstance(value, int):
            raise TypeError("P9.4 observation model-call count must be an integer.")
        return value

    def _persist(self, status: str, extra: JsonObject) -> Path:
        """Atomically replace the current machine-readable result."""

        path = self._root / "p9.4-result.json"
        payload: JsonObject = {
            "schema_version": 1,
            "phase": "p9.4",
            "status": status,
            "sampling": {
                "seed": SEED,
                "steps": STEPS,
                "sampler": SAMPLER,
                "scheduler": SCHEDULER,
                "full_dimensions": [1024, 1024],
                "upscale_dimensions": [1024, 1024, 1536, 1536],
            },
            "observations": self._observations,
            "text_encoder_lora_fixture": self._fixture_identity,
            "public_node_metadata_file": "public-node-metadata.json",
            **extra,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        """Write one deterministic human-reviewable sidecar."""

        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
