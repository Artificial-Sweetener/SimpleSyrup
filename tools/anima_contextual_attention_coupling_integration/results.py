# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate and persist complete labeled P7.8 managed-run evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import cast

from PIL import Image

from tools.anima_attention_coupling_workflow import (
    SAMPLER,
    SCHEDULER,
    SEED,
    STEPS,
    BuiltAnimaAttentionCouplingWorkflow,
)
from tools.comfy_api import ImageReference, JsonObject

from .diagnostics import ContextualDiagnosticsValidator
from .matrix import (
    CONTEXT_BATCH_SIZE,
    CONTEXT_OVERLAP,
    CONTEXT_SIZE,
    GLOBAL_DECAY,
    GLOBAL_WEIGHT,
    REFINEMENT_DENOISE,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    SUBTOKEN_MASK_CASE_ID,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    ContextualIntegrationCase,
)


class ContextualIntegrationResultRecorder:
    """Own P7.8 case association, validation, and evidence persistence."""

    def __init__(self, root: Path) -> None:
        """Retain one existing collision-resistant managed-run directory."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("P7.8 result root must already exist.")
        self._observations: list[JsonObject] = []

    def record_metadata(self, metadata: JsonObject) -> Path:
        """Persist the exact validated live node metadata response."""

        path = self._root / "public-node-metadata.json"
        self._write_json(path, metadata)
        return path

    def record_masks(
        self,
        names_by_case: dict[str, tuple[str, ...]],
        *,
        input_root: Path,
    ) -> JsonObject:
        """Copy every submitted authored mask and retain its digest."""

        destination = self._root / "masks"
        destination.mkdir(exist_ok=False)
        evidence: JsonObject = {}
        for case_id, names in names_by_case.items():
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
            evidence[case_id] = entries
        self._write_json(self._root / "mask-evidence.json", evidence)
        return evidence

    def record_case(
        self,
        case: ContextualIntegrationCase,
        workflow: BuiltAnimaAttentionCouplingWorkflow,
        *,
        history: JsonObject,
        prompt_id: str,
        image_reference: ImageReference,
        image_bytes: bytes,
        source_image_reference: ImageReference,
        source_image_bytes: bytes,
        wall_runtime_ms: float,
    ) -> Path:
        """Validate metrics and persist one labeled image with exact sidecars."""

        if any(item.get("case_id") == case.case_id for item in self._observations):
            raise ValueError(f"P7.8 case was already recorded: {case.case_id}")
        if wall_runtime_ms <= 0:
            raise ValueError("P7.8 wall runtime must be positive.")
        history_path = self._root / f"{case.case_id}.history.json"
        workflow_path = self._root / f"{case.case_id}.workflow.json"
        self._write_json(history_path, history)
        self._write_json(workflow_path, workflow.prompt)
        dimensions = self._image_dimensions(image_bytes)
        if dimensions != (TARGET_WIDTH, TARGET_HEIGHT):
            raise ValueError(
                "P7.8 refinement image must be "
                f"{TARGET_WIDTH}x{TARGET_HEIGHT}; received {dimensions}."
            )
        source_dimensions = self._image_dimensions(source_image_bytes)
        if source_dimensions != (SOURCE_WIDTH, SOURCE_HEIGHT):
            raise ValueError(
                f"P7.8 source image must be {SOURCE_WIDTH}x{SOURCE_HEIGHT}; "
                f"received {source_dimensions}."
            )
        metrics = self._metrics(history, workflow)
        diagnostics = ContextualDiagnosticsValidator.validate(history, workflow, case)
        image_path = self._root / f"{case.case_id}.png"
        source_image_path = self._root / f"{case.case_id}.source.png"
        image_path.write_bytes(image_bytes)
        source_image_path.write_bytes(source_image_bytes)
        self._observations.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "diffusion_mode": case.diffusion_mode,
                "branch_mode": case.branch_mode,
                "mask_case_id": case.mask_case_id,
                "uses_segs": case.use_segs,
                "global_steps": case.global_steps,
                "cfg": case.cfg,
                "region_mask_feather": case.feather,
                "declared_regional_adapter_executions": case.regional_lora_count,
                "global_loras": [asdict(adapter) for adapter in case.global_loras],
                "regional_loras": [asdict(adapter) for adapter in case.regional_loras],
                "subtoken_authored_width_pixels": (
                    1 if case.mask_case_id == SUBTOKEN_MASK_CASE_ID else None
                ),
                "prompt_id": prompt_id,
                "metrics": metrics,
                "regional_diagnostics": diagnostics,
                "wall_runtime_ms": wall_runtime_ms,
                "image_file": image_path.name,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_size_bytes": len(image_bytes),
                "image_width": dimensions[0],
                "image_height": dimensions[1],
                "image_reference": {
                    "filename": image_reference.filename,
                    "subfolder": image_reference.subfolder,
                    "type": image_reference.output_type,
                },
                "source_image_file": source_image_path.name,
                "source_image_sha256": hashlib.sha256(source_image_bytes).hexdigest(),
                "source_image_size_bytes": len(source_image_bytes),
                "source_image_width": source_dimensions[0],
                "source_image_height": source_dimensions[1],
                "source_image_reference": {
                    "filename": source_image_reference.filename,
                    "subfolder": source_image_reference.subfolder,
                    "type": source_image_reference.output_type,
                },
                "workflow_file": workflow_path.name,
                "history_file": history_path.name,
            }
        )
        self._persist(status="in_progress")
        return image_path

    def finalize(
        self,
        definitions: tuple[ContextualIntegrationCase, ...],
        *,
        system_stats: JsonObject,
        cleanup_verified: bool,
    ) -> Path:
        """Write success only after ordered coverage, branch proof, and cleanup."""

        expected = tuple(case.case_id for case in definitions)
        observed = tuple(str(item["case_id"]) for item in self._observations)
        if observed != expected:
            raise ValueError(f"P7.8 observations must match case order: {observed!r}.")
        if not cleanup_verified:
            raise ValueError("P7.8 result requires managed-server cleanup proof.")
        self._validate_branch_call_counts(definitions)
        return self._persist(
            status="completed",
            extra={
                "system_stats": system_stats,
                "logs": self._record_logs(),
                "cleanup_verified": True,
            },
        )

    def _validate_branch_call_counts(
        self,
        definitions: tuple[ContextualIntegrationCase, ...],
    ) -> None:
        """Prove reduced-global cases add work over matched local-only cases."""

        selected = {case.case_id for case in definitions}
        observations = {str(item["case_id"]): item for item in self._observations}
        for mode in ("multidiffusion", "mixture_of_diffusers"):
            local_id = f"{mode}-local-segs"
            global_id = f"{mode}-reduced-global-segs"
            if {local_id, global_id} <= selected:
                local_calls = self._model_calls(observations[local_id])
                global_calls = self._model_calls(observations[global_id])
                if global_calls <= local_calls:
                    raise ValueError(
                        "P7.8 reduced-global execution must add model calls over "
                        f"the matched local-only {mode} case."
                    )

    @staticmethod
    def _model_calls(observation: JsonObject) -> int:
        """Return one narrowed measured model-call count."""

        metrics = observation.get("metrics")
        if not isinstance(metrics, dict):
            raise TypeError("P7.8 observation metrics must be an object.")
        value = metrics.get("model_call_count")
        if not isinstance(value, int):
            raise TypeError("P7.8 model-call count must be an integer.")
        return value

    @staticmethod
    def _metrics(
        history: JsonObject,
        workflow: BuiltAnimaAttentionCouplingWorkflow,
    ) -> JsonObject:
        """Extract and validate the benchmark node's terminal metrics payload."""

        outputs = history.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError("P7.8 history is missing outputs.")
        output = outputs.get(workflow.metrics_node_id)
        if not isinstance(output, dict):
            raise ValueError("P7.8 history is missing metrics output.")
        items = output.get("benchmark_metrics")
        if (
            not isinstance(items, list)
            or len(items) != 1
            or not isinstance(items[0], dict)
        ):
            raise ValueError("P7.8 history must contain one metrics object.")
        metrics = cast(JsonObject, items[0])
        if metrics.get("run_id") != workflow.metrics_run_id:
            raise ValueError("P7.8 metrics identity does not match its workflow.")
        call_count = metrics.get("model_call_count")
        runtime = metrics.get("runtime_ms")
        peak = metrics.get("peak_vram_bytes")
        if not isinstance(call_count, int) or call_count < STEPS:
            raise ValueError("P7.8 model calls must cover every denoising step.")
        if not isinstance(runtime, int | float) or runtime <= 0:
            raise ValueError("P7.8 measured runtime must be positive.")
        if not isinstance(peak, int) or peak <= 0:
            raise ValueError("P7.8 peak VRAM must be positive.")
        return metrics

    def _record_logs(self) -> JsonObject:
        """Hash complete logs and extract a reviewable diagnostic slice."""

        evidence: JsonObject = {}
        diagnostics: list[str] = []
        for name in ("comfy.stdout.log", "comfy.stderr.log"):
            path = self._root / name
            data = path.read_bytes()
            evidence[name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
            text = data.decode("utf-8", errors="replace")
            diagnostics.extend(
                f"[{name}] {line}"
                for line in text.splitlines()
                if any(
                    token in line.lower()
                    for token in (
                        "attention",
                        "contextual",
                        "regional",
                        "lora",
                        "warning",
                        "error",
                    )
                )
            )
        diagnostic_path = self._root / "diagnostics.log"
        diagnostic_text = "\n".join(diagnostics) + ("\n" if diagnostics else "")
        diagnostic_path.write_text(diagnostic_text, encoding="utf-8")
        evidence["diagnostics.log"] = {
            "sha256": hashlib.sha256(diagnostic_text.encode()).hexdigest(),
            "line_count": len(diagnostics),
        }
        return evidence

    def _persist(self, *, status: str, extra: JsonObject | None = None) -> Path:
        """Atomically replace the authoritative P7.8 result record."""

        path = self._root / "p7.8-result.json"
        payload: JsonObject = {
            "schema_version": 1,
            "status": status,
            "sampling": {
                "seed": SEED,
                "width": TARGET_WIDTH,
                "height": TARGET_HEIGHT,
                "source_width": SOURCE_WIDTH,
                "source_height": SOURCE_HEIGHT,
                "refinement_denoise": REFINEMENT_DENOISE,
                "steps": STEPS,
                "sampler": SAMPLER,
                "scheduler": SCHEDULER,
                "latent_context_size": CONTEXT_SIZE,
                "latent_context_overlap": CONTEXT_OVERLAP,
                "latent_context_batch_size": CONTEXT_BATCH_SIZE,
                "global_weight": GLOBAL_WEIGHT,
                "global_decay": GLOBAL_DECAY,
            },
            "observations": self._observations,
        }
        if extra:
            payload.update(extra)
        temporary = path.with_suffix(".json.tmp")
        self._write_json(temporary, payload)
        temporary.replace(path)
        return path

    @staticmethod
    def _image_dimensions(image_bytes: bytes) -> tuple[int, int]:
        """Validate image bytes and return their decoded dimensions."""

        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
            return image.size

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        """Write stable human-reviewable JSON."""

        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
