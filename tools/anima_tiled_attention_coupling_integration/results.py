# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate and persist complete labeled P6.9 tiled-node evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
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

from .matrix import (
    REFINEMENT_DENOISE,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    TILE_HEIGHT,
    TILE_OVERLAP,
    TILE_WIDTH,
    TiledIntegrationCase,
)


class TiledIntegrationResultRecorder:
    """Own tiled case association, validation, and evidence persistence."""

    def __init__(self, root: Path) -> None:
        """Retain one existing collision-resistant managed-run directory."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("P6.9 result root must already exist.")
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
        case: TiledIntegrationCase,
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
            raise ValueError(f"P6.9 case was already recorded: {case.case_id}")
        metrics = self._metrics(history, workflow)
        if wall_runtime_ms <= 0:
            raise ValueError("P6.9 wall runtime must be positive.")
        image_path = self._root / f"{case.case_id}.png"
        source_image_path = self._root / f"{case.case_id}.source.png"
        history_path = self._root / f"{case.case_id}.history.json"
        workflow_path = self._root / f"{case.case_id}.workflow.json"
        image_path.write_bytes(image_bytes)
        source_image_path.write_bytes(source_image_bytes)
        self._write_json(history_path, history)
        self._write_json(workflow_path, workflow.prompt)
        self._observations.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "diffusion_mode": case.diffusion_mode,
                "tile_batch_size": case.tile_batch_size,
                "cfg": case.cfg,
                "mask_case_id": case.mask_case_id,
                "region_mask_feather": case.feather,
                "regional_lora_count": case.regional_lora_count,
                "global_loras": [asdict(adapter) for adapter in case.global_loras],
                "regional_loras": [asdict(adapter) for adapter in case.regional_loras],
                "prompt_id": prompt_id,
                "metrics": metrics,
                "wall_runtime_ms": wall_runtime_ms,
                "image_file": image_path.name,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "image_size_bytes": len(image_bytes),
                "image_reference": {
                    "filename": image_reference.filename,
                    "subfolder": image_reference.subfolder,
                    "type": image_reference.output_type,
                },
                "source_image_file": source_image_path.name,
                "source_image_sha256": hashlib.sha256(source_image_bytes).hexdigest(),
                "source_image_size_bytes": len(source_image_bytes),
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
        definitions: tuple[TiledIntegrationCase, ...],
        *,
        system_stats: JsonObject,
        cleanup_verified: bool,
    ) -> Path:
        """Write success only after complete ordered coverage and cleanup proof."""

        expected = tuple(case.case_id for case in definitions)
        observed = tuple(str(item["case_id"]) for item in self._observations)
        if observed != expected:
            raise ValueError(f"P6.9 observations must match case order: {observed!r}.")
        if not cleanup_verified:
            raise ValueError("P6.9 result requires managed-server cleanup proof.")
        return self._persist(
            status="completed",
            extra={
                "system_stats": system_stats,
                "logs": self._record_logs(),
                "cleanup_verified": True,
            },
        )

    @staticmethod
    def _metrics(
        history: JsonObject,
        workflow: BuiltAnimaAttentionCouplingWorkflow,
    ) -> JsonObject:
        """Extract and validate the benchmark node's terminal metrics payload."""

        outputs = history.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError("P6.9 history is missing outputs.")
        output = outputs.get(workflow.metrics_node_id)
        if not isinstance(output, dict):
            raise ValueError("P6.9 history is missing metrics output.")
        items = output.get("benchmark_metrics")
        if (
            not isinstance(items, list)
            or len(items) != 1
            or not isinstance(items[0], dict)
        ):
            raise ValueError("P6.9 history must contain one metrics object.")
        metrics = cast(JsonObject, items[0])
        if metrics.get("run_id") != workflow.metrics_run_id:
            raise ValueError("P6.9 metrics identity does not match its workflow.")
        call_count = metrics.get("model_call_count")
        runtime = metrics.get("runtime_ms")
        peak = metrics.get("peak_vram_bytes")
        if not isinstance(call_count, int) or call_count < STEPS:
            raise ValueError("P6.9 model-call count must cover every denoising step.")
        if not isinstance(runtime, int | float) or runtime <= 0:
            raise ValueError("P6.9 measured runtime must be positive.")
        if not isinstance(peak, int) or peak <= 0:
            raise ValueError("P6.9 peak VRAM must be positive.")
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
                    for token in ("attention", "regional", "lora", "warning", "error")
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
        """Atomically replace the authoritative P6.9 result record."""

        path = self._root / "p6.9-result.json"
        payload: JsonObject = {
            "schema_version": 1,
            "status": status,
            "sampling": {
                "seed": SEED,
                "width": TARGET_WIDTH,
                "height": TARGET_HEIGHT,
                "steps": STEPS,
                "sampler": SAMPLER,
                "scheduler": SCHEDULER,
                "latent_tile_width": TILE_WIDTH,
                "latent_tile_height": TILE_HEIGHT,
                "latent_tile_overlap": TILE_OVERLAP,
                "source_width": SOURCE_WIDTH,
                "source_height": SOURCE_HEIGHT,
                "refinement_denoise": REFINEMENT_DENOISE,
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
    def _write_json(path: Path, value: object) -> None:
        """Write stable human-reviewable JSON."""

        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
