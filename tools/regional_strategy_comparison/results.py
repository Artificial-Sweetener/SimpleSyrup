# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persist and validate complete labeled P10.2 comparison evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

from PIL import Image

from tools.comfy_api import JsonObject

from .comparison import StrategyComparisonObservation, StrategyComparisonValidator
from .history import CompletedStrategyEvidence
from .matrix import (
    SOURCE_CASE_ID,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    StrategyComparisonCase,
)
from .workflow import BuiltStrategyComparisonWorkflow


class StrategyComparisonResultRecorder:
    """Own case association, durable evidence, and terminal acceptance."""

    def __init__(self, root: Path) -> None:
        """Retain one existing collision-resistant managed-run directory."""

        self._root = root.resolve()
        if not self._root.is_dir():
            raise ValueError("P10.2 result root must already exist.")
        self._records: list[JsonObject] = []
        self._observations: list[StrategyComparisonObservation] = []

    def record_metadata(self, metadata: JsonObject) -> Path:
        """Persist live public-node metadata for every compared sampler."""

        path = self._root / "public-node-metadata.json"
        self._write_json(path, metadata)
        return path

    def record_case(
        self,
        case: StrategyComparisonCase,
        workflow: BuiltStrategyComparisonWorkflow,
        evidence: CompletedStrategyEvidence,
        *,
        history: JsonObject,
        prompt_id: str,
        image_bytes: bytes,
        wall_runtime_ms: float,
    ) -> Path:
        """Persist one completed case and retain typed comparison evidence."""

        if any(record.get("case_id") == case.case_id for record in self._records):
            raise ValueError(f"P10.2 case was already recorded: {case.case_id}")
        if wall_runtime_ms <= 0:
            raise ValueError("P10.2 wall runtime must be positive.")
        expected_size = (
            (TARGET_WIDTH, TARGET_HEIGHT)
            if case.is_refinement
            else (SOURCE_WIDTH, SOURCE_HEIGHT)
        )
        self._validate_image(image_bytes, expected_size)
        image_path = self._root / f"{case.case_id}.png"
        history_path = self._root / f"{case.case_id}.history.json"
        workflow_path = self._root / f"{case.case_id}.workflow.json"
        image_path.write_bytes(image_bytes)
        if not history_path.exists() or not workflow_path.exists():
            self.record_raw_case(case, workflow, history)
        digest = hashlib.sha256(image_bytes).hexdigest()
        diagnostic_summary = self._diagnostic_summary(evidence)
        record: JsonObject = {
            "case_id": case.case_id,
            "label": case.label,
            "strategy": case.strategy,
            "spatial_profile": case.spatial_profile,
            "diffusion_mode": case.diffusion_mode,
            "regional_lora_count": case.regional_lora_count,
            "regional_loras": [asdict(adapter) for adapter in case.regional_loras],
            "prompt_id": prompt_id,
            "metrics": asdict(evidence.metrics),
            "diagnostics": diagnostic_summary,
            "wall_runtime_ms": wall_runtime_ms,
            "image_file": image_path.name,
            "image_sha256": digest,
            "image_size_bytes": len(image_bytes),
            "image_width": expected_size[0],
            "image_height": expected_size[1],
            "history_file": history_path.name,
            "workflow_file": workflow_path.name,
        }
        self._records.append(record)
        self._observations.append(
            StrategyComparisonObservation(case, evidence.metrics, evidence.diagnostics)
        )
        self._persist(status="running")
        return image_path

    def record_raw_case(
        self,
        case: StrategyComparisonCase,
        workflow: BuiltStrategyComparisonWorkflow,
        history: JsonObject,
    ) -> None:
        """Persist raw workflow and history before semantic acceptance."""

        self._write_json(self._root / f"{case.case_id}.history.json", history)
        self._write_json(
            self._root / f"{case.case_id}.workflow.json",
            workflow.prompt,
        )

    def finalize(
        self,
        definitions: tuple[StrategyComparisonCase, ...],
        *,
        source_sha256: str,
        system_stats: JsonObject,
        cleanup_verified: bool,
    ) -> Path:
        """Validate the complete matrix and persist the accepted result."""

        expected = tuple(case.case_id for case in definitions)
        observed = tuple(str(record["case_id"]) for record in self._records)
        if observed != expected:
            raise ValueError(
                "P10.2 result does not contain the complete ordered matrix."
            )
        if not cleanup_verified:
            raise RuntimeError("P10.2 managed process or input cleanup failed.")
        source = next(
            record for record in self._records if record["case_id"] == SOURCE_CASE_ID
        )
        if source.get("image_sha256") != source_sha256:
            raise ValueError(
                "P10.2 shared refinement source does not match its full case."
            )
        source_path = self._root / "shared-source.png"
        if (
            not source_path.is_file()
            or hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha256
        ):
            raise ValueError(
                "P10.2 durable shared-source artifact is missing or changed."
            )
        StrategyComparisonValidator().validate(tuple(self._observations))
        return self._persist(
            status="completed",
            extra={
                "source_sha256": source_sha256,
                "system_stats": system_stats,
                "cleanup_verified": True,
            },
        )

    @staticmethod
    def _diagnostic_summary(evidence: CompletedStrategyEvidence) -> JsonObject:
        """Summarize exact snapshot work without duplicating full histories."""

        cross_attention: set[object] = set()
        denoiser: set[object] = set()
        low_rank: set[object] = set()
        spatial_modes: set[object] = set()
        for snapshot in evidence.diagnostics:
            spatial_modes.add(snapshot.get("spatial_mode"))
            work = snapshot.get("estimated_work")
            if isinstance(work, dict):
                cross_attention.add(work.get("cross_attention_branch_multiplier"))
                denoiser.add(work.get("denoiser_call_multiplier"))
                low_rank.add(work.get("low_rank_adapter_multiplier", 0.0))
        return {
            "record_count": len(evidence.diagnostics),
            "cross_attention_branch_multipliers": sorted(cross_attention, key=str),
            "denoiser_call_multipliers": sorted(denoiser, key=str),
            "low_rank_adapter_multipliers": sorted(low_rank, key=str),
            "spatial_modes": sorted(spatial_modes, key=str),
        }

    def _persist(self, *, status: str, extra: JsonObject | None = None) -> Path:
        """Atomically replace the authoritative comparison result."""

        path = self._root / "p10.2-result.json"
        payload: JsonObject = {
            "schema_version": 1,
            "status": status,
            "public_node_metadata_file": "public-node-metadata.json",
            "shared_source_file": "shared-source.png",
            "observations": self._records,
        }
        if extra:
            payload.update(extra)
        self._write_json(path, payload)
        return path

    @staticmethod
    def _validate_image(image_bytes: bytes, expected_size: tuple[int, int]) -> None:
        """Require one finite decodable RGB-compatible PNG at exact dimensions."""

        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            if image.format != "PNG" or image.size != expected_size:
                raise ValueError(
                    f"P10.2 image must be PNG {expected_size}, got "
                    f"{image.format} {image.size}."
                )
            image.convert("RGB")

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        """Atomically write one stable JSON sidecar."""

        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
