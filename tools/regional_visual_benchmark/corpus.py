"""Persist and validate the complete managed P10.3 visual corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import cast

from PIL import Image

from tools.attention_coupling_benchmark.manifest_types import (
    BenchmarkManifest,
    JsonObject,
)
from tools.attention_coupling_benchmark.results import CompletedOutputs

from .matrix import SourcePosition, VisualPosition, source_positions, visual_positions
from .workflow import BuiltVisualWorkflow


class VisualCorpusRecorder:
    """Own durable source/output association and terminal corpus validation."""

    def __init__(self, root: Path, manifest: BenchmarkManifest) -> None:
        """Initialize or resume one exact P10.3 corpus journal."""

        self._root = root.resolve()
        self._manifest = manifest
        self._sources_root = self._root / "sources"
        self._images_root = self._root / "images"
        self._workflows_root = self._root / "workflows"
        self._histories_root = self._root / "histories"
        self._journal = self._root / "p10.3-corpus.inprogress.json"
        self._result = self._root / "p10.3-corpus.json"
        for directory in (
            self._sources_root,
            self._images_root,
            self._workflows_root,
            self._histories_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._sources: dict[str, JsonObject] = {}
        self._observations: dict[str, JsonObject] = {}
        if self._journal.exists():
            self._resume()

    @property
    def completed_source_ids(self) -> frozenset[str]:
        """Return every successfully recorded neutral source identity."""

        return frozenset(self._sources)

    @property
    def completed_artifact_ids(self) -> frozenset[str]:
        """Return every successfully recorded scored-output identity."""

        return frozenset(self._observations)

    @property
    def journal_path(self) -> Path:
        """Expose the authoritative in-progress corpus path for bounded runs."""

        return self._journal

    def source_path(self, source_id: str) -> Path:
        """Return one validated durable neutral-source path."""

        record = self._sources.get(source_id)
        if record is None:
            raise KeyError(f"P10.3 source was not recorded: {source_id}")
        path = self._root / _text(record.get("image_file"), "source image_file")
        expected = _text(record.get("image_sha256"), "source image_sha256")
        if not path.is_file() or _sha256(path.read_bytes()) != expected:
            raise ValueError(f"P10.3 durable source is missing or changed: {source_id}")
        return path

    def record_source(
        self,
        source: SourcePosition,
        workflow: BuiltVisualWorkflow,
        outputs: CompletedOutputs,
        *,
        history: JsonObject,
        prompt_id: str,
        image_bytes: bytes,
    ) -> Path:
        """Persist one strategy-neutral 1024 source and measured evidence."""

        if source.source_id in self._sources:
            raise ValueError(f"P10.3 source was already recorded: {source.source_id}")
        self._validate_png(image_bytes, (1024, 1024))
        image_path = self._sources_root / f"{source.source_id}.png"
        self._write_evidence(source.source_id, workflow, history)
        image_path.write_bytes(image_bytes)
        self._sources[source.source_id] = {
            "source_id": source.source_id,
            "case_id": source.case_id,
            "seed": source.seed,
            "prompt_id": prompt_id,
            "image_file": image_path.relative_to(self._root).as_posix(),
            "image_sha256": _sha256(image_bytes),
            "image_size_bytes": len(image_bytes),
            "metrics": asdict(outputs.metrics),
            "workflow_file": self._evidence_name(source.source_id, "workflow"),
            "history_file": self._evidence_name(source.source_id, "history"),
        }
        self._persist(status="running")
        return image_path

    def record_visual(
        self,
        position: VisualPosition,
        workflow: BuiltVisualWorkflow,
        outputs: CompletedOutputs,
        *,
        history: JsonObject,
        prompt_id: str,
        image_bytes: bytes,
        mask_sha256s: tuple[str, ...],
    ) -> Path:
        """Persist one scored-position output and its exact source association."""

        if position.artifact_id in self._observations:
            raise ValueError(
                f"P10.3 visual output was already recorded: {position.artifact_id}"
            )
        self._validate_png(image_bytes, position.expected_size)
        source_sha256: str | None = None
        if position.source_id is not None:
            source_record = self._sources.get(position.source_id)
            if source_record is None:
                raise ValueError(
                    f"P10.3 refinement source was not recorded: {position.source_id}"
                )
            self.source_path(position.source_id)
            source_sha256 = _text(
                source_record.get("image_sha256"), "source image_sha256"
            )
        image_path = self._images_root / f"{position.artifact_id}.png"
        self._write_evidence(position.artifact_id, workflow, history)
        image_path.write_bytes(image_bytes)
        self._observations[position.artifact_id] = {
            "artifact_id": position.artifact_id,
            "case_id": position.case_id,
            "seed": position.seed,
            "strategy": position.strategy.value,
            "spatial_profile": position.spatial_profile.value,
            "source_id": position.source_id,
            "source_sha256": source_sha256,
            "mask_sha256s": list(mask_sha256s),
            "prompt_id": prompt_id,
            "image_file": image_path.relative_to(self._root).as_posix(),
            "image_sha256": _sha256(image_bytes),
            "image_size_bytes": len(image_bytes),
            "image_width": position.expected_size[0],
            "image_height": position.expected_size[1],
            "metrics": asdict(outputs.metrics),
            "workflow_file": self._evidence_name(position.artifact_id, "workflow"),
            "history_file": self._evidence_name(position.artifact_id, "history"),
        }
        self._persist(status="running")
        return image_path

    def finalize(
        self,
        *,
        system_stats: JsonObject,
        cleanup_verified: bool,
    ) -> Path:
        """Publish only a complete hash-valid corpus after external cleanup."""

        if not cleanup_verified:
            raise RuntimeError("P10.3 managed process or input cleanup failed.")
        expected_sources = {
            value.source_id for value in source_positions(self._manifest)
        }
        expected_outputs = {
            value.artifact_id for value in visual_positions(self._manifest)
        }
        if set(self._sources) != expected_sources:
            raise ValueError("P10.3 source corpus is incomplete or contains extras.")
        if set(self._observations) != expected_outputs:
            raise ValueError("P10.3 visual corpus is incomplete or contains extras.")
        self._validate_durable_files()
        path = self._persist(
            status="completed",
            extra={
                "system_stats": system_stats,
                "cleanup_verified": True,
            },
            path=self._result,
        )
        self._journal.unlink(missing_ok=True)
        return path

    def _validate_durable_files(self) -> None:
        """Re-read every durable image and paired-source hash before publication."""

        for source_id in self._sources:
            self.source_path(source_id)
        for position in visual_positions(self._manifest):
            record = self._observations[position.artifact_id]
            path = self._root / _text(record.get("image_file"), "image_file")
            digest = _text(record.get("image_sha256"), "image_sha256")
            if not path.is_file() or _sha256(path.read_bytes()) != digest:
                raise ValueError(
                    "P10.3 durable visual is missing or changed: "
                    f"{position.artifact_id}"
                )
            self._validate_png(path.read_bytes(), position.expected_size)
            if position.source_id is not None:
                source_record = self._sources[position.source_id]
                if record.get("source_sha256") != source_record.get("image_sha256"):
                    raise ValueError(
                        "P10.3 paired refinement source hash is inconsistent."
                    )

    def _write_evidence(
        self,
        identity: str,
        workflow: BuiltVisualWorkflow,
        history: JsonObject,
    ) -> None:
        """Persist exact workflow and history under their focused directories."""

        self._write_json(
            self._workflows_root / f"{identity}.json",
            workflow.prompt,
        )
        self._write_json(self._histories_root / f"{identity}.json", history)

    @staticmethod
    def _evidence_name(identity: str, kind: str) -> str:
        """Return one root-relative evidence path."""

        return f"{kind}s/{identity}.json"

    def _resume(self) -> None:
        """Restore only a journal with the exact benchmark identity."""

        payload = _object(
            json.loads(self._journal.read_text(encoding="utf-8")), "corpus journal"
        )
        if payload.get("benchmark_id") != self._manifest.benchmark_id:
            raise ValueError("P10.3 journal belongs to another benchmark.")
        for value in _array(payload.get("sources"), "journal sources"):
            record = _object(value, "source record")
            self._sources[_text(record.get("source_id"), "source_id")] = record
        for value in _array(payload.get("observations"), "journal observations"):
            record = _object(value, "observation record")
            self._observations[_text(record.get("artifact_id"), "artifact_id")] = record

    def _persist(
        self,
        *,
        status: str,
        extra: JsonObject | None = None,
        path: Path | None = None,
    ) -> Path:
        """Atomically replace the ordered authoritative corpus record."""

        destination = path or self._journal
        payload: JsonObject = {
            "schema_version": 1,
            "benchmark_id": self._manifest.benchmark_id,
            "status": status,
            "sources": [
                self._sources[value.source_id]
                for value in source_positions(self._manifest)
                if value.source_id in self._sources
            ],
            "observations": [
                self._observations[value.artifact_id]
                for value in visual_positions(self._manifest)
                if value.artifact_id in self._observations
            ],
        }
        if extra:
            payload.update(extra)
        self._write_json(destination, payload)
        return destination

    @staticmethod
    def _validate_png(value: bytes, expected_size: tuple[int, int]) -> None:
        """Require one decoded RGB-compatible PNG at exact dimensions."""

        with Image.open(BytesIO(value)) as image:
            image.load()
            if image.format != "PNG" or image.size != expected_size:
                raise ValueError(
                    f"P10.3 image must be PNG {expected_size}, got "
                    f"{image.format} {image.size}."
                )
            image.convert("RGB")

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        """Atomically write one stable JSON artifact."""

        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def _sha256(value: bytes) -> str:
    """Return one lowercase byte identity."""

    return hashlib.sha256(value).hexdigest()


def _object(value: object, label: str) -> JsonObject:
    """Narrow one JSON object."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"P10.3 {label} must be an object.")
    return cast(JsonObject, value)


def _array(value: object, label: str) -> list[object]:
    """Narrow one JSON array."""

    if not isinstance(value, list):
        raise TypeError(f"P10.3 {label} must be an array.")
    return value


def _text(value: object, label: str) -> str:
    """Narrow one required nonempty string."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"P10.3 {label} must be a nonempty string.")
    return value
