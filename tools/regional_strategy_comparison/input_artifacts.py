# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own exact Comfy input artifacts for the P10.2 managed comparison."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import TracebackType

from PIL import Image

from tools.attention_coupling_benchmark.manifest_types import BenchmarkCase
from tools.attention_coupling_benchmark.mask_artifacts import MaskArtifactWriter

from .matrix import SOURCE_HEIGHT, SOURCE_WIDTH, TARGET_HEIGHT, TARGET_WIDTH


@dataclass(frozen=True, slots=True)
class ComparisonMaskNames:
    """Expose full and refinement mask filenames at their exact dimensions."""

    full: tuple[str, ...]
    refinement: tuple[str, ...]


class ComparisonInputArtifacts:
    """Materialize, evidence, and remove only one run's exact input files."""

    def __init__(self, input_root: Path, evidence_root: Path, run_id: str) -> None:
        """Reserve one collision-resistant filename namespace."""

        self._input_root = input_root.resolve()
        self._evidence_root = evidence_root.resolve()
        self._prefix = f"p102-{run_id.lower()}"
        self.source_name = f"{self._prefix}__shared-source.png"
        self._owned_paths: list[Path] = []
        self._source_sha256: str | None = None

    def __enter__(self) -> ComparisonInputArtifacts:
        """Return this exact lifecycle owner."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Remove every exact owned input file on success or failure."""

        del exc_type, exc_value, traceback
        self.cleanup()

    @property
    def source_sha256(self) -> str:
        """Return the published source digest."""

        if self._source_sha256 is None:
            raise RuntimeError("P10.2 shared source has not been published.")
        return self._source_sha256

    @property
    def cleanup_verified(self) -> bool:
        """Return whether every exact owned input path is absent."""

        return all(not path.exists() for path in self._owned_paths)

    def materialize_masks(self, case: BenchmarkCase) -> ComparisonMaskNames:
        """Write and evidence the two mask sizes used by the comparison."""

        full = self._write_masks(
            case, suffix="full", width=SOURCE_WIDTH, height=SOURCE_HEIGHT
        )
        refinement = self._write_masks(
            case,
            suffix="refinement",
            width=TARGET_WIDTH,
            height=TARGET_HEIGHT,
        )
        return ComparisonMaskNames(full, refinement)

    def publish_source(self, image_bytes: bytes) -> str:
        """Publish the accepted 1024 source to Comfy input and durable evidence."""

        if self._source_sha256 is not None:
            raise ValueError("P10.2 shared source was already published.")
        self._validate_png(image_bytes, SOURCE_WIDTH, SOURCE_HEIGHT)
        input_path = self._input_root / self.source_name
        if input_path.exists():
            raise FileExistsError(f"P10.2 source input already exists: {input_path}")
        input_path.write_bytes(image_bytes)
        self._owned_paths.append(input_path)
        evidence_path = self._evidence_root / "shared-source.png"
        if evidence_path.exists():
            raise FileExistsError(
                f"P10.2 source evidence already exists: {evidence_path}"
            )
        evidence_path.write_bytes(image_bytes)
        self._source_sha256 = hashlib.sha256(image_bytes).hexdigest()
        return self._source_sha256

    def cleanup(self) -> None:
        """Remove only exact run-owned Comfy input paths."""

        for path in reversed(self._owned_paths):
            path.unlink(missing_ok=True)

    def _write_masks(
        self,
        case: BenchmarkCase,
        *,
        suffix: str,
        width: int,
        height: int,
    ) -> tuple[str, ...]:
        """Write one mask size after proving its exact targets are unused."""

        prefix = f"{self._prefix}-{suffix}"
        expected = tuple(
            self._input_root / f"{prefix}__{case.case_id}__region-{index:02d}.png"
            for index in range(len(case.masks))
        )
        collision = next((path for path in expected if path.exists()), None)
        if collision is not None:
            raise FileExistsError(f"P10.2 mask input already exists: {collision}")
        names = MaskArtifactWriter(self._input_root, prefix).write_case(
            case,
            width=width,
            height=height,
        )
        paths = tuple(self._input_root / name for name in names)
        self._owned_paths.extend(paths)
        destination = self._evidence_root / "masks" / suffix
        destination.mkdir(parents=True, exist_ok=False)
        for path in paths:
            shutil.copyfile(path, destination / path.name)
        return names

    @staticmethod
    def _validate_png(image_bytes: bytes, width: int, height: int) -> None:
        """Require one decodable PNG at the exact source dimensions."""

        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            if image.format != "PNG":
                raise ValueError("P10.2 shared source must be a PNG image.")
            if image.size != (width, height):
                raise ValueError(
                    f"P10.2 shared source must be {width}x{height}, got {image.size}."
                )
