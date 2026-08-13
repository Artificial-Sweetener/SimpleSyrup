# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own exact Comfy input files for one P10.3 visual position."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from types import TracebackType

from PIL import Image

from tools.attention_coupling_benchmark.manifest_types import BenchmarkCase
from tools.attention_coupling_benchmark.mask_artifacts import MaskArtifactWriter

from .matrix import SOURCE_HEIGHT, SOURCE_WIDTH, VisualPosition


class VisualPositionInputs:
    """Materialize and remove only one position's exact input artifacts."""

    def __init__(
        self,
        input_root: Path,
        run_id: str,
        position: VisualPosition,
        case: BenchmarkCase,
        *,
        source_path: Path | None,
    ) -> None:
        """Retain one collision-resistant position namespace."""

        self._input_root = input_root.resolve()
        self._position = position
        self._case = case
        self._source_path = source_path.resolve() if source_path is not None else None
        digest = hashlib.sha256(position.artifact_id.encode("utf-8")).hexdigest()[:16]
        self._prefix = f"p103-{run_id.lower()}-{digest}"
        self._owned_paths: list[Path] = []
        self.mask_names: tuple[str, ...] = ()
        self.mask_sha256s: tuple[str, ...] = ()
        self.source_name: str | None = None
        self.source_sha256: str | None = None

    def __enter__(self) -> VisualPositionInputs:
        """Publish the exact mask and optional source files."""

        self._validate_source_contract()
        self.mask_names = self._write_masks()
        self.mask_sha256s = tuple(
            _sha256((self._input_root / name).read_bytes()) for name in self.mask_names
        )
        if self._source_path is not None:
            self.source_name = f"{self._prefix}__source.png"
            destination = self._input_root / self.source_name
            self._require_unused(destination)
            shutil.copyfile(self._source_path, destination)
            self._owned_paths.append(destination)
            self.source_sha256 = _sha256(destination.read_bytes())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Remove every exact owned file on success or failure."""

        del exc_type, exc_value, traceback
        self.cleanup()

    @property
    def cleanup_verified(self) -> bool:
        """Return whether every exact owned input path is absent."""

        return all(not path.exists() for path in self._owned_paths)

    def cleanup(self) -> None:
        """Remove only this position's exact owned input paths."""

        for path in reversed(self._owned_paths):
            path.unlink(missing_ok=True)

    def _validate_source_contract(self) -> None:
        """Require source presence only for corrected refinement profiles."""

        if self._position.is_refinement != (self._source_path is not None):
            raise ValueError("P10.3 input source does not match the spatial profile.")
        if self._source_path is None:
            return
        if not self._source_path.is_file():
            raise FileNotFoundError(f"P10.3 source is missing: {self._source_path}")
        with Image.open(self._source_path) as image:
            image.load()
            if image.format != "PNG" or image.size != (SOURCE_WIDTH, SOURCE_HEIGHT):
                raise ValueError(
                    "P10.3 source must be a decoded PNG at exactly "
                    f"{SOURCE_WIDTH}x{SOURCE_HEIGHT}."
                )

    def _write_masks(self) -> tuple[str, ...]:
        """Materialize authored masks at the exact output dimensions."""

        if not self._case.masks:
            return ()
        width, height = self._position.expected_size
        expected = tuple(
            self._input_root
            / f"{self._prefix}__{self._case.case_id}__region-{index:02d}.png"
            for index in range(len(self._case.masks))
        )
        for path in expected:
            self._require_unused(path)
        names = MaskArtifactWriter(self._input_root, self._prefix).write_case(
            self._case,
            width=width,
            height=height,
        )
        self._owned_paths.extend(self._input_root / name for name in names)
        return names

    @staticmethod
    def _require_unused(path: Path) -> None:
        """Reject collision with any pre-existing external input file."""

        if path.exists():
            raise FileExistsError(f"P10.3 input artifact already exists: {path}")


def _sha256(value: bytes) -> str:
    """Return one lowercase byte identity."""

    return hashlib.sha256(value).hexdigest()
