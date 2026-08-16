# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own one exact all-one mask for regional reference-parity execution."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image


class ManagedAllOneMask:
    """Write and remove one collision-safe all-white 1024-square mask."""

    def __init__(self, *, input_root: Path, run_id: str) -> None:
        """Validate the Comfy input root and safe run identity."""

        self._input_root = input_root.resolve()
        if not self._input_root.is_dir():
            raise FileNotFoundError("Comfy input directory does not exist.")
        if not run_id or any(character in run_id for character in "\\/:"):
            raise ValueError("Full-strength fidelity run ID is invalid.")
        self._path = self._input_root / f"simple_syrup_fidelity_{run_id}_all-one.png"
        self.cleaned = False

    @property
    def name(self) -> str:
        """Return the exact Comfy input filename."""

        return self._path.name

    def __enter__(self) -> ManagedAllOneMask:
        """Write one exact all-one mask after collision validation."""

        if self._path.exists():
            raise FileExistsError("Full-strength fidelity mask already exists.")
        Image.new("L", (1024, 1024), 255).save(self._path, format="PNG")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Remove only the mask created by this owner."""

        del exc_type, exc, traceback
        self._path.unlink(missing_ok=True)
        self.cleaned = not self._path.exists()

    def evidence(self) -> dict[str, object]:
        """Return stable geometry and digest evidence for the active mask."""

        return {
            "name": self.name,
            "width": 1024,
            "height": 1024,
            "minimum": 255,
            "maximum": 255,
            "sha256": hashlib.sha256(self._path.read_bytes()).hexdigest(),
        }
