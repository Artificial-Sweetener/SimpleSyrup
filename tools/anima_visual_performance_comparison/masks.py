# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own exact hard left/right masks for the Anima visual comparison."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


class ManagedAnimaComparisonMasks:
    """Write and remove one collision-safe 1024-square hard partition."""

    def __init__(self, *, input_root: Path, run_id: str) -> None:
        """Retain one existing Comfy input root and safe run identity."""

        self._root = input_root.resolve()
        if not self._root.is_dir() or not run_id or any(c in run_id for c in "\\/:"):
            raise ValueError("Anima comparison mask location or run ID is invalid.")
        self._paths = tuple(
            self._root / f"anima-visual-{run_id}-{side}.png"
            for side in ("left", "right")
        )
        self.cleaned = False

    @property
    def names(self) -> tuple[str, str]:
        """Return ordered Comfy input names."""

        return self._paths[0].name, self._paths[1].name

    def __enter__(self) -> ManagedAnimaComparisonMasks:
        """Render the exact non-overlapping hard partition."""

        if any(path.exists() for path in self._paths):
            raise FileExistsError("Anima comparison mask already exists.")
        for index, path in enumerate(self._paths):
            image = Image.new("L", (1024, 1024), 0)
            draw = ImageDraw.Draw(image)
            draw.rectangle((index * 512, 0, index * 512 + 511, 1023), fill=255)
            image.save(path, format="PNG")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Remove only files owned by this instance."""

        del exc_type, exc, traceback
        for path in self._paths:
            path.unlink(missing_ok=True)
        self.cleaned = not any(path.exists() for path in self._paths)
