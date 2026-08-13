# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Write exact cleanup-owned U11 regional mask profile images."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from .matrix import TARGET_HEIGHT, TARGET_WIDTH
from .visual_cases import VisualMaskProfile


class ManagedSdxlVisualMasks:
    """Own left/right input mask pairs for every declared profile."""

    def __init__(self, *, input_root: Path, run_id: str) -> None:
        """Validate one existing Comfy input root and safe run identity."""

        self._input_root = input_root.resolve()
        if not self._input_root.is_dir():
            raise FileNotFoundError("Comfy input directory does not exist.")
        if not run_id or any(character in run_id for character in "\\/:"):
            raise ValueError("U11 mask run ID contains invalid path characters.")
        self._paths = {
            profile: tuple(
                self._input_root
                / f"simple_syrup_u11_{run_id}_{profile.value}_{side}.png"
                for side in ("left", "right")
            )
            for profile in VisualMaskProfile
        }
        self.cleaned = False

    def names(self, profile: VisualMaskProfile) -> tuple[str, str]:
        """Return one profile's ordered Comfy input filenames."""

        left, right = self._paths[profile]
        return left.name, right.name

    def __enter__(self) -> ManagedSdxlVisualMasks:
        """Write every exact profile after collision validation."""

        paths = tuple(path for pair in self._paths.values() for path in pair)
        if any(path.exists() for path in paths):
            raise FileExistsError("An owned U11 mask already exists.")
        written: list[Path] = []
        try:
            for profile, pair in self._paths.items():
                left, right = _render_profile(profile)
                left.save(pair[0], format="PNG")
                written.append(pair[0])
                right.save(pair[1], format="PNG")
                written.append(pair[1])
        except BaseException:
            for path in written:
                path.unlink(missing_ok=True)
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Remove only exact profile files created by this owner."""

        del exc_type, exc, traceback
        paths = tuple(path for pair in self._paths.values() for path in pair)
        for path in paths:
            path.unlink(missing_ok=True)
        self.cleaned = not any(path.exists() for path in paths)

    def evidence(self) -> tuple[dict[str, object], ...]:
        """Return stable profile, side, dimension, and digest evidence."""

        return tuple(
            {
                "profile": profile.value,
                "side": side,
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": TARGET_WIDTH,
                "height": TARGET_HEIGHT,
            }
            for profile, pair in self._paths.items()
            for side, path in zip(("left", "right"), pair, strict=True)
        )


def _render_profile(profile: VisualMaskProfile) -> tuple[Image.Image, Image.Image]:
    """Render one pair without relying on sampler-side implicit geometry."""

    left = Image.new("L", (TARGET_WIDTH, TARGET_HEIGHT), 0)
    right = Image.new("L", (TARGET_WIDTH, TARGET_HEIGHT), 0)
    left_draw = ImageDraw.Draw(left)
    right_draw = ImageDraw.Draw(right)
    if profile is VisualMaskProfile.HARD:
        left_draw.rectangle((0, 0, 819, TARGET_HEIGHT - 1), fill=255)
        right_draw.rectangle((717, 0, TARGET_WIDTH - 1, TARGET_HEIGHT - 1), fill=255)
    elif profile is VisualMaskProfile.SOFT_OVERLAP:
        left_draw.rectangle((0, 0, 895, TARGET_HEIGHT - 1), fill=255)
        right_draw.rectangle((640, 0, TARGET_WIDTH - 1, TARGET_HEIGHT - 1), fill=255)
    elif profile is VisualMaskProfile.UNCOVERED_CENTER:
        left_draw.rectangle((0, 0, 639, TARGET_HEIGHT - 1), fill=255)
        right_draw.rectangle((896, 0, TARGET_WIDTH - 1, TARGET_HEIGHT - 1), fill=255)
    else:
        raise ValueError(f"Unsupported U11 mask profile: {profile!r}")
    return left, right
