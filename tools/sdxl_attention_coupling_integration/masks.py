"""Own deterministic temporary split masks for the SDXL integration run."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from .matrix import REGION_WIDTH, RIGHT_REGION_START, TARGET_HEIGHT, TARGET_WIDTH


class ManagedSdxlSplitMasks:
    """Write and remove two exact normal-input mask files for one run."""

    def __init__(self, *, input_root: Path, run_id: str) -> None:
        """Retain one validated input root and collision-resistant prefix."""

        self._input_root = input_root.resolve()
        if not self._input_root.is_dir():
            raise FileNotFoundError("Comfy input directory does not exist.")
        if not run_id or any(character in run_id for character in "\\/:"):
            raise ValueError("SDXL mask run ID contains invalid path characters.")
        self._names = (
            f"simple_syrup_p8_5_{run_id}_left.png",
            f"simple_syrup_p8_5_{run_id}_right.png",
        )
        self._paths = tuple(self._input_root / name for name in self._names)
        self.cleaned = False

    @property
    def names(self) -> tuple[str, str]:
        """Return ordered Comfy input filenames."""

        return self._names

    def __enter__(self) -> ManagedSdxlSplitMasks:
        """Create the exact left/right masks after collision validation."""

        if any(path.exists() for path in self._paths):
            raise FileExistsError("Owned SDXL integration mask already exists.")
        for path, bounds in zip(
            self._paths,
            (
                (0, 0, REGION_WIDTH - 1, TARGET_HEIGHT - 1),
                (RIGHT_REGION_START, 0, TARGET_WIDTH - 1, TARGET_HEIGHT - 1),
            ),
            strict=True,
        ):
            image = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), "black")
            ImageDraw.Draw(image).rectangle(bounds, fill="white")
            image.save(path, format="PNG")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Remove only the two exact files owned by this context."""

        del exc_type, exc, traceback
        for path in self._paths:
            path.unlink(missing_ok=True)
        self.cleaned = not any(path.exists() for path in self._paths)

    def evidence(self) -> tuple[dict[str, object], ...]:
        """Return stable filenames and hashes without exposing local paths."""

        return tuple(
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "width": TARGET_WIDTH,
                "height": TARGET_HEIGHT,
            }
            for path in self._paths
        )
