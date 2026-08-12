"""Compare decoded RGB image pixels independently of container metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True, slots=True)
class DecodedImageComparison:
    """Describe one exact decoded RGB pixel comparison."""

    changed_pixels: int
    maximum_channel_delta: int
    width: int
    height: int

    def as_json(self) -> dict[str, int]:
        """Return deterministic machine-readable comparison evidence."""

        return {
            "changed_pixels": self.changed_pixels,
            "maximum_channel_delta": self.maximum_channel_delta,
            "width": self.width,
            "height": self.height,
        }


def compare_decoded_rgb_images(
    left_path: Path,
    right_path: Path,
) -> DecodedImageComparison:
    """Compare decoded RGB pixels without considering file metadata."""

    with Image.open(left_path) as left_source, Image.open(right_path) as right_source:
        left = left_source.convert("RGB")
        right = right_source.convert("RGB")
    if left.size != right.size:
        raise ValueError("Compared decoded images must have identical dimensions.")
    left_bytes = left.tobytes()
    right_bytes = right.tobytes()
    maximum_delta = 0
    changed_pixels = 0
    for offset in range(0, len(left_bytes), 3):
        deltas = tuple(
            abs(left_bytes[offset + channel] - right_bytes[offset + channel])
            for channel in range(3)
        )
        maximum_delta = max(maximum_delta, *deltas)
        changed_pixels += int(any(deltas))
    return DecodedImageComparison(changed_pixels, maximum_delta, *left.size)
