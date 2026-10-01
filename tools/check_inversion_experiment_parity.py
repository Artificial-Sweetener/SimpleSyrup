# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Require lossless reproduction of accepted inversion images through both APIs."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REFERENCE_SHA256 = {
    "anima": "2b9a5932af9209651dc0bb72f317b64598a3cce5b08674eb8f485f98df689608",
    "krea": "4f8b256bbb3738004f2e9dc0f677aa2a7660faad1e0a7c89ec68606b1c26ccd0",
    "klein": "6485944a7633a2d2f4bf977f5b398f76f08678beea364dd6bc01d84430df7491",
}
INTERFACES = ("convenience", "options_stack")


@dataclass(frozen=True, slots=True)
class ImageParity:
    """Describe decoded-pixel equality independently of PNG metadata."""

    reference_size: tuple[int, int]
    candidate_size: tuple[int, int]
    changed_pixels: int | None
    maximum_channel_difference: int | None

    @property
    def exact(self) -> bool:
        """Accept only equal dimensions with no changed RGBA pixel."""

        return self.reference_size == self.candidate_size and self.changed_pixels == 0


def compare_images(reference: Path, candidate: Path) -> ImageParity:
    """Compare native-resolution decoded RGBA pixels without resizing or tolerance."""

    if reference.resolve() == candidate.resolve():
        raise ValueError(
            "Candidate output must not be the experimental reference file."
        )
    with (
        Image.open(reference) as reference_image,
        Image.open(candidate) as candidate_image,
    ):
        for image in (reference_image, candidate_image):
            if image.format != "PNG" or image.mode not in {"RGB", "RGBA"}:
                raise ValueError(
                    "Parity images must be lossless 8-bit RGB or RGBA PNGs."
                )
        sizes = (reference_image.size, candidate_image.size)
        if sizes[0] != sizes[1]:
            return ImageParity(*sizes, None, None)
        reference_pixels = np.asarray(reference_image.convert("RGBA"), dtype=np.int16)
        candidate_pixels = np.asarray(candidate_image.convert("RGBA"), dtype=np.int16)
    difference = np.abs(candidate_pixels - reference_pixels)
    return ImageParity(
        *sizes,
        changed_pixels=int(np.count_nonzero(np.any(difference != 0, axis=-1))),
        maximum_channel_difference=int(difference.max()),
    )


def check_experiment_parity(
    *, experimental_root: Path, generated_root: Path
) -> tuple[tuple[str, str, ImageParity], ...]:
    """Check all six outputs against fingerprinted, immutable winning references.

    Candidate images must be freshly generated through each production interface
    before this checker is run. Missing files, changed references, size differences,
    and any changed pixel block completion; visual similarity is insufficient.
    """

    observations: list[tuple[str, str, ImageParity]] = []
    for case, fingerprint in REFERENCE_SHA256.items():
        reference = experimental_root / case / "half2_full1" / "save_final.png"
        if hashlib.sha256(reference.read_bytes()).hexdigest() != fingerprint:
            raise ValueError(f"Frozen experimental reference changed: {case}.")
        for interface in INTERFACES:
            candidate = generated_root / interface / case / "save_final.png"
            observations.append((case, interface, compare_images(reference, candidate)))
    return tuple(observations)


def main(argv: Sequence[str] | None = None) -> int:
    """Fail the live-generation gate unless every required output is pixel-exact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental-root", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        observations = check_experiment_parity(
            experimental_root=args.experimental_root, generated_root=args.generated_root
        )
    except (OSError, ValueError) as error:
        print(
            f"FAILED: Inversion experiment parity is incomplete: {error}",
            file=sys.stderr,
        )
        return 1
    for case, interface, observation in observations:
        label = "PIXEL-EXACT" if observation.exact else "FAILED"
        print(
            f"{label}: {case}/{interface}; "
            f"dimensions={observation.candidate_size} vs {observation.reference_size}; "
            f"changed pixels={observation.changed_pixels}; "
            f"maximum channel difference={observation.maximum_channel_difference}"
        )
    return 0 if all(observation.exact for _, _, observation in observations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
