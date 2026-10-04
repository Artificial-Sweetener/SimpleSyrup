"""Define bounded learned scaling stages and verified format recommendations."""

from __future__ import annotations

import math

NATIVE_LATENT_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")
FORMAT_FAMILIES = {
    "SDXL": "sdxl",
    "Flux": "flux",
    "Flux2": "flux2",
    "Wan21": "wan",
    "Cosmos1CV8x8x8": "wan",
}


def upscale_target(shape: tuple[int, int], factor: float) -> tuple[int, int]:
    """Match Comfy's rounded latent dimensions while rejecting empty outputs."""
    if not math.isfinite(factor) or not 0.01 <= factor <= 8:
        raise ValueError("Latent scale factor must be between 0.01 and 8.")
    target = (round(shape[0] * factor), round(shape[1] * factor))
    if min(target) < 1:
        raise ValueError("Latent scale factor produces an empty spatial dimension.")
    return target


def learned_upscale_stages(
    source: tuple[int, int], target: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    """Reach exact dimensions through stages within the learned model's 1–2x range."""
    if min(*source, *target) < 2 or any(
        new < old for old, new in zip(source, target, strict=True)
    ):
        raise ValueError(
            "Learned latent upscaling requires enlargement and "
            "dimensions of at least 2 latent pixels."
        )
    current = source
    result: list[tuple[int, int]] = []
    while current != target:
        current = (min(current[0] * 2, target[0]), min(current[1] * 2, target[1]))
        result.append(current)
    return tuple(result)


def recommended_family(latent_format: str) -> str:
    """Select by verified format identity instead of ambiguous channel count."""
    family = FORMAT_FAMILIES.get(latent_format)
    if family is None:
        raise ValueError(
            f"Auto latent upscaling has no compatible model for "
            f"{latent_format}; select a native method or a compatible "
            f"checkpoint."
        )
    return family
