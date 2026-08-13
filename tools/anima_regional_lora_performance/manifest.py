# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the immutable pinned P5.7 performance benchmark contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PerformanceArtifact:
    """Identify one exact local benchmark artifact."""

    role: str
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PerformanceProfile:
    """Name one required regional adapter-count profile and overhead limit."""

    profile_id: str
    adapter_count: int
    maximum_overhead_percent: float


@dataclass(frozen=True, slots=True)
class PerformanceManifest:
    """Retain fixed full-context geometry, calls, repeats, and profiles."""

    benchmark_id: str
    width: int
    height: int
    latent_channels: int
    context_tokens: int
    context_features: int
    denoiser_calls: int
    warmup_calls: int
    repeats: int
    seed: int
    artifacts: tuple[PerformanceArtifact, ...]
    profiles: tuple[PerformanceProfile, ...]

    def __post_init__(self) -> None:
        """Require the exact zero/one/four gate and viable measurements."""

        if (self.width, self.height) != (1024, 1024):
            raise ValueError("P5.7 performance geometry must be 1024x1024.")
        if self.width % 8 or self.height % 8:
            raise ValueError("P5.7 dimensions must align to latent pixels.")
        if self.denoiser_calls != 30:
            raise ValueError("P5.7 requires exactly 30 complete denoiser calls.")
        if self.warmup_calls < 1 or self.repeats < 3:
            raise ValueError("P5.7 requires warmup and at least three repeats.")
        if tuple(profile.adapter_count for profile in self.profiles) != (0, 1, 4):
            raise ValueError("P5.7 profiles must retain zero, one, and four adapters.")
        if tuple(profile.maximum_overhead_percent for profile in self.profiles) != (
            0.0,
            15.0,
            35.0,
        ):
            raise ValueError(
                "P5.7 overhead limits must remain zero, 15, and 35 percent."
            )


def default_manifest(*, repeats: int = 3) -> PerformanceManifest:
    """Return the authoritative local RTX 5090 benchmark definition."""

    return PerformanceManifest(
        benchmark_id="anima-regional-lora-p5.7-v1",
        width=1024,
        height=1024,
        latent_channels=16,
        context_tokens=512,
        context_features=1024,
        denoiser_calls=30,
        warmup_calls=2,
        repeats=repeats,
        seed=1_029_384_756,
        artifacts=(
            PerformanceArtifact(
                role="anima_base",
                path=Path(r"<MODEL_ROOT>\diffusion_models\Anima")
                / "diffusion-model.safetensors",
                size_bytes=4_182_218_328,
                sha256=(
                    "bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e"
                ),
            ),
            PerformanceArtifact(
                role="regional_lora",
                path=Path(
                    r"<MODEL_ROOT>\Loras\Anima\style\adapter-a.safetensors"
                ),
                size_bytes=138_663_768,
                sha256=(
                    "0c915b59f464fd3d72c49a241440f8a41582ec76b597b0029e18b657b600c7d8"
                ),
            ),
        ),
        profiles=(
            PerformanceProfile("attention-only", 0, 0.0),
            PerformanceProfile("regional-lora-1", 1, 15.0),
            PerformanceProfile("regional-lora-4", 4, 35.0),
        ),
    )
