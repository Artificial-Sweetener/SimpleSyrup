# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own immutable chunk-major regional attention batch alignment values."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .regional_attention import RegionalAttentionBranch


@dataclass(frozen=True, slots=True)
class RegionalAttentionChunkBatch:
    """Describe one Comfy chunk's contiguous slice of the model batch."""

    chunk_index: int
    branch: RegionalAttentionBranch
    batch_start: int
    batch_stop: int

    def __post_init__(self) -> None:
        """Validate one positive non-empty contiguous batch slice."""

        for name, value in (
            ("chunk_index", self.chunk_index),
            ("batch_start", self.batch_start),
            ("batch_stop", self.batch_stop),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Regional attention {name} must be an integer.")
        if self.chunk_index < 0 or self.batch_start < 0:
            raise ValueError("Regional attention chunk indices must be non-negative.")
        if self.batch_stop <= self.batch_start:
            raise ValueError("Regional attention chunk batch slice must be non-empty.")
        if not isinstance(self.branch, RegionalAttentionBranch):
            raise TypeError("Regional attention chunk branch has an invalid type.")


@dataclass(frozen=True, slots=True)
class BatchedRegionalAttentionEntry:
    """Retain one aligned regional entry and its per-sample Comfy strengths."""

    entry_index: int
    context: torch.Tensor
    strengths: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate entry order, aligned context, and finite sample strengths."""

        if isinstance(self.entry_index, bool) or not isinstance(self.entry_index, int):
            raise TypeError("Regional attention entry_index must be an integer.")
        if self.entry_index < 0:
            raise ValueError("Regional attention entry_index must be non-negative.")
        _validate_aligned_context(self.context, name="entry")
        if not isinstance(self.strengths, tuple):
            raise TypeError("Regional attention entry strengths must be a tuple.")
        if len(self.strengths) != int(self.context.shape[0]):
            raise ValueError(
                "Regional attention entry strength count must match its batch."
            )
        for strength in self.strengths:
            if isinstance(strength, bool) or not isinstance(strength, int | float):
                raise TypeError(
                    "Regional attention entry strength must be a real number."
                )
            if not math.isfinite(float(strength)):
                raise ValueError("Regional attention entry strength must be finite.")


@dataclass(frozen=True, slots=True)
class BatchedRegionalAttentionRegion:
    """Retain every aligned active conditioning entry for one region."""

    region_index: int
    entries: tuple[BatchedRegionalAttentionEntry, ...]

    def __post_init__(self) -> None:
        """Require a non-empty canonical entry bank for one region."""

        if isinstance(self.region_index, bool) or not isinstance(
            self.region_index, int
        ):
            raise TypeError("Regional attention region_index must be an integer.")
        if self.region_index < 0:
            raise ValueError("Regional attention region_index must be non-negative.")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("Regional attention region requires active entries.")
        if any(
            not isinstance(entry, BatchedRegionalAttentionEntry)
            for entry in self.entries
        ):
            raise TypeError("Regional attention region contains an invalid entry.")
        if tuple(entry.entry_index for entry in self.entries) != tuple(
            range(len(self.entries))
        ):
            raise ValueError("Regional attention region entries must be canonical.")


@dataclass(frozen=True, slots=True)
class BatchedRegionalAttentionContexts:
    """Retain chunk-major base and per-region active-entry model contexts."""

    latent_batch_size: int
    chunks: tuple[RegionalAttentionChunkBatch, ...]
    base_context: torch.Tensor
    regions: tuple[BatchedRegionalAttentionRegion, ...]

    def __post_init__(self) -> None:
        """Validate complete chunk and tensor alignment."""

        if isinstance(self.latent_batch_size, bool) or not isinstance(
            self.latent_batch_size, int
        ):
            raise TypeError("Regional attention latent_batch_size must be an integer.")
        if self.latent_batch_size < 1:
            raise ValueError("Regional attention latent_batch_size must be positive.")
        if not isinstance(self.chunks, tuple) or not self.chunks:
            raise ValueError("Regional attention batch requires at least one chunk.")
        if any(
            not isinstance(chunk, RegionalAttentionChunkBatch) for chunk in self.chunks
        ):
            raise TypeError("Regional attention batch contains an invalid chunk.")
        expected_start = 0
        for chunk_index, chunk in enumerate(self.chunks):
            if chunk.chunk_index != chunk_index or chunk.batch_start != expected_start:
                raise ValueError(
                    "Regional attention chunks must be contiguous and ordered."
                )
            if chunk.batch_stop - chunk.batch_start != self.latent_batch_size:
                raise ValueError(
                    "Regional attention chunk size must match latent batch."
                )
            expected_start = chunk.batch_stop
        _validate_aligned_context(
            self.base_context,
            expected_batch=expected_start,
            name="base",
        )
        if not isinstance(self.regions, tuple):
            raise TypeError("Regional attention regions must be a tuple.")
        if tuple(region.region_index for region in self.regions) != tuple(
            range(len(self.regions))
        ):
            raise ValueError("Regional attention regions must use canonical order.")
        for region in self.regions:
            for entry in region.entries:
                _validate_aligned_context(
                    entry.context,
                    expected_batch=expected_start,
                    name=f"region {region.region_index} entry {entry.entry_index}",
                )
                if entry.context.shape[1:] != self.base_context.shape[1:]:
                    raise ValueError(
                        "Regional attention context sequence shapes must match."
                    )
                if (
                    entry.context.device != self.base_context.device
                    or entry.context.dtype != self.base_context.dtype
                ):
                    raise ValueError(
                        "Regional attention context device and dtype must match."
                    )


def _validate_aligned_context(
    context: object,
    *,
    name: str,
    expected_batch: int | None = None,
) -> None:
    """Validate one finite floating BxSxD context tensor."""

    if not isinstance(context, torch.Tensor):
        raise TypeError(f"Regional attention {name} context must be a tensor.")
    if context.ndim != 3 or (
        expected_batch is not None and int(context.shape[0]) != expected_batch
    ):
        raise ValueError(
            f"Regional attention {name} context has an invalid aligned batch."
        )
    if not context.is_floating_point() or not bool(
        torch.isfinite(context).all().item()
    ):
        raise ValueError(
            f"Regional attention {name} context must contain finite floating values."
        )
