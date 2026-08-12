# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pack supported standard-UNet attention rows and restore source order."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class UnetAttentionBranchKey:
    """Identify the base branch or one ordered regional conditioning entry."""

    region_index: int | None
    entry_index: int | None

    def __post_init__(self) -> None:
        """Require either base identity or one complete regional identity."""

        if self.region_index is None or self.entry_index is None:
            if self.region_index is not None or self.entry_index is not None:
                raise ValueError(
                    "UNet attention branch indices must both be absent or present."
                )
            return
        if (
            isinstance(self.region_index, bool)
            or not isinstance(self.region_index, int)
            or self.region_index < 0
            or isinstance(self.entry_index, bool)
            or not isinstance(self.entry_index, int)
            or self.entry_index < 0
        ):
            raise ValueError(
                "UNet regional attention branch indices must be non-negative."
            )

    @property
    def is_base(self) -> bool:
        """Return whether this key identifies the base attention branch."""

        return self.region_index is None


UNET_BASE_ATTENTION_BRANCH = UnetAttentionBranchKey(None, None)


@dataclass(frozen=True, slots=True)
class UnetAttentionBranchSegment:
    """Bind one branch identity to its ordered supported source rows."""

    key: UnetAttentionBranchKey
    source_indices: torch.Tensor

    def __post_init__(self) -> None:
        """Require a non-empty, ordered, unique int64 index vector."""

        if not isinstance(self.key, UnetAttentionBranchKey):
            raise TypeError("UNet attention branch segment key has an invalid type.")
        if (
            not isinstance(self.source_indices, torch.Tensor)
            or self.source_indices.ndim != 1
            or self.source_indices.dtype is not torch.int64
            or int(self.source_indices.shape[0]) < 1
        ):
            raise ValueError(
                "UNet attention branch indices must be a non-empty int64 vector."
            )
        values = tuple(int(value) for value in self.source_indices.tolist())
        if values != tuple(sorted(set(values))) or values[0] < 0:
            raise ValueError(
                "UNet attention branch indices must be unique and ordered."
            )


@dataclass(frozen=True, slots=True)
class UnetAttentionBranchBatch:
    """Retain one compact UNet branch plan and exact restoration geometry."""

    source_batch_size: int
    segments: tuple[UnetAttentionBranchSegment, ...]
    packed_batch_size: int = field(init=False)

    def __post_init__(self) -> None:
        """Require canonical branch order and source-contained indices."""

        if type(self.source_batch_size) is not int or self.source_batch_size < 1:
            raise ValueError("UNet attention source batch size must be positive.")
        if not isinstance(self.segments, tuple) or not self.segments:
            raise ValueError("UNet attention branch batch requires segments.")
        if any(
            not isinstance(segment, UnetAttentionBranchSegment)
            for segment in self.segments
        ):
            raise TypeError("UNet attention branch batch contains an invalid segment.")
        keys = tuple(segment.key for segment in self.segments)
        if keys != tuple(sorted(set(keys), key=_branch_sort_key)):
            raise ValueError("UNet attention branch segments must be canonical.")
        device = self.segments[0].source_indices.device
        for segment in self.segments:
            if segment.source_indices.device != device:
                raise ValueError("UNet attention branch indices must share a device.")
            if int(segment.source_indices[-1]) >= self.source_batch_size:
                raise ValueError("UNet attention branch exceeds the source batch.")
        object.__setattr__(
            self,
            "packed_batch_size",
            sum(int(segment.source_indices.shape[0]) for segment in self.segments),
        )

    def pack_source(self, source: torch.Tensor) -> torch.Tensor:
        """Gather every supported row from one shared source tensor."""

        self._validate_source(source, name="source")
        return torch.cat(
            tuple(
                source.index_select(0, segment.source_indices)
                for segment in self.segments
            )
        )

    def pack_branch_values(
        self,
        values: Mapping[UnetAttentionBranchKey, torch.Tensor],
    ) -> torch.Tensor:
        """Gather each segment from its matching branch-owned tensor."""

        packed: list[torch.Tensor] = []
        authority: torch.Tensor | None = None
        for segment in self.segments:
            value = values.get(segment.key)
            if not isinstance(value, torch.Tensor):
                raise KeyError(
                    f"Missing UNet attention branch value for {segment.key!r}."
                )
            self._validate_source(value, name="branch value")
            if authority is None:
                authority = value
            elif (
                value.shape[1:] != authority.shape[1:]
                or value.device != authority.device
                or value.dtype != authority.dtype
            ):
                raise ValueError(
                    "UNet attention branch values must share shape, device, and dtype."
                )
            packed.append(value.index_select(0, segment.source_indices))
        return torch.cat(tuple(packed))

    def restore(
        self,
        packed: torch.Tensor,
    ) -> dict[UnetAttentionBranchKey, torch.Tensor]:
        """Scatter compact outputs into zero-filled source tensors by branch."""

        if not isinstance(packed, torch.Tensor) or packed.ndim < 1:
            raise TypeError("UNet compact attention output must be a tensor batch.")
        if int(packed.shape[0]) != self.packed_batch_size:
            raise ValueError("UNet compact attention output batch has an invalid size.")
        restored: dict[UnetAttentionBranchKey, torch.Tensor] = {}
        start = 0
        for segment in self.segments:
            count = int(segment.source_indices.shape[0])
            output = packed.new_zeros((self.source_batch_size, *packed.shape[1:]))
            output.index_copy_(
                0,
                segment.source_indices,
                packed[start : start + count],
            )
            restored[segment.key] = output
            start += count
        return restored

    def _validate_source(self, source: torch.Tensor, *, name: str) -> None:
        """Require one row per source item on the branch-index device."""

        if not isinstance(source, torch.Tensor) or source.ndim < 1:
            raise TypeError(f"UNet attention {name} must be a tensor batch.")
        if int(source.shape[0]) != self.source_batch_size:
            raise ValueError(f"UNet attention {name} batch does not match its plan.")
        if source.device != self.segments[0].source_indices.device:
            raise ValueError(
                f"UNet attention {name} and branch indices must share a device."
            )


class UnetAttentionBranchBatchBuilder:
    """Build exact compact UNet attention branches from row-level support."""

    def build(
        self,
        *,
        source_batch_size: int,
        supports: tuple[tuple[UnetAttentionBranchKey, torch.Tensor], ...],
    ) -> UnetAttentionBranchBatch:
        """Drop empty segments while preserving canonical branch and row order."""

        if type(source_batch_size) is not int or source_batch_size < 1:
            raise ValueError("UNet attention source batch size must be positive.")
        if not isinstance(supports, tuple) or not supports:
            raise ValueError("UNet attention branch construction requires supports.")
        keys = tuple(key for key, _support in supports)
        if keys != tuple(sorted(set(keys), key=_branch_sort_key)):
            raise ValueError("UNet attention branch supports must be canonical.")
        segments: list[UnetAttentionBranchSegment] = []
        device: torch.device | None = None
        for key, support in supports:
            if not isinstance(key, UnetAttentionBranchKey):
                raise TypeError("UNet attention support key has an invalid type.")
            if (
                not isinstance(support, torch.Tensor)
                or support.ndim != 1
                or support.dtype is not torch.bool
                or int(support.shape[0]) != source_batch_size
            ):
                raise ValueError(
                    "UNet attention support must be one boolean per source row."
                )
            if device is None:
                device = support.device
            elif support.device != device:
                raise ValueError("UNet attention supports must share one device.")
            indices = torch.nonzero(support, as_tuple=False).flatten()
            if int(indices.shape[0]) > 0:
                segments.append(UnetAttentionBranchSegment(key, indices))
        if not segments:
            raise ValueError("UNet attention supports cannot all be empty.")
        return UnetAttentionBranchBatch(source_batch_size, tuple(segments))


def _branch_sort_key(key: UnetAttentionBranchKey) -> tuple[int, int, int]:
    """Place the base first, followed by canonical region and entry order."""

    if key.is_base:
        return (0, -1, -1)
    if key.region_index is None or key.entry_index is None:
        raise AssertionError("Validated UNet branch key became incomplete.")
    return (1, key.region_index, key.entry_index)


UNET_ATTENTION_BRANCH_BATCH_BUILDER = UnetAttentionBranchBatchBuilder()
