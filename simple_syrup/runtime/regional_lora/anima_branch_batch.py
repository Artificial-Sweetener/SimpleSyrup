# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pack supported Anima regional branches and restore source-batch order."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class AnimaRegionalBranchKey:
    """Identify the base branch or one ordered regional conditioning entry."""

    region_index: int | None
    entry_index: int | None

    def __post_init__(self) -> None:
        """Require base identity or one complete non-negative regional identity."""

        if self.region_index is None or self.entry_index is None:
            if self.region_index is not None or self.entry_index is not None:
                raise ValueError(
                    "Anima branch region and entry indices must both be absent or set."
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
            raise ValueError("Anima regional branch indices must be non-negative.")

    @property
    def is_base(self) -> bool:
        """Return whether this key identifies the base branch."""

        return self.region_index is None


ANIMA_BASE_BRANCH_KEY = AnimaRegionalBranchKey(None, None)


@dataclass(frozen=True, slots=True)
class AnimaRegionalBranchSegment:
    """Bind one branch identity to its ordered supported source rows."""

    key: AnimaRegionalBranchKey
    source_indices: torch.Tensor

    def __post_init__(self) -> None:
        """Require a non-empty strictly ordered int64 index vector."""

        if not isinstance(self.key, AnimaRegionalBranchKey):
            raise TypeError("Anima branch segment key has an invalid type.")
        if (
            not isinstance(self.source_indices, torch.Tensor)
            or self.source_indices.ndim != 1
            or self.source_indices.dtype is not torch.int64
            or int(self.source_indices.shape[0]) < 1
        ):
            raise ValueError(
                "Anima branch segment source indices must be a non-empty int64 vector."
            )
        values = tuple(int(value) for value in self.source_indices.tolist())
        if values != tuple(sorted(set(values))) or values[0] < 0:
            raise ValueError(
                "Anima branch segment source indices must be unique and ordered."
            )


@dataclass(frozen=True, slots=True)
class AnimaRegionalBranchInvocation:
    """Map every compact branch row to its source row and regional identity."""

    source_batch_size: int
    source_batch_indices: tuple[int, ...]
    region_indices: tuple[int | None, ...]

    def __post_init__(self) -> None:
        """Require complete immutable compact-row identity."""

        if type(self.source_batch_size) is not int or self.source_batch_size < 1:
            raise ValueError("Anima branch source batch size must be positive.")
        if not isinstance(self.source_batch_indices, tuple) or not isinstance(
            self.region_indices,
            tuple,
        ):
            raise TypeError("Anima branch invocation mappings must be tuples.")
        if not self.source_batch_indices or len(self.source_batch_indices) != len(
            self.region_indices
        ):
            raise ValueError("Anima branch invocation mappings must be non-empty.")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.source_batch_size
            for index in self.source_batch_indices
        ):
            raise ValueError(
                "Anima branch source indices are outside the source batch."
            )
        if any(
            index is not None
            and (isinstance(index, bool) or not isinstance(index, int) or index < 0)
            for index in self.region_indices
        ):
            raise ValueError("Anima branch region identities are invalid.")

    @property
    def packed_batch_size(self) -> int:
        """Return the number of rows evaluated by the compact branch call."""

        return len(self.source_batch_indices)


@dataclass(frozen=True, slots=True)
class AnimaRegionalBranchBatch:
    """Retain one compact branch plan and exact restoration geometry."""

    source_batch_size: int
    segments: tuple[AnimaRegionalBranchSegment, ...]
    invocation: AnimaRegionalBranchInvocation = field(init=False)

    def __post_init__(self) -> None:
        """Require canonical keys, one device, and source-contained indices."""

        if type(self.source_batch_size) is not int or self.source_batch_size < 1:
            raise ValueError("Anima branch source batch size must be positive.")
        if not isinstance(self.segments, tuple) or not self.segments:
            raise ValueError("Anima branch batch requires supported segments.")
        if any(
            not isinstance(segment, AnimaRegionalBranchSegment)
            for segment in self.segments
        ):
            raise TypeError("Anima branch batch contains an invalid segment.")
        keys = tuple(segment.key for segment in self.segments)
        if keys != tuple(sorted(set(keys), key=_branch_sort_key)):
            raise ValueError("Anima branch segments must be unique and canonical.")
        device = self.segments[0].source_indices.device
        for segment in self.segments:
            if segment.source_indices.device != device:
                raise ValueError("Anima branch segment indices must share one device.")
            if int(segment.source_indices[-1]) >= self.source_batch_size:
                raise ValueError("Anima branch segment exceeds the source batch.")
        object.__setattr__(
            self,
            "invocation",
            AnimaRegionalBranchInvocation(
                self.source_batch_size,
                tuple(
                    int(index)
                    for segment in self.segments
                    for index in segment.source_indices.tolist()
                ),
                tuple(
                    segment.key.region_index
                    for segment in self.segments
                    for _index in range(int(segment.source_indices.shape[0]))
                ),
            ),
        )

    @property
    def packed_batch_size(self) -> int:
        """Return the total number of supported rows across all segments."""

        return sum(int(segment.source_indices.shape[0]) for segment in self.segments)

    def pack_source(self, source: torch.Tensor) -> torch.Tensor:
        """Repeat supported rows from one shared source tensor by segment."""

        self._validate_source(source, name="source")
        return torch.cat(
            tuple(
                source.index_select(0, segment.source_indices)
                for segment in self.segments
            )
        )

    def pack_branch_values(
        self,
        values: Mapping[AnimaRegionalBranchKey, torch.Tensor],
    ) -> torch.Tensor:
        """Pack each segment from its matching branch-owned source tensor."""

        packed: list[torch.Tensor] = []
        authority_shape: tuple[int, ...] | None = None
        authority_device: torch.device | None = None
        authority_dtype: torch.dtype | None = None
        for segment in self.segments:
            value = values.get(segment.key)
            if not isinstance(value, torch.Tensor):
                raise KeyError(f"Missing Anima branch value for {segment.key!r}.")
            self._validate_source(value, name="branch value")
            shape = tuple(int(size) for size in value.shape[1:])
            if authority_shape is None:
                authority_shape = shape
                authority_device = value.device
                authority_dtype = value.dtype
            elif (
                shape != authority_shape
                or value.device != authority_device
                or value.dtype != authority_dtype
            ):
                raise ValueError(
                    "Anima branch values must share shape and execution type."
                )
            packed.append(value.index_select(0, segment.source_indices))
        return torch.cat(tuple(packed))

    def restore(
        self, packed: torch.Tensor
    ) -> dict[AnimaRegionalBranchKey, torch.Tensor]:
        """Scatter compact outputs into zero-filled source-batch tensors by key."""

        if not isinstance(packed, torch.Tensor) or packed.ndim < 1:
            raise TypeError("Anima compact branch output must be a tensor batch.")
        if int(packed.shape[0]) != self.packed_batch_size:
            raise ValueError("Anima compact branch output batch has an invalid size.")
        restored: dict[AnimaRegionalBranchKey, torch.Tensor] = {}
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
        """Require one tensor row per authoritative source-batch item."""

        if not isinstance(source, torch.Tensor) or source.ndim < 1:
            raise TypeError(f"Anima branch {name} must be a tensor batch.")
        if int(source.shape[0]) != self.source_batch_size:
            raise ValueError(f"Anima branch {name} batch does not match its plan.")
        if source.device != self.segments[0].source_indices.device:
            raise ValueError(f"Anima branch {name} and indices must share one device.")


class AnimaRegionalBranchBatchBuilder:
    """Build one exact compact branch batch from canonical row support."""

    def build(
        self,
        *,
        source_batch_size: int,
        supports: tuple[tuple[AnimaRegionalBranchKey, torch.Tensor], ...],
    ) -> AnimaRegionalBranchBatch:
        """Drop empty segments while preserving canonical branch and row order."""

        if type(source_batch_size) is not int or source_batch_size < 1:
            raise ValueError("Anima branch source batch size must be positive.")
        if not isinstance(supports, tuple) or not supports:
            raise ValueError("Anima branch batch construction requires supports.")
        keys = tuple(key for key, _support in supports)
        if keys != tuple(sorted(set(keys), key=_branch_sort_key)):
            raise ValueError("Anima branch supports must be unique and canonical.")
        segments: list[AnimaRegionalBranchSegment] = []
        device: torch.device | None = None
        for key, support in supports:
            if not isinstance(key, AnimaRegionalBranchKey):
                raise TypeError("Anima branch support key has an invalid type.")
            if (
                not isinstance(support, torch.Tensor)
                or support.ndim != 1
                or support.dtype is not torch.bool
                or int(support.shape[0]) != source_batch_size
            ):
                raise ValueError(
                    "Anima branch support must be one boolean per source row."
                )
            if device is None:
                device = support.device
            elif support.device != device:
                raise ValueError("Anima branch supports must share one device.")
            indices = torch.nonzero(support, as_tuple=False).flatten()
            if int(indices.shape[0]) > 0:
                segments.append(AnimaRegionalBranchSegment(key, indices))
        if not segments:
            raise ValueError("Anima branch supports cannot all be empty.")
        return AnimaRegionalBranchBatch(source_batch_size, tuple(segments))


def _branch_sort_key(key: AnimaRegionalBranchKey) -> tuple[int, int, int]:
    """Place base first, followed by canonical region and entry order."""

    if key.is_base:
        return (0, -1, -1)
    if key.region_index is None or key.entry_index is None:
        raise AssertionError("Validated regional branch key became incomplete.")
    return (1, key.region_index, key.entry_index)


ANIMA_REGIONAL_BRANCH_BATCH_BUILDER = AnimaRegionalBranchBatchBuilder()
