# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Group exact Anima ownership relations into compact attention batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .anima_self_attention_coherence import AnimaSelfAttentionCoherenceProfile


@dataclass(frozen=True, slots=True)
class AnimaSelfAttentionPartitionGroup:
    """Pack calls that share query and key sequence lengths."""

    batch_indices: torch.Tensor
    query_indices: torch.Tensor
    key_indices: torch.Tensor

    @property
    def call_count(self) -> int:
        """Return the number of compact attention calls in this batch."""

        return int(self.batch_indices.numel())

    @property
    def query_count(self) -> int:
        """Return the packed query sequence length."""

        return int(self.query_indices.shape[1])

    @property
    def key_count(self) -> int:
        """Return the packed key/value sequence length."""

        return int(self.key_indices.shape[1])


@dataclass(frozen=True, slots=True)
class AnimaSelfAttentionPartitionPlan:
    """Retain shape-compatible calls and exact per-token query coverage."""

    regional_groups: tuple[AnimaSelfAttentionPartitionGroup, ...]
    global_groups: tuple[AnimaSelfAttentionPartitionGroup, ...]
    query_coverage: torch.Tensor
    global_blend: torch.Tensor

    @classmethod
    def build(
        cls, profile: AnimaSelfAttentionCoherenceProfile
    ) -> AnimaSelfAttentionPartitionPlan:
        """Build compact regional and boundary-global attention calls."""

        cls._validate(profile)
        owners = profile.query_owners
        regional: dict[
            tuple[int, int], list[tuple[int, torch.Tensor, torch.Tensor]]
        ] = {}
        global_calls: dict[
            tuple[int, int], list[tuple[int, torch.Tensor, torch.Tensor]]
        ] = {}
        coverage = torch.zeros_like(owners)
        for batch_index in range(int(owners.shape[0])):
            row = owners[batch_index]
            shared_keys = profile.shared_keys[batch_index]
            global_queries = profile.global_blend[batch_index].gt(0).nonzero().flatten()
            if global_queries.numel():
                cls._append(
                    global_calls,
                    batch_index,
                    global_queries,
                    torch.arange(row.numel(), device=row.device),
                )
                coverage[batch_index, global_queries] += 1
            for owner in torch.unique(row[row >= 0]).tolist():
                queries = row.eq(owner).nonzero().flatten()
                keys = (row.eq(owner) | shared_keys).nonzero().flatten()
                cls._append(regional, batch_index, queries, keys)
                coverage[batch_index, queries] += 1
        expected = 1 + profile.global_blend.gt(0).to(torch.long)
        expected = torch.where(owners.eq(-1), expected.new_ones(()), expected)
        if not torch.equal(coverage, expected):
            raise ValueError("Anima attention partitions have invalid query coverage.")
        return cls(
            cls._groups(regional, owners),
            cls._groups(global_calls, owners),
            coverage,
            profile.global_blend,
        )

    @staticmethod
    def _groups(
        calls: dict[tuple[int, int], list[tuple[int, torch.Tensor, torch.Tensor]]],
        owners: torch.Tensor,
    ) -> tuple[AnimaSelfAttentionPartitionGroup, ...]:
        """Pack compatible calls into deterministic attention groups."""

        return tuple(
            AnimaSelfAttentionPartitionGroup(
                torch.tensor(
                    [call[0] for call in grouped],
                    device=owners.device,
                    dtype=torch.long,
                ),
                torch.stack([call[1] for call in grouped]),
                torch.stack([call[2] for call in grouped]),
            )
            for _, grouped in sorted(calls.items())
        )

    @staticmethod
    def _append(
        calls: dict[tuple[int, int], list[tuple[int, torch.Tensor, torch.Tensor]]],
        batch_index: int,
        queries: torch.Tensor,
        keys: torch.Tensor,
    ) -> None:
        """Append one call to the batch sharing its Q/K sequence lengths."""

        calls.setdefault((int(queries.numel()), int(keys.numel())), []).append(
            (batch_index, queries, keys)
        )

    @staticmethod
    def _validate(profile: AnimaSelfAttentionCoherenceProfile) -> None:
        """Require aligned ownership, shared-key, and blend tensors."""

        if not isinstance(profile, AnimaSelfAttentionCoherenceProfile):
            raise TypeError("Anima attention partition requires a coherence profile.")
        owners = profile.query_owners
        if not isinstance(owners, torch.Tensor):
            raise TypeError("Anima attention partition owners must be a tensor.")
        if owners.dtype != torch.long:
            raise TypeError("Anima attention partition owners must be integers.")
        if owners.ndim != 2 or any(int(size) < 1 for size in owners.shape):
            raise ValueError(
                "Anima attention partition owners must use non-empty B/Q layout."
            )
        if bool((owners < -1).any()):
            raise ValueError(
                "Anima attention partition owners contain invalid indices."
            )
        if (
            not isinstance(profile.shared_keys, torch.Tensor)
            or profile.shared_keys.dtype != torch.bool
            or profile.shared_keys.shape != owners.shape
        ):
            raise TypeError("Anima shared keys must use boolean B/Q layout.")
        blend = profile.global_blend
        if (
            not isinstance(blend, torch.Tensor)
            or not blend.is_floating_point()
            or blend.shape != owners.shape
            or not bool(torch.isfinite(blend).all())
            or not bool(((blend >= 0.0) & (blend <= 1.0)).all())
        ):
            raise ValueError("Anima global blend must use finite B/Q values in [0, 1].")
        if not bool(blend[owners.eq(-1)].eq(1.0).all()):
            raise ValueError("Shared Anima queries must use full global attention.")
