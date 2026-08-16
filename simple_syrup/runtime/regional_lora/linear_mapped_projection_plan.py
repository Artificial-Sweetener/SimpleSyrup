# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plan exact repeated-target mapping for regional Linear projection fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import torch
from torch.nn import functional

from .preparation import RegionalLoraTargetPreparation


@dataclass(frozen=True, slots=True)
class RegionalLinearMappedWeights:
    """Retain padded full-rank unique A/B tensors for one execution type."""

    down: torch.Tensor
    up: torch.Tensor


@dataclass(slots=True)
class RegionalLinearMappedProjectionPlan:
    """Retain one unique-target batch and declared group mapping."""

    preparations: tuple[RegionalLoraTargetPreparation, ...]
    unique_preparations: tuple[RegionalLoraTargetPreparation, ...] = field(init=False)
    group_target_indices: tuple[int, ...] = field(init=False)
    rank: int = field(init=False)
    input_features: int = field(init=False)
    output_features: int = field(init=False)
    supported: bool = field(init=False)
    _prepared: dict[tuple[torch.device, torch.dtype], RegionalLinearMappedWeights] = (
        field(
            default_factory=dict,
            init=False,
            repr=False,
        )
    )
    _device_indices: dict[torch.device, torch.Tensor] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Intern exact A/B identities while retaining every target's full rank."""

        if not isinstance(self.preparations, tuple) or not self.preparations:
            raise ValueError("Mapped Linear projection requires preparations.")
        if any(
            not isinstance(preparation, RegionalLoraTargetPreparation)
            for preparation in self.preparations
        ):
            raise TypeError("Mapped Linear projection preparation is invalid.")
        first = self.preparations[0].target
        self.input_features = first.input_features
        self.output_features = first.output_features
        if any(
            preparation.target.input_features != self.input_features
            or preparation.target.output_features != self.output_features
            for preparation in self.preparations
        ):
            self.unique_preparations = ()
            self.group_target_indices = ()
            self.rank = 0
            self.supported = False
            return
        unique: list[RegionalLoraTargetPreparation] = []
        indices: dict[tuple[int, int], int] = {}
        group_target_indices: list[int] = []
        for preparation in self.preparations:
            target = preparation.target
            identity = (id(target.down), id(target.up))
            target_index = indices.get(identity)
            if target_index is None:
                target_index = len(unique)
                indices[identity] = target_index
                unique.append(preparation)
            group_target_indices.append(target_index)
        self.unique_preparations = tuple(unique)
        self.group_target_indices = tuple(group_target_indices)
        self.rank = max(preparation.target.rank for preparation in unique)
        self.supported = True

    @property
    def target_count(self) -> int:
        """Return the number of exact unique A/B targets."""

        return len(self.unique_preparations)

    def weights(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RegionalLinearMappedWeights:
        """Return full-rank targets zero-padded only in fused storage."""

        if not self.supported:
            raise ValueError("Mapped Linear projection contract is incompatible.")
        key = (torch.device(device), dtype)
        existing = self._prepared.get(key)
        if existing is not None:
            return existing
        with self._lock:
            existing = self._prepared.get(key)
            if existing is not None:
                return existing
            targets = tuple(
                preparation.weights(device=key[0], dtype=dtype)
                for preparation in self.unique_preparations
            )
            prepared = RegionalLinearMappedWeights(
                torch.cat(
                    tuple(self._pad_down(weight.down, self.rank) for weight in targets),
                    dim=0,
                ),
                torch.stack(
                    tuple(self._pad_up(weight.up, self.rank) for weight in targets),
                    dim=0,
                ),
            )
            self._prepared[key] = prepared
            return prepared

    def target_indices(self, device: torch.device) -> torch.Tensor:
        """Return the declared group mapping on one execution device."""

        if not self.supported:
            raise ValueError("Mapped Linear projection contract is incompatible.")
        target_device = torch.device(device)
        existing = self._device_indices.get(target_device)
        if existing is not None:
            return existing
        prepared = torch.tensor(
            self.group_target_indices,
            device=target_device,
            dtype=torch.int64,
        )
        self._device_indices[target_device] = prepared
        return prepared

    def clear(self) -> None:
        """Release padded weights and device-local mapping tensors."""

        with self._lock:
            self._prepared.clear()
            self._device_indices.clear()

    @staticmethod
    def _pad_down(down: torch.Tensor, rank: int) -> torch.Tensor:
        """Pad rank rows without altering admitted A values."""

        padding = rank - int(down.shape[0])
        return down if padding == 0 else functional.pad(down, (0, 0, 0, padding))

    @staticmethod
    def _pad_up(up: torch.Tensor, rank: int) -> torch.Tensor:
        """Pad rank columns without altering admitted B values."""

        padding = rank - int(up.shape[1])
        return up if padding == 0 else functional.pad(up, (0, padding))
