# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare and cache exact regional-LoRA target tensors for execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import torch

from ...domain.regional_lora_plan import RegionalLoraAdapterIdentity
from .execution_cache import (
    REGIONAL_LORA_EXECUTION_CACHE_KEY_FACTORY,
    ModelCloneLineage,
    RegionalLoraExecutionCache,
    RegionalLoraExecutionCacheKeyFactory,
    RegionalLoraPreparedWeights,
)
from .standard_adapter import StandardLoraTarget


@dataclass
class RegionalLoraTargetPreparation:
    """Prepare one admitted target once per exact device and dtype."""

    adapter_identity: RegionalLoraAdapterIdentity
    model_lineage: ModelCloneLineage
    target: StandardLoraTarget
    cache: RegionalLoraExecutionCache
    key_factory: RegionalLoraExecutionCacheKeyFactory = (
        REGIONAL_LORA_EXECUTION_CACHE_KEY_FACTORY
    )
    _prepared: dict[tuple[torch.device, torch.dtype], RegionalLoraPreparedWeights] = (
        field(
            default_factory=dict,
            init=False,
            repr=False,
        )
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the immutable cache identity sources before device work."""

        if not isinstance(self.adapter_identity, RegionalLoraAdapterIdentity):
            raise TypeError("Regional LoRA preparation identity has an invalid type.")
        if not isinstance(self.model_lineage, ModelCloneLineage):
            raise TypeError("Regional LoRA preparation lineage has an invalid type.")
        if not isinstance(self.target, StandardLoraTarget):
            raise TypeError("Regional LoRA preparation target has an invalid type.")
        if not isinstance(self.cache, RegionalLoraExecutionCache):
            raise TypeError("Regional LoRA preparation cache has an invalid type.")
        if not isinstance(self.key_factory, RegionalLoraExecutionCacheKeyFactory):
            raise TypeError(
                "Regional LoRA preparation key factory has an invalid type."
            )

    def weights(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RegionalLoraPreparedWeights:
        """Return full-rank A/B weights prepared once for one execution type."""

        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("Regional LoRA execution dtype must be floating point.")
        target_device = torch.device(device)
        local_key = (target_device, dtype)
        existing = self._prepared.get(local_key)
        if existing is not None:
            return existing
        with self._lock:
            existing = self._prepared.get(local_key)
            if existing is not None:
                return existing
            cache_key = self.key_factory.build(
                adapter_identity=self.adapter_identity,
                model_lineage=self.model_lineage,
                adapter=self.target,
                device=target_device,
                dtype=dtype,
            )
            prepared = self.cache.get_or_prepare(
                cache_key,
                lambda: RegionalLoraPreparedWeights(
                    down=self.target.down.detach().to(
                        device=target_device,
                        dtype=dtype,
                        copy=True,
                    ),
                    up=self.target.up.detach().to(
                        device=target_device,
                        dtype=dtype,
                        copy=True,
                    ),
                ),
            )
            self._prepared[local_key] = prepared
            return prepared

    def clear(self) -> None:
        """Release locally retained prepared device weights."""

        with self._lock:
            self._prepared.clear()


@dataclass(frozen=True, slots=True)
class RegionalLoraPreparedBatch:
    """Retain concatenated A and stacked B weights for compatible targets."""

    down: torch.Tensor
    up: torch.Tensor


@dataclass
class RegionalLoraCompatibleBatchPreparation:
    """Prepare one immutable compatible target group per device and dtype."""

    targets: tuple[RegionalLoraTargetPreparation, ...]
    _prepared: dict[tuple[torch.device, torch.dtype], RegionalLoraPreparedBatch] = (
        field(
            default_factory=dict,
            init=False,
            repr=False,
        )
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Require non-empty targets with one exact rank and feature shape."""

        if not isinstance(self.targets, tuple) or not self.targets:
            raise ValueError("Regional LoRA compatible batch cannot be empty.")
        if any(
            not isinstance(target, RegionalLoraTargetPreparation)
            for target in self.targets
        ):
            raise TypeError("Regional LoRA compatible batch has an invalid target.")
        first = self.targets[0].target
        contract = (first.rank, first.input_features, first.output_features)
        if any(
            (
                target.target.rank,
                target.target.input_features,
                target.target.output_features,
            )
            != contract
            for target in self.targets
        ):
            raise ValueError(
                "Regional LoRA compatible batch targets must share rank and shape."
            )

    @property
    def rank(self) -> int:
        """Return the shared full adapter rank."""

        return self.targets[0].target.rank

    @property
    def input_features(self) -> int:
        """Return the shared input feature width."""

        return self.targets[0].target.input_features

    @property
    def output_features(self) -> int:
        """Return the shared output feature width."""

        return self.targets[0].target.output_features

    def weights(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RegionalLoraPreparedBatch:
        """Return concatenated full-rank weights prepared once per execution type."""

        key = (torch.device(device), dtype)
        existing = self._prepared.get(key)
        if existing is not None:
            return existing
        with self._lock:
            existing = self._prepared.get(key)
            if existing is not None:
                return existing
            weights = tuple(
                target.weights(device=device, dtype=dtype) for target in self.targets
            )
            prepared = RegionalLoraPreparedBatch(
                down=torch.cat(tuple(weight.down for weight in weights), dim=0),
                up=torch.stack(tuple(weight.up for weight in weights), dim=0),
            )
            self._prepared[key] = prepared
            return prepared

    def clear(self) -> None:
        """Release locally retained compatible device batches."""

        with self._lock:
            self._prepared.clear()
