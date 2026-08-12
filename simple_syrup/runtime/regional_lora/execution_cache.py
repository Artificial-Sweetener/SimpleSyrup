# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own safe cache identity and prepared weights for regional LoRA execution."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from uuid import UUID

import torch

from ...domain.regional_lora_plan import RegionalLoraAdapterIdentity
from .standard_adapter import StandardLoraTarget


@dataclass(frozen=True)
class ModelCloneLineage:
    """Identify one Comfy clone family and effective global weight patch state."""

    clone_base_uuid: UUID
    patches_uuid: UUID

    @classmethod
    def from_model(cls, model: object) -> ModelCloneLineage:
        """Capture immutable cache identity without retaining the model patcher."""

        clone_base_uuid = getattr(model, "clone_base_uuid", None)
        patches_uuid = getattr(model, "patches_uuid", None)
        if not isinstance(clone_base_uuid, UUID):
            raise TypeError("MODEL clone_base_uuid must be a UUID for LoRA caching.")
        if not isinstance(patches_uuid, UUID):
            raise TypeError("MODEL patches_uuid must be a UUID for LoRA caching.")
        return cls(clone_base_uuid, patches_uuid)


@dataclass(frozen=True)
class AdapterPatchContent:
    """Identify immutable admitted A/B tensor shape, dtype, and bytes."""

    sha256: str

    def __post_init__(self) -> None:
        """Require one lowercase SHA-256 digest."""

        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("Regional LoRA patch content requires a SHA-256 digest.")


@dataclass(frozen=True)
class RegionalLoraExecutionCacheKey:
    """Key prepared adapter weights without schedule or spatial runtime inputs."""

    adapter_identity: RegionalLoraAdapterIdentity
    model_lineage: ModelCloneLineage
    target: str
    device: str
    dtype: torch.dtype
    rank: int
    input_features: int
    output_features: int
    patch_content: AdapterPatchContent


@dataclass(frozen=True)
class RegionalLoraPreparedWeights:
    """Retain one device/dtype-prepared low-rank tensor pair."""

    down: torch.Tensor
    up: torch.Tensor


class RegionalLoraExecutionCacheKeyFactory:
    """Build complete cache identity from admitted CPU adapter state."""

    def build(
        self,
        *,
        adapter_identity: RegionalLoraAdapterIdentity,
        model_lineage: ModelCloneLineage,
        adapter: StandardLoraTarget,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> RegionalLoraExecutionCacheKey:
        """Create one deterministic key without moving adapter tensors."""

        if not isinstance(adapter_identity, RegionalLoraAdapterIdentity):
            raise TypeError("Regional LoRA cache adapter identity has an invalid type.")
        if not isinstance(model_lineage, ModelCloneLineage):
            raise TypeError("Regional LoRA cache model lineage has an invalid type.")
        if not isinstance(adapter, StandardLoraTarget):
            raise TypeError("Regional LoRA cache target has an invalid type.")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("Regional LoRA cache dtype must be a torch dtype.")
        target_device = torch.device(device)
        return RegionalLoraExecutionCacheKey(
            adapter_identity=adapter_identity,
            model_lineage=model_lineage,
            target=adapter.target,
            device=str(target_device),
            dtype=dtype,
            rank=adapter.rank,
            input_features=adapter.input_features,
            output_features=adapter.output_features,
            patch_content=self._content(adapter),
        )

    @staticmethod
    def _content(adapter: StandardLoraTarget) -> AdapterPatchContent:
        """Hash admitted CPU tensors including structural metadata and bytes."""

        digest = hashlib.sha256()
        for side, tensor in (("A", adapter.down), ("B", adapter.up)):
            if tensor.device.type != "cpu":
                raise ValueError(
                    "Regional LoRA cache content must be computed from CPU tensors."
                )
            contiguous = tensor.detach().contiguous()
            digest.update(side.encode("ascii"))
            digest.update(str(tuple(contiguous.shape)).encode("ascii"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(contiguous.view(torch.uint8).numpy().tobytes())
        return AdapterPatchContent(digest.hexdigest())


class RegionalLoraExecutionCache:
    """Prepare each complete cache key once and validate cached tensor residency."""

    def __init__(self) -> None:
        """Create an empty process-local prepared-weight cache."""

        self._values: dict[
            RegionalLoraExecutionCacheKey,
            RegionalLoraPreparedWeights,
        ] = {}
        self._lock = Lock()

    def get_or_prepare(
        self,
        key: RegionalLoraExecutionCacheKey,
        prepare: Callable[[], RegionalLoraPreparedWeights],
    ) -> RegionalLoraPreparedWeights:
        """Return a validated hit or atomically prepare and store one value."""

        if not isinstance(key, RegionalLoraExecutionCacheKey):
            raise TypeError("Regional LoRA execution cache key has an invalid type.")
        if not callable(prepare):
            raise TypeError(
                "Regional LoRA execution cache prepare callback is invalid."
            )
        with self._lock:
            existing = self._values.get(key)
            if existing is not None:
                return existing
            prepared = prepare()
            self._validate_prepared(key, prepared)
            self._values[key] = prepared
            return prepared

    @property
    def size(self) -> int:
        """Return the number of complete prepared entries."""

        with self._lock:
            return len(self._values)

    def clear(self) -> None:
        """Release every prepared device tensor while retaining cache identity."""

        with self._lock:
            self._values.clear()

    @staticmethod
    def _validate_prepared(
        key: RegionalLoraExecutionCacheKey,
        prepared: object,
    ) -> None:
        """Require cached tensors to match all execution-shape key fields."""

        if not isinstance(prepared, RegionalLoraPreparedWeights):
            raise TypeError("Regional LoRA prepared cache value has an invalid type.")
        expected_device = torch.device(key.device)
        for name, tensor in (("down", prepared.down), ("up", prepared.up)):
            if tensor.device != expected_device:
                raise ValueError(
                    f"Regional LoRA prepared {name} tensor is on the wrong device."
                )
            if tensor.dtype != key.dtype:
                raise ValueError(
                    f"Regional LoRA prepared {name} tensor has the wrong dtype."
                )
            if tensor.ndim != 2 or not tensor.is_floating_point():
                raise ValueError(
                    f"Regional LoRA prepared {name} tensor must be floating rank 2."
                )
        if tuple(prepared.down.shape) != (key.rank, key.input_features):
            raise ValueError("Regional LoRA prepared down tensor has the wrong shape.")
        if tuple(prepared.up.shape) != (key.output_features, key.rank):
            raise ValueError("Regional LoRA prepared up tensor has the wrong shape.")


REGIONAL_LORA_EXECUTION_CACHE_KEY_FACTORY = RegionalLoraExecutionCacheKeyFactory()
