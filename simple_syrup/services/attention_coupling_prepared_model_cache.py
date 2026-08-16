# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reuse only exact immutable-input Attention Coupling preparations."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import TypeAlias
from uuid import UUID

import torch

from ..domain.conditioning_batch import ConditioningBatch
from ..domain.regional_attention_execution import RegionalAttentionExecutionMode
from ..runtime.regional_lora.execution_cache import ModelCloneLineage
from .attention_coupling_model_family import AttentionCouplingPreparedModelReuse
from .prepared_attention_coupling_model import PreparedAttentionCouplingModel

_IdentityToken: TypeAlias = tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _ObjectGraphSnapshot:
    """Retain hashable identity/version tokens and their strong references."""

    token: _IdentityToken
    references: tuple[object, ...] = field(compare=False, hash=False, repr=False)

    @classmethod
    def capture(cls, value: object) -> _ObjectGraphSnapshot:
        """Capture one built-in container graph without reading tensor values."""

        references: list[object] = []
        active: set[int] = set()

        def visit(current: object) -> _IdentityToken:
            if current is None or isinstance(current, bool | int | str | bytes):
                return ("value", type(current).__qualname__, current)
            if isinstance(current, float):
                return ("float", current.hex())
            if isinstance(current, UUID):
                return ("uuid", current.hex)
            if isinstance(current, Enum):
                return (
                    "enum",
                    type(current).__module__,
                    type(current).__qualname__,
                    current.value,
                )
            references.append(current)
            identity = id(current)
            if isinstance(current, torch.Tensor):
                return (
                    "tensor",
                    identity,
                    _tensor_mutation_version(current),
                    tuple(int(size) for size in current.shape),
                    tuple(int(stride) for stride in current.stride()),
                    str(current.dtype),
                    str(current.device),
                    bool(current.requires_grad),
                )
            if identity in active:
                return ("cycle", identity)
            if isinstance(current, ConditioningBatch):
                active.add(identity)
                try:
                    batch_entries = tuple(visit(item) for item in current.entries)
                finally:
                    active.remove(identity)
                return ("ConditioningBatch", identity, batch_entries)
            if isinstance(current, dict):
                active.add(identity)
                try:
                    dict_entries = tuple(
                        (visit(key), visit(item)) for key, item in current.items()
                    )
                finally:
                    active.remove(identity)
                return ("dict", identity, dict_entries)
            if isinstance(current, list | tuple):
                active.add(identity)
                try:
                    sequence_entries = tuple(visit(item) for item in current)
                finally:
                    active.remove(identity)
                return (type(current).__qualname__, identity, sequence_entries)
            return (
                "identity",
                type(current).__module__,
                type(current).__qualname__,
                identity,
            )

        return cls(visit(value), tuple(references))


@dataclass(frozen=True, slots=True)
class AttentionCouplingPreparedRequest:
    """Identify every input that participates in model preparation."""

    model_lineage: ModelCloneLineage
    model: _ObjectGraphSnapshot
    positive: _ObjectGraphSnapshot
    negative: _ObjectGraphSnapshot
    region_masks: _ObjectGraphSnapshot
    latent_image: _ObjectGraphSnapshot
    regional_prompt_weight: str
    region_mask_feather: int
    execution_mode: RegionalAttentionExecutionMode

    @classmethod
    def capture(
        cls,
        *,
        model: object,
        positive: object,
        negative: object,
        region_masks: object,
        regional_prompt_weight: float,
        region_mask_feather: int,
        latent_image: object,
        execution_mode: RegionalAttentionExecutionMode,
    ) -> AttentionCouplingPreparedRequest:
        """Capture one complete exact-input preparation identity."""

        if not isinstance(execution_mode, RegionalAttentionExecutionMode):
            raise TypeError("Prepared request execution mode is invalid.")
        if isinstance(region_mask_feather, bool) or not isinstance(
            region_mask_feather, int
        ):
            raise TypeError("Prepared request mask feather must be an integer.")
        if isinstance(regional_prompt_weight, bool) or not isinstance(
            regional_prompt_weight, int | float
        ):
            raise TypeError("Prepared request prompt weight must be real.")
        return cls(
            model_lineage=ModelCloneLineage.from_model(model),
            model=_ObjectGraphSnapshot.capture(model),
            positive=_ObjectGraphSnapshot.capture(positive),
            negative=_ObjectGraphSnapshot.capture(negative),
            region_masks=_ObjectGraphSnapshot.capture(region_masks),
            latent_image=_ObjectGraphSnapshot.capture(latent_image),
            regional_prompt_weight=float(regional_prompt_weight).hex(),
            region_mask_feather=region_mask_feather,
            execution_mode=execution_mode,
        )


@dataclass(frozen=True, slots=True)
class _PreparedEntry:
    """Retain one result and its post-publication MODEL patch identity."""

    prepared: PreparedAttentionCouplingModel
    prepared_patches_uuid: UUID

    @classmethod
    def capture(cls, prepared: PreparedAttentionCouplingModel) -> _PreparedEntry:
        """Capture the mutation-sensitive identity of one prepared result."""

        patches_uuid = getattr(prepared.model, "patches_uuid", None)
        if not isinstance(patches_uuid, UUID):
            raise TypeError("Prepared MODEL patches_uuid must be a UUID.")
        return cls(prepared, patches_uuid)

    def is_valid(self) -> bool:
        """Report whether the cached prepared MODEL remains unmodified."""

        return (
            getattr(self.prepared.model, "patches_uuid", None)
            == self.prepared_patches_uuid
        )


class AttentionCouplingPreparedModelCache:
    """Bound exact prepared requests and serialize first preparation per key."""

    def __init__(self, maximum_entries: int = 2) -> None:
        """Create one synchronized least-recently-used request cache."""

        if isinstance(maximum_entries, bool) or not isinstance(maximum_entries, int):
            raise TypeError("Prepared model cache size must be an integer.")
        if maximum_entries < 1:
            raise ValueError("Prepared model cache size must be positive.")
        self._maximum_entries = maximum_entries
        self._entries: OrderedDict[AttentionCouplingPreparedRequest, _PreparedEntry] = (
            OrderedDict()
        )
        self._lock = Lock()

    def resolve(
        self,
        *,
        request: AttentionCouplingPreparedRequest,
        policy: AttentionCouplingPreparedModelReuse,
        prepare: Callable[[], PreparedAttentionCouplingModel],
    ) -> PreparedAttentionCouplingModel:
        """Return one exact hit or atomically execute and publish its miss."""

        if not isinstance(request, AttentionCouplingPreparedRequest):
            raise TypeError("Prepared model cache requires a request identity.")
        if not isinstance(policy, AttentionCouplingPreparedModelReuse):
            raise TypeError("Prepared model cache reuse policy is invalid.")
        if not callable(prepare):
            raise TypeError("Prepared model cache requires a prepare callback.")
        if policy is AttentionCouplingPreparedModelReuse.DISABLED:
            return self._require_prepared(prepare())
        if policy is not AttentionCouplingPreparedModelReuse.EXACT_REQUEST:
            raise ValueError(f"Unsupported prepared model reuse policy: {policy!r}.")
        with self._lock:
            entry = self._entries.get(request)
            if entry is not None and entry.is_valid():
                self._entries.move_to_end(request)
                return entry.prepared
            if entry is not None:
                del self._entries[request]
            prepared = self._require_prepared(prepare())
            self._entries[request] = _PreparedEntry.capture(prepared)
            while len(self._entries) > self._maximum_entries:
                self._entries.popitem(last=False)
            return prepared

    def clear(self) -> None:
        """Release every bounded prepared-request owner."""

        with self._lock:
            self._entries.clear()

    @property
    def entry_count(self) -> int:
        """Return the current bounded entry count."""

        with self._lock:
            return len(self._entries)

    @staticmethod
    def _require_prepared(value: object) -> PreparedAttentionCouplingModel:
        """Narrow one successful cache callback result."""

        if not isinstance(value, PreparedAttentionCouplingModel):
            raise TypeError("Prepared model cache callback returned an invalid result.")
        return value


ATTENTION_COUPLING_PREPARED_MODEL_CACHE = AttentionCouplingPreparedModelCache()


def _tensor_mutation_version(tensor: torch.Tensor) -> int | None:
    """Return the mutation counter when the tensor runtime exposes one."""

    try:
        return int(tensor._version)
    except RuntimeError:
        return None
