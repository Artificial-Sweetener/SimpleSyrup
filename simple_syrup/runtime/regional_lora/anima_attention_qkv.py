# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt installed Anima QKV calls by inspected host capability."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self, cast

import torch
from torch import nn


class AnimaAttentionQkvCapability(Enum):
    """Identify the installed Comfy QKV invocation contract."""

    LEGACY = "legacy"
    TRANSFORMER_OPTIONS = "transformer_options"


@dataclass(frozen=True)
class AnimaAttentionQkvAdapter:
    """Invoke one validated installed Comfy QKV boundary."""

    capability: AnimaAttentionQkvCapability

    @classmethod
    def discover(cls, attention: nn.Module) -> Self:
        """Classify the bound host method without relying on version strings."""

        compute_qkv = getattr(attention, "compute_qkv", None)
        if not callable(compute_qkv):
            raise ValueError("compute_qkv must be callable")
        try:
            signature = inspect.signature(compute_qkv)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"compute_qkv signature cannot be inspected: {error}"
            ) from error

        if cls._binds(
            signature,
            object(),
            None,
            rope_emb=None,
            transformer_options={},
        ):
            return cls(AnimaAttentionQkvCapability.TRANSFORMER_OPTIONS)
        if cls._binds(signature, object(), None, rope_emb=None):
            return cls(AnimaAttentionQkvCapability.LEGACY)
        raise ValueError(
            "compute_qkv must accept x, context, rope_emb, and optionally "
            "transformer_options"
        )

    def compute(
        self,
        attention: nn.Module,
        x: torch.Tensor,
        context: torch.Tensor | None,
        *,
        rope_emb: torch.Tensor | None,
        transformer_options: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Call the installed signature and validate its tensor contract."""

        raw_compute_qkv = getattr(attention, "compute_qkv", None)
        if not callable(raw_compute_qkv):
            raise TypeError("Installed Anima attention compute_qkv is not callable.")
        compute_qkv = cast(Callable[..., object], raw_compute_qkv)
        if self.capability is AnimaAttentionQkvCapability.TRANSFORMER_OPTIONS:
            result = compute_qkv(
                x,
                context,
                rope_emb=rope_emb,
                transformer_options=transformer_options,
            )
        else:
            result = compute_qkv(x, context, rope_emb=rope_emb)
        if (
            not isinstance(result, tuple)
            or len(result) != 3
            or not all(isinstance(value, torch.Tensor) for value in result)
        ):
            raise TypeError("Installed Anima compute_qkv returned invalid tensors.")
        return cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], result)

    @staticmethod
    def _binds(
        signature: inspect.Signature,
        *args: object,
        **kwargs: object,
    ) -> bool:
        """Report whether one complete invocation satisfies the host signature."""

        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            return False
        return True


__all__ = ["AnimaAttentionQkvAdapter", "AnimaAttentionQkvCapability"]
