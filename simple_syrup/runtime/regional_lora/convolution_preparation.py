# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare immutable regional convolution LoRA tensors per device and dtype."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import torch


@dataclass(frozen=True, slots=True)
class RegionalConvolutionPreparedWeights:
    """Retain one exact prepared down/optional-middle/up tensor set."""

    down: torch.Tensor
    middle: torch.Tensor | None
    up: torch.Tensor


@dataclass(slots=True)
class RegionalConvolutionPreparation:
    """Prepare one admitted convolution target once per execution type."""

    down: torch.Tensor
    middle: torch.Tensor | None
    up: torch.Tensor
    _prepared: dict[
        tuple[torch.device, torch.dtype], RegionalConvolutionPreparedWeights
    ] = field(default_factory=dict, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Require finite CPU floating tensors without copying their contents."""

        for name, tensor in (
            ("down", self.down),
            ("up", self.up),
            ("middle", self.middle),
        ):
            if tensor is None and name == "middle":
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Regional convolution {name} weight must be a tensor.")
            if tensor.device.type != "cpu" or not tensor.is_floating_point():
                raise TypeError(
                    f"Regional convolution {name} weight must be floating CPU data."
                )
            if tensor.ndim < 2 or not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"Regional convolution {name} weight must be finite "
                    "rank-two-or-higher data."
                )

    def weights(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RegionalConvolutionPreparedWeights:
        """Return detached full-quality tensors prepared once for one execution type."""

        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("Regional convolution execution dtype must be floating.")
        key = (torch.device(device), dtype)
        existing = self._prepared.get(key)
        if existing is not None:
            return existing
        with self._lock:
            existing = self._prepared.get(key)
            if existing is not None:
                return existing
            prepared = RegionalConvolutionPreparedWeights(
                self.down.detach().to(device=key[0], dtype=dtype, copy=True),
                (
                    None
                    if self.middle is None
                    else self.middle.detach().to(
                        device=key[0],
                        dtype=dtype,
                        copy=True,
                    )
                ),
                self.up.detach().to(device=key[0], dtype=dtype, copy=True),
            )
            self._prepared[key] = prepared
            return prepared

    def clear(self) -> None:
        """Release every locally retained device/dtype tensor set."""

        with self._lock:
            self._prepared.clear()
