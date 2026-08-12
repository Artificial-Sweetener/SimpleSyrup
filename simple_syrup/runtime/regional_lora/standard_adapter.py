# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decode standard rank-decomposed LoRA tensor pairs without model policy."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import torch

_PAIR_SUFFIXES = {
    ".lora_A.weight": ("A", "a_b"),
    ".lora_B.weight": ("B", "a_b"),
    ".lora_down.weight": ("A", "down_up"),
    ".lora_up.weight": ("B", "down_up"),
}
_ALPHA_SUFFIX = ".alpha"


@dataclass(frozen=True)
class StandardLoraDecodeIssue:
    """Describe one adapter-format failure without losing its source key."""

    key: str
    reason: str


@dataclass(frozen=True)
class StandardLoraTarget:
    """Retain one validated CPU low-rank pair and its intrinsic scale."""

    target: str
    down: torch.Tensor
    up: torch.Tensor
    rank: int
    input_features: int
    output_features: int
    intrinsic_scale: float = 1.0


@dataclass(frozen=True)
class StandardLoraDecodeResult:
    """Return every decodable target and every format issue in one pass."""

    targets: tuple[StandardLoraTarget, ...]
    issues: tuple[StandardLoraDecodeIssue, ...]


class StandardLoraAdapterDecoder:
    """Decode common standard linear LoRA tensor layouts on CPU."""

    def decode(self, weights: object) -> StandardLoraDecodeResult:
        """Inspect all supplied entries and retain every issue before admission."""

        if not isinstance(weights, Mapping):
            return StandardLoraDecodeResult(
                targets=(),
                issues=(
                    StandardLoraDecodeIssue(
                        key="<adapter>",
                        reason="adapter weights must be a mapping",
                    ),
                ),
            )
        grouped: dict[str, dict[str, tuple[str, object, str]]] = defaultdict(dict)
        issues: list[StandardLoraDecodeIssue] = []
        sortable_entries: list[tuple[str, object]] = []
        for raw_key, value in weights.items():
            if not isinstance(raw_key, str):
                issues.append(
                    StandardLoraDecodeIssue(
                        key=repr(raw_key),
                        reason="adapter tensor key must be a string",
                    )
                )
                continue
            sortable_entries.append((raw_key, value))
        for key, value in sorted(sortable_entries, key=lambda item: item[0]):
            matched = next(
                (
                    (suffix, side, layout)
                    for suffix, (side, layout) in _PAIR_SUFFIXES.items()
                    if key.endswith(suffix)
                ),
                None,
            )
            if matched is not None:
                suffix, side, layout = matched
                target = key.removesuffix(suffix)
            elif key.endswith(_ALPHA_SUFFIX):
                target = key.removesuffix(_ALPHA_SUFFIX)
                side = "alpha"
                layout = "scalar"
            else:
                issues.append(
                    StandardLoraDecodeIssue(
                        key=key,
                        reason=(
                            "unsupported adapter format; expected lora_A/lora_B or "
                            "lora_down/lora_up weight"
                        ),
                    )
                )
                continue
            if not target:
                issues.append(
                    StandardLoraDecodeIssue(
                        key=key,
                        reason="adapter target must be non-empty",
                    )
                )
                continue
            if side in grouped[target]:
                issues.append(
                    StandardLoraDecodeIssue(
                        key=key,
                        reason=f"duplicate adapter {side} entry for target",
                    )
                )
                continue
            grouped[target][side] = (key, value, layout)

        targets: list[StandardLoraTarget] = []
        for target, pair in sorted(grouped.items()):
            missing = tuple(side for side in ("A", "B") if side not in pair)
            if missing:
                present_key = next(
                    (entry[0] for side, entry in pair.items() if side != "alpha"),
                    target,
                )
                issues.append(
                    StandardLoraDecodeIssue(
                        key=present_key,
                        reason=f"incomplete adapter pair; missing {', '.join(missing)}",
                    )
                )
                continue
            down_key, down_value, down_layout = pair["A"]
            up_key, up_value, up_layout = pair["B"]
            if down_layout != up_layout:
                issues.append(
                    StandardLoraDecodeIssue(
                        key=target,
                        reason="adapter pair mixes incompatible key layouts",
                    )
                )
                continue
            pair_issues = self._validate_pair(
                target,
                down_key,
                down_value,
                up_key,
                up_value,
            )
            if pair_issues:
                issues.extend(pair_issues)
                continue
            down = down_value
            up = up_value
            assert isinstance(down, torch.Tensor)
            assert isinstance(up, torch.Tensor)
            rank = int(down.shape[0])
            intrinsic_scale, alpha_issues = self._decode_intrinsic_scale(
                pair.get("alpha"),
                rank,
            )
            if alpha_issues:
                issues.extend(alpha_issues)
                continue
            targets.append(
                StandardLoraTarget(
                    target=target,
                    down=down,
                    up=up,
                    rank=rank,
                    input_features=int(down.shape[1]),
                    output_features=int(up.shape[0]),
                    intrinsic_scale=intrinsic_scale,
                )
            )
        return StandardLoraDecodeResult(tuple(targets), tuple(issues))

    @staticmethod
    def _validate_pair(
        target: str,
        down_key: str,
        down_value: object,
        up_key: str,
        up_value: object,
    ) -> tuple[StandardLoraDecodeIssue, ...]:
        """Validate one complete pair without moving or modifying its tensors."""

        issues: list[StandardLoraDecodeIssue] = []
        for key, value in ((down_key, down_value), (up_key, up_value)):
            if not isinstance(value, torch.Tensor):
                issues.append(
                    StandardLoraDecodeIssue(key, "adapter value must be a tensor")
                )
                continue
            if value.device.type != "cpu":
                issues.append(
                    StandardLoraDecodeIssue(
                        key, "adapter tensor must be on CPU during admission"
                    )
                )
            if value.ndim != 2:
                issues.append(
                    StandardLoraDecodeIssue(key, "adapter tensor must be rank 2")
                )
            if not value.is_floating_point():
                issues.append(
                    StandardLoraDecodeIssue(
                        key, "adapter tensor must be floating point"
                    )
                )
            elif not bool(torch.isfinite(value).all().item()):
                issues.append(
                    StandardLoraDecodeIssue(
                        key, "adapter tensor must contain only finite values"
                    )
                )
        if (
            issues
            or not isinstance(down_value, torch.Tensor)
            or not isinstance(up_value, torch.Tensor)
        ):
            return tuple(issues)
        if down_value.ndim != 2 or up_value.ndim != 2:
            return tuple(issues)
        if down_value.dtype != up_value.dtype:
            issues.append(
                StandardLoraDecodeIssue(target, "adapter A/B dtypes must match")
            )
        if (
            down_value.shape[0] <= 0
            or down_value.shape[1] <= 0
            or up_value.shape[0] <= 0
        ):
            issues.append(
                StandardLoraDecodeIssue(target, "adapter dimensions must be positive")
            )
        if down_value.shape[0] != up_value.shape[1]:
            issues.append(
                StandardLoraDecodeIssue(
                    target, "adapter A/B rank dimensions do not match"
                )
            )
        return tuple(issues)

    @staticmethod
    def _decode_intrinsic_scale(
        alpha_entry: tuple[str, object, str] | None,
        rank: int,
    ) -> tuple[float, tuple[StandardLoraDecodeIssue, ...]]:
        """Convert one optional scalar alpha into its alpha/rank multiplier."""

        if alpha_entry is None:
            return 1.0, ()
        key, value, _layout = alpha_entry
        if not isinstance(value, torch.Tensor):
            return 1.0, (
                StandardLoraDecodeIssue(key, "adapter alpha must be a tensor"),
            )
        if value.device.type != "cpu":
            return 1.0, (
                StandardLoraDecodeIssue(
                    key,
                    "adapter alpha must be on CPU during admission",
                ),
            )
        if value.numel() != 1 or not value.is_floating_point():
            return 1.0, (
                StandardLoraDecodeIssue(
                    key,
                    "adapter alpha must be one floating-point scalar",
                ),
            )
        alpha = float(value.item())
        if not math.isfinite(alpha):
            return 1.0, (StandardLoraDecodeIssue(key, "adapter alpha must be finite"),)
        return alpha / rank, ()


STANDARD_LORA_ADAPTER_DECODER = StandardLoraAdapterDecoder()
