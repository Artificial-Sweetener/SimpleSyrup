# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Latent metadata reporting for ComfyUI diagnostic nodes."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


class LatentDiagnosticsService:
    """Build deterministic reports for ComfyUI latent dictionaries."""

    def describe(self, latent: Mapping[str, Any]) -> str:
        """Return a readable latent metadata report without sampling tensor values."""

        LOGGER.debug(
            "describing_latent",
            extra={
                "operation": "latent_diagnostics.describe",
                "latent_key_count": len(latent),
            },
        )
        lines = [
            "SimpleSyrup Latent Diagnostics",
            f"latent_type: {type(latent).__module__}.{type(latent).__qualname__}",
            f"latent_keys: {_format_keys(latent)}",
        ]

        samples = latent.get("samples")
        lines.extend(_describe_samples(samples))

        other_keys = [key for key in sorted(latent) if key != "samples"]
        if other_keys:
            lines.append("other_entries:")
            for key in other_keys:
                lines.append(f"  {key}: {_describe_value(latent[key])}")
        else:
            lines.append("other_entries: none")

        return "\n".join(lines)


def _format_keys(latent: Mapping[str, Any]) -> str:
    """Return sorted latent keys in a compact display form."""

    return "[" + ", ".join(sorted(latent)) + "]"


def _describe_samples(samples: object) -> list[str]:
    """Return report lines for the primary latent samples entry."""

    if not isinstance(samples, torch.Tensor):
        return [
            "samples: missing or not a torch.Tensor",
            f"samples_type: {_type_name(samples)}",
            "mixture_of_diffusers_current_compatible: no",
            "compatibility_reason: samples must be a torch.Tensor.",
        ]

    is_nested = bool(getattr(samples, "is_nested", False))
    shape = tuple(int(dim) for dim in samples.shape)
    lines = [
        "samples:",
        f"  type: {_type_name(samples)}",
        f"  shape: {list(shape)}",
        f"  ndim: {samples.ndim}",
        f"  dtype: {samples.dtype}",
        f"  device: {samples.device}",
        f"  layout: {samples.layout}",
        f"  is_nested: {is_nested}",
        f"  is_sparse: {samples.is_sparse}",
        f"  requires_grad: {samples.requires_grad}",
        f"  is_contiguous: {_safe_bool(samples.is_contiguous)}",
        f"  numel: {_safe_int(samples.numel)}",
        f"  stride: {_safe_sequence(samples.stride)}",
    ]
    lines.extend(_describe_shape_interpretation(samples, is_nested))
    return lines


def _describe_shape_interpretation(
    samples: torch.Tensor,
    is_nested: bool,
) -> list[str]:
    """Return Mixture-of-Diffusers-relevant shape interpretation lines."""

    if samples.ndim >= 2:
        lines = [
            "spatial_last_dims:",
            f"  height: {int(samples.shape[-2])}",
            f"  width: {int(samples.shape[-1])}",
        ]
    else:
        lines = ["spatial_last_dims: unavailable"]

    if is_nested:
        lines.extend(
            [
                "mixture_of_diffusers_current_compatible: no",
                "compatibility_reason: samples is a nested tensor.",
            ]
        )
    elif samples.ndim == 4:
        lines.extend(
            [
                "bchw_interpretation:",
                f"  batch: {int(samples.shape[0])}",
                f"  channels: {int(samples.shape[1])}",
                f"  height: {int(samples.shape[2])}",
                f"  width: {int(samples.shape[3])}",
                "mixture_of_diffusers_current_compatible: yes",
            ]
        )
    elif samples.ndim == 5 and int(samples.shape[2]) == 1:
        lines.extend(
            [
                "bcdhw_interpretation:",
                f"  batch: {int(samples.shape[0])}",
                f"  channels: {int(samples.shape[1])}",
                f"  depth: {int(samples.shape[2])}",
                f"  height: {int(samples.shape[3])}",
                f"  width: {int(samples.shape[4])}",
                "mixture_of_diffusers_current_compatible: yes",
            ]
        )
    elif samples.ndim == 5:
        lines.extend(
            [
                "bcdhw_interpretation:",
                f"  batch: {int(samples.shape[0])}",
                f"  channels: {int(samples.shape[1])}",
                f"  depth: {int(samples.shape[2])}",
                f"  height: {int(samples.shape[3])}",
                f"  width: {int(samples.shape[4])}",
                "mixture_of_diffusers_current_compatible: no",
                (
                    "compatibility_reason: current Mixture of Diffusers sampler "
                    "expects 5D samples to use a singleton depth axis."
                ),
            ]
        )
    else:
        lines.extend(
            [
                "bchw_interpretation: unavailable",
                "mixture_of_diffusers_current_compatible: no",
                (
                    "compatibility_reason: current Mixture of Diffusers sampler "
                    "expects non-nested 4D BCHW or singleton-depth 5D BCDHW samples."
                ),
            ]
        )
    return lines


def _describe_value(value: object) -> str:
    """Return one-line metadata for a non-primary latent entry."""

    if isinstance(value, torch.Tensor):
        return (
            f"{_type_name(value)} shape={list(value.shape)} ndim={value.ndim} "
            f"dtype={value.dtype} device={value.device}"
        )
    if isinstance(value, list | tuple):
        return f"{_type_name(value)} len={len(value)}"
    if isinstance(value, Mapping):
        return f"{_type_name(value)} keys={_format_mapping_keys(value)}"
    if value is None:
        return "None"
    return _type_name(value)


def _format_mapping_keys(value: Mapping[object, object]) -> str:
    """Return stable display text for mapping keys."""

    return "[" + ", ".join(sorted(str(key) for key in value)) + "]"


def _type_name(value: object) -> str:
    """Return a fully qualified type name for diagnostics."""

    return f"{type(value).__module__}.{type(value).__qualname__}"


def _safe_bool(method: Any) -> str:
    """Return a boolean method result or a diagnostic error marker."""

    try:
        return str(bool(method()))
    except RuntimeError as error:
        return f"unavailable ({error.__class__.__name__}: {error})"


def _safe_int(method: Any) -> str:
    """Return an integer method result or a diagnostic error marker."""

    try:
        return str(int(method()))
    except RuntimeError as error:
        return f"unavailable ({error.__class__.__name__}: {error})"


def _safe_sequence(method: Any) -> str:
    """Return an integer sequence method result or a diagnostic error marker."""

    try:
        return str([int(value) for value in method()])
    except RuntimeError as error:
        return f"unavailable ({error.__class__.__name__}: {error})"
