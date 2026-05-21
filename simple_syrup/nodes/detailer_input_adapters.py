# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Input normalization helpers shared by SEGS detailer nodes."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..domain.conditioning_batch import ConditioningBatch
from ..domain.segs import NativeSegs
from ..masking.segs_mask_ops import iter_single_images, validate_image_batch


def image_inputs(image: object, operation_name: str) -> tuple[torch.Tensor, ...]:
    """Normalize direct or list-mode image inputs into one-image tensors."""

    if isinstance(image, list):
        if not image:
            raise ValueError(f"{operation_name} requires an image input.")
        images: list[torch.Tensor] = []
        for item in image:
            images.extend(
                iter_single_images(validate_image_batch(item, operation_name))
            )
        return tuple(images)
    return iter_single_images(validate_image_batch(image, operation_name))


def single_input(
    value: object,
    name: str,
    list_mode: bool,
    operation_name: str,
) -> object:
    """Return one scalar input from a direct or Comfy list-mode value."""

    if not list_mode:
        return value
    if not isinstance(value, list):
        return value
    if len(value) != 1:
        raise ValueError(
            f"{operation_name} requires one '{name}' value; received {len(value)}."
        )
    return value[0]


def float_input(
    value: object,
    name: str,
    list_mode: bool,
    operation_name: str,
) -> float:
    """Return a float scalar from a direct or list-mode input."""

    raw_value = single_input(value, name, list_mode, operation_name)
    if isinstance(raw_value, (int, float, str)):
        return float(raw_value)
    raise TypeError(f"{operation_name} requires '{name}' to be a float.")


def int_input(
    value: object,
    name: str,
    list_mode: bool,
    operation_name: str,
) -> int:
    """Return an integer scalar from a direct or list-mode input."""

    raw_value = single_input(value, name, list_mode, operation_name)
    if isinstance(raw_value, (int, float, str)):
        return int(raw_value)
    raise TypeError(f"{operation_name} requires '{name}' to be an int.")


def str_input(
    value: object,
    name: str,
    list_mode: bool,
    operation_name: str,
) -> str:
    """Return a string scalar from a direct or list-mode input."""

    raw_value = single_input(value, name, list_mode, operation_name)
    if isinstance(raw_value, str):
        return raw_value
    raise TypeError(f"{operation_name} requires '{name}' to be a string.")


def bool_input(
    value: object,
    name: str,
    list_mode: bool,
    operation_name: str,
) -> bool:
    """Return a boolean scalar from a direct or list-mode input."""

    raw_value = single_input(value, name, list_mode, operation_name)
    if isinstance(raw_value, bool):
        return raw_value
    raise TypeError(f"{operation_name} requires '{name}' to be a boolean.")


def validate_image_segs_pairing(
    images: tuple[torch.Tensor, ...],
    segs_group: tuple[NativeSegs, ...],
    operation_name: str,
) -> None:
    """Ensure image and SEGS group cardinality is unambiguous."""

    image_count = len(images)
    segs_count = len(segs_group)
    if image_count == segs_count:
        return
    raise ValueError(
        f"{operation_name} requires one SEGS payload per image when "
        f"processing an image batch; received {image_count} images and "
        f"{segs_count} SEGS payloads."
    )


def conditioning_batch_group(
    value: object,
    expected_count: int,
    operation_name: str,
) -> tuple[ConditioningBatch, ...]:
    """Return one regional conditioning batch per SEGS payload."""

    if expected_count == 1 and isinstance(value, ConditioningBatch):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(
            f"{operation_name} requires one region_positive CONDITIONING_BATCH "
            "per SEGS payload."
        )
    if len(value) != expected_count:
        raise ValueError(
            f"{operation_name} requires one region_positive CONDITIONING_BATCH "
            f"per SEGS payload; received {len(value)} conditioning batches for "
            f"{expected_count} SEGS payloads."
        )

    batches: list[ConditioningBatch] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, ConditioningBatch):
            raise TypeError(
                f"{operation_name} requires region_positive item {index} to be "
                "CONDITIONING_BATCH."
            )
        batches.append(item)
    return tuple(batches)
