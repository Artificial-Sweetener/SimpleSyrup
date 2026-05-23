# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Ultralytics inference result parsing for SimpleSyrup detection services."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..domain.segs import BoundingBox
from ..masking.segs_mask_ops import normalize_mask
from .ultralytics_loader import UltralyticsDetectorModel


@dataclass(frozen=True)
class UltralyticsDetection:
    """Represent one parsed Ultralytics prediction."""

    bbox: BoundingBox
    confidence: float
    label: str
    mask: torch.Tensor | None


def run_ultralytics_detection(
    detector_model: UltralyticsDetectorModel,
    image: torch.Tensor,
    threshold: float,
    prefer_segmentation: bool,
) -> tuple[UltralyticsDetection, ...]:
    """Run Ultralytics inference and parse detections into native values."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    raw_results = detector_model.model(
        _image_to_pil(image),
        conf=threshold,
        verbose=False,
    )
    result = _first_result(raw_results)
    if result is None:
        return ()
    return parse_ultralytics_result(
        result=result,
        image_height=int(image.shape[1]),
        image_width=int(image.shape[2]),
        class_names=detector_model.names,
        prefer_segmentation=prefer_segmentation,
    )


def parse_ultralytics_result(
    result: object,
    image_height: int,
    image_width: int,
    class_names: dict[int, str],
    prefer_segmentation: bool,
) -> tuple[UltralyticsDetection, ...]:
    """Parse one Ultralytics result object without depending on its concrete type."""

    boxes_object = getattr(result, "boxes", None)
    if boxes_object is None:
        return ()

    boxes = _tensor_attr(boxes_object, "xyxy")
    if boxes.numel() == 0:
        return ()
    confidences = _tensor_attr(boxes_object, "conf")
    classes = _tensor_attr(boxes_object, "cls")
    masks = (
        _result_masks(result, image_height, image_width) if prefer_segmentation else []
    )

    names = _result_names(result, class_names)
    detections: list[UltralyticsDetection] = []
    for index, raw_box in enumerate(boxes):
        bbox = _bbox_from_tensor(raw_box, image_height, image_width)
        confidence = (
            float(confidences[index].item()) if confidences.numel() > index else 1.0
        )
        class_index = int(classes[index].item()) if classes.numel() > index else -1
        detections.append(
            UltralyticsDetection(
                bbox=bbox,
                confidence=confidence,
                label=names.get(class_index, str(class_index)),
                mask=masks[index] if index < len(masks) else None,
            )
        )
    return tuple(detections)


def _image_to_pil(image: torch.Tensor) -> object:
    """Convert a single ComfyUI BHWC image tensor to a PIL image."""

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ultralytics detection requires Pillow in the ComfyUI venv."
        ) from exc

    array = image[0, :, :, :3].detach().cpu().clamp(0.0, 1.0).mul(255.0).byte().numpy()
    return Image.fromarray(array, mode="RGB")


def _first_result(raw_results: object) -> object | None:
    """Return the first Ultralytics result from a result container."""

    if isinstance(raw_results, Sequence) and not isinstance(raw_results, (str, bytes)):
        return raw_results[0] if raw_results else None
    return raw_results


def _tensor_attr(source: object, name: str) -> torch.Tensor:
    """Read a tensor-like result attribute as a CPU tensor."""

    value = getattr(source, name, None)
    if value is None:
        return torch.empty((0,), dtype=torch.float32)
    return torch.as_tensor(value).detach().cpu()


def _result_masks(
    result: object,
    image_height: int,
    image_width: int,
) -> list[torch.Tensor]:
    """Return normalized segmentation masks from a result object."""

    masks_object = getattr(result, "masks", None)
    if masks_object is None:
        return []
    data = getattr(masks_object, "data", None)
    if data is None:
        return []
    tensor = torch.as_tensor(data).detach().cpu().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError("Ultralytics masks must be shaped as BHW.")
    return [
        normalize_mask(
            _remove_aspect_padding(mask, image_height, image_width),
            image_height,
            image_width,
        )
        for mask in tensor
    ]


def _remove_aspect_padding(
    mask: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Remove Ultralytics mask padding before resizing to image geometry."""

    mask_height = int(mask.shape[0])
    mask_width = int(mask.shape[1])
    mask_ratio = mask_height / mask_width
    image_ratio = image_height / image_width
    if mask_ratio == image_ratio:
        return mask
    if mask_ratio > image_ratio:
        height_gap = int((mask_ratio - image_ratio) * mask_height)
        if height_gap <= 0 or height_gap * 2 >= mask_height:
            return mask
        return mask[height_gap : mask_height - height_gap, :]

    width_ratio = mask_width / mask_height
    image_width_ratio = image_width / image_height
    width_gap = int((width_ratio - image_width_ratio) * mask_width)
    if width_gap <= 0 or width_gap * 2 >= mask_width:
        return mask
    return mask[:, width_gap : mask_width - width_gap]


def _result_names(result: object, fallback: dict[int, str]) -> dict[int, str]:
    """Return class names from the result object or detector model metadata."""

    names = getattr(result, "names", None)
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return fallback


def _bbox_from_tensor(
    raw_box: torch.Tensor,
    image_height: int,
    image_width: int,
) -> BoundingBox:
    """Clamp one xyxy tensor into a valid integer bounding box."""

    values = raw_box.tolist()
    if len(values) != 4:
        raise ValueError("Ultralytics bbox must contain four xyxy coordinates.")
    left = max(0, min(image_width - 1, int(round(float(values[0])))))
    top = max(0, min(image_height - 1, int(round(float(values[1])))))
    right = max(left + 1, min(image_width, int(round(float(values[2])))))
    bottom = max(top + 1, min(image_height, int(round(float(values[3])))))
    return BoundingBox(left, top, right, bottom)
