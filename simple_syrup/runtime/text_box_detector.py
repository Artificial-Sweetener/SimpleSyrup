# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt GroundingDINO-compatible objects into text-prompt box detection."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from PIL import Image

from ..domain.segs import BoundingBox
from .grounding_dino_loader import GROUNDING_DINO_RUNTIME_PACKAGE
from .loaded_models import LoadedGroundingDINOModel, unwrap_grounding_dino_model
from .model_device_manager import (
    LoadedManagedModel,
    TorchModelDeviceManager,
    external_model_inference,
)
from .progress import create_comfy_phase_progress


@dataclass(frozen=True)
class TextBoxDetection:
    """Represent one prompt box detection with confidence."""

    bbox: BoundingBox
    confidence: float


class TextBoxDetector(Protocol):
    """Detect pixel-space boxes from a text prompt."""

    def detect(
        self,
        grounding_dino_model: object,
        image: torch.Tensor,
        prompt: str,
        threshold: float,
        execution_device: str,
    ) -> tuple[TextBoxDetection, ...]:
        """Return filtered prompt detections in pixel coordinates."""


class GroundingDINOTextBoxDetector:
    """Detect prompt boxes using protocol objects or raw GroundingDINO models."""

    def detect(
        self,
        grounding_dino_model: object,
        image: torch.Tensor,
        prompt: str,
        threshold: float,
        execution_device: str,
    ) -> tuple[TextBoxDetection, ...]:
        """Return filtered prompt detections in pixel coordinates."""

        progress = create_comfy_phase_progress(
            operation="grounding_dino_inference",
            subject=_grounding_dino_subject(grounding_dino_model),
            total_phases=3,
        )
        progress.advance("preparing_device")
        try:
            progress.advance("running_inference")
            detections = self._detect(
                grounding_dino_model,
                image,
                prompt,
                threshold,
                execution_device,
            )
        except Exception:
            progress.advance("failed")
            raise
        progress.advance("completed")
        return detections

    def _detect(
        self,
        grounding_dino_model: object,
        image: torch.Tensor,
        prompt: str,
        threshold: float,
        execution_device: str,
    ) -> tuple[TextBoxDetection, ...]:
        """Run GroundingDINO adaptation after progress ownership is established."""

        if (
            isinstance(grounding_dino_model, LoadedGroundingDINOModel)
            and grounding_dino_model.managed_model is not None
        ):
            with _managed_inference(
                grounding_dino_model,
                execution_device,
            ) as loaded:
                return _detect_with_raw_grounding_dino(
                    loaded.model,
                    image,
                    prompt,
                    threshold,
                    loaded.device,
                )

        model = unwrap_grounding_dino_model(grounding_dino_model)
        predict_boxes = getattr(model, "predict_boxes", None)
        if callable(predict_boxes):
            boxes = torch.as_tensor(
                predict_boxes(image, prompt, threshold),
                dtype=torch.float32,
            )
            boxes = _validate_boxes(boxes)
            scores = torch.ones((int(boxes.shape[0]),), dtype=torch.float32)
            return _detections_from_boxes(
                boxes,
                scores,
                image_height=int(image.shape[0]),
                image_width=int(image.shape[1]),
            )

        if callable(model):
            with external_model_inference(model, execution_device) as loaded:
                return _detect_with_raw_grounding_dino(
                    loaded.model,
                    image,
                    prompt,
                    threshold,
                    loaded.device,
                )

        raise TypeError(
            "GroundingDINO model is not compatible with Prompt SEGS w/ SAM. Expected "
            "GROUNDING_DINO_MODEL or DINO_MODEL with predict_boxes(...) or a callable "
            "GroundingDINO model."
        )


def _grounding_dino_subject(model: object) -> str:
    """Return non-sensitive model identity for progress diagnostics."""

    if isinstance(model, LoadedGroundingDINOModel):
        return model.model_id
    model_name = getattr(model, "model_name", None)
    return (
        model_name
        if isinstance(model_name, str) and model_name
        else type(model).__name__
    )


def _detect_with_raw_grounding_dino(
    model: Any,
    image: torch.Tensor,
    prompt: str,
    threshold: float,
    device: torch.device,
) -> tuple[TextBoxDetection, ...]:
    """Run GroundingDINO inference for a raw callable model object."""

    try:
        transforms_module = importlib.import_module(
            f"{GROUNDING_DINO_RUNTIME_PACKAGE}.datasets.transforms"
        )
    except ImportError as error:
        raise RuntimeError(
            "GroundingDINO inference dependencies are missing."
        ) from error

    pil_image = _tensor_to_pil(image)
    transform = transforms_module.Compose(
        [
            transforms_module.RandomResize([800], max_size=1333),
            transforms_module.ToTensor(),
            transforms_module.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    dino_image, _unused = transform(pil_image.convert("RGB"), None)
    caption = prompt.lower().strip()
    if not caption.endswith("."):
        caption = f"{caption}."
    with torch.no_grad():
        outputs: dict[str, torch.Tensor] = model(
            dino_image.to(device)[None],
            captions=[caption],
        )
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]
    scores = logits.max(dim=1)[0]
    keep = scores > threshold
    boxes = boxes[keep].cpu()
    scores = scores[keep].cpu()
    width, height = pil_image.size
    for index in range(boxes.size(0)):
        boxes[index] = boxes[index] * torch.tensor([width, height, width, height])
        boxes[index][:2] -= boxes[index][2:] / 2
        boxes[index][2:] += boxes[index][:2]
    return _detections_from_boxes(
        _validate_boxes(boxes),
        scores.float(),
        image_height=height,
        image_width=width,
    )


def _validate_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Validate and normalize box tensor shape."""

    if boxes.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(
            f"GroundingDINO model returned invalid boxes shape: {tuple(boxes.shape)}."
        )
    return boxes.float().cpu()


def _detections_from_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    image_height: int,
    image_width: int,
) -> tuple[TextBoxDetection, ...]:
    """Convert XYXY boxes to clipped prompt detections."""

    if boxes.numel() == 0:
        return ()
    if scores.ndim != 1 or int(scores.shape[0]) != int(boxes.shape[0]):
        raise ValueError(
            "GroundingDINO model returned confidence scores that do not match boxes."
        )

    detections: list[TextBoxDetection] = []
    for box, score in zip(boxes, scores, strict=True):
        left = max(0, min(image_width, int(round(float(box[0].item())))))
        top = max(0, min(image_height, int(round(float(box[1].item())))))
        right = max(0, min(image_width, int(round(float(box[2].item())))))
        bottom = max(0, min(image_height, int(round(float(box[3].item())))))
        if right <= left or bottom <= top:
            continue
        detections.append(
            TextBoxDetection(
                bbox=BoundingBox(left, top, right, bottom),
                confidence=float(score.item()),
            )
        )
    return tuple(detections)


def _tensor_to_pil(sample: torch.Tensor) -> Image.Image:
    """Convert one ComfyUI HWC image tensor to RGB PIL image."""

    import numpy as np

    array = sample.detach().cpu().float().clamp(0.0, 1.0).numpy()
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


@contextmanager
def _managed_inference(
    loaded_model: LoadedGroundingDINOModel,
    execution_device: str,
) -> Iterator[LoadedManagedModel]:
    """Open the managed GroundingDINO inference context."""

    managed_model = loaded_model.managed_model
    if managed_model is None:
        raise TypeError("Loaded GroundingDINO model is missing device metadata.")
    manager = TorchModelDeviceManager()
    with manager.inference(managed_model, execution_device) as loaded:
        yield loaded
