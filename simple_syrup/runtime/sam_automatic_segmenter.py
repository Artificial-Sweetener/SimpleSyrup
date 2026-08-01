# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run unprompted automatic segmentation with supported SAM model families."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import torch

from .loaded_models import LoadedSAMModel, unwrap_sam_model
from .model_device_manager import TorchModelDeviceManager, external_model_inference
from .sam_loader import SAM_HQ_RUNTIME_PACKAGE


@dataclass(frozen=True)
class AutomaticSAMMask:
    """Describe one unprompted mask emitted by a SAM-compatible runtime."""

    mask: torch.Tensor
    confidence: float
    label: str | None = None


class SAMAutomaticSegmenter(Protocol):
    """Generate unprompted masks from one SAM-compatible model and image."""

    def segment_all(
        self,
        sam_model: object,
        image: torch.Tensor,
        execution_device: str = "auto",
    ) -> tuple[AutomaticSAMMask, ...]:
        """Return source-image masks with optional confidence and labels."""


class SAMModelAutomaticSegmenter:
    """Adapt SimpleSyrup SAM models to their unprompted segmentation APIs."""

    def segment_all(
        self,
        sam_model: object,
        image: torch.Tensor,
        execution_device: str = "auto",
    ) -> tuple[AutomaticSAMMask, ...]:
        """Return automatic masks for one single-image BHWC tensor."""

        image_array = _tensor_to_rgb_array(image)
        model = unwrap_sam_model(sam_model)
        if _is_fast_sam(sam_model, model):
            return self._segment_fast_sam(
                sam_model=sam_model,
                model=model,
                image_array=image_array,
                execution_device=execution_device,
            )
        return self._segment_segment_anything(
            sam_model=sam_model,
            model=model,
            image_array=image_array,
            execution_device=execution_device,
        )

    def _segment_fast_sam(
        self,
        *,
        sam_model: object,
        model: object,
        image_array: np.ndarray[Any, Any],
        execution_device: str,
    ) -> tuple[AutomaticSAMMask, ...]:
        """Run FastSAM's unprompted everything-mask path."""

        if (
            isinstance(sam_model, LoadedSAMModel)
            and sam_model.managed_model is not None
        ):
            with TorchModelDeviceManager().inference(
                sam_model.managed_model,
                execution_device,
            ) as loaded:
                return _fast_sam_masks(loaded.model, image_array, loaded.device)
        with external_model_inference(model, execution_device) as loaded:
            return _fast_sam_masks(loaded.model, image_array, loaded.device)

    def _segment_segment_anything(
        self,
        *,
        sam_model: object,
        model: object,
        image_array: np.ndarray[Any, Any],
        execution_device: str,
    ) -> tuple[AutomaticSAMMask, ...]:
        """Run Segment Anything automatic-mask generation under device management."""

        if (
            isinstance(sam_model, LoadedSAMModel)
            and sam_model.managed_model is not None
        ):
            with TorchModelDeviceManager().inference(
                sam_model.managed_model,
                execution_device,
            ) as loaded:
                return _segment_anything_masks(
                    loaded.model,
                    image_array,
                    use_hq_generator=_uses_sam_hq_generator(sam_model),
                )
        with external_model_inference(model, execution_device) as loaded:
            return _segment_anything_masks(
                loaded.model,
                image_array,
                use_hq_generator=_uses_sam_hq_generator(model),
            )


def _tensor_to_rgb_array(image: torch.Tensor) -> np.ndarray[Any, Any]:
    """Convert one BHWC ComfyUI image into a uint8 RGB array."""

    if image.ndim != 4 or int(image.shape[0]) != 1:
        raise ValueError(
            "SAM automatic segmentation requires one BHWC image at a time."
        )
    sample = image[0].detach().cpu().float().clamp(0.0, 1.0).numpy()
    channels = int(sample.shape[-1])
    if channels == 1:
        sample = np.repeat(sample, 3, axis=-1)
    elif channels >= 3:
        sample = sample[..., :3]
    else:
        raise ValueError(
            "SAM automatic segmentation requires at least one image channel."
        )
    return (sample * 255.0).round().astype(np.uint8)


def _is_fast_sam(container: object, model: object) -> bool:
    """Return whether a loaded or external model uses FastSAM's API."""

    if isinstance(container, LoadedSAMModel):
        return container.model_id.startswith("fast_sam")
    return type(model).__name__ == "FastSAM"


def _fast_sam_masks(
    model: object,
    image_array: np.ndarray[Any, Any],
    device: torch.device,
) -> tuple[AutomaticSAMMask, ...]:
    """Extract FastSAM masks and scores from one Ultralytics result."""

    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise TypeError("FastSAM model does not expose the required predict method.")
    results = predict(
        image_array,
        imgsz=max(image_array.shape[:2]),
        retina_masks=True,
        verbose=False,
        device=str(device),
    )
    if not isinstance(results, list | tuple) or not results:
        return ()
    result = results[0]
    masks_container = getattr(result, "masks", None)
    mask_data = getattr(masks_container, "data", None)
    if not isinstance(mask_data, torch.Tensor):
        return ()
    boxes = getattr(result, "boxes", None)
    confidences = getattr(boxes, "conf", None)
    entries: list[AutomaticSAMMask] = []
    for index, mask in enumerate(mask_data.detach().cpu()):
        confidence = 1.0
        if isinstance(confidences, torch.Tensor) and index < int(confidences.numel()):
            confidence = float(confidences[index].detach().cpu().item())
        entries.append(
            AutomaticSAMMask(
                mask=mask.float().clamp(0.0, 1.0),
                confidence=_clamp_confidence(confidence),
            )
        )
    return tuple(entries)


def _segment_anything_masks(
    model: object,
    image_array: np.ndarray[Any, Any],
    *,
    use_hq_generator: bool,
) -> tuple[AutomaticSAMMask, ...]:
    """Extract binary masks and predicted quality from automatic generators."""

    generator_class = _automatic_generator_class(use_hq_generator)
    generated = generator_class(model).generate(image_array)
    if not isinstance(generated, list):
        raise TypeError("SAM automatic mask generator returned an invalid result.")
    masks: list[AutomaticSAMMask] = []
    for entry in generated:
        if not isinstance(entry, dict) or "segmentation" not in entry:
            raise ValueError("SAM automatic mask generator returned an invalid mask.")
        raw_confidence = entry.get("predicted_iou", entry.get("stability_score", 1.0))
        confidence = (
            float(raw_confidence) if isinstance(raw_confidence, int | float) else 1.0
        )
        masks.append(
            AutomaticSAMMask(
                mask=torch.as_tensor(entry["segmentation"], dtype=torch.float32)
                .detach()
                .cpu()
                .clamp(0.0, 1.0),
                confidence=_clamp_confidence(confidence),
            )
        )
    return tuple(masks)


def _automatic_generator_class(use_hq_generator: bool) -> type[Any]:
    """Return the automatic generator matching the loaded SAM family."""

    try:
        if use_hq_generator:
            automatic = importlib.import_module(f"{SAM_HQ_RUNTIME_PACKAGE}.automatic")
            return cast(type[Any], automatic.SamAutomaticMaskGeneratorHQ)
        segment_anything = importlib.import_module("segment_anything")
        return cast(type[Any], segment_anything.SamAutomaticMaskGenerator)
    except ImportError as error:
        raise RuntimeError(
            "SAM automatic segmentation requires the matching Segment Anything "
            "runtime. "
            f"Import failed: {error}."
        ) from error


def _uses_sam_hq_generator(model: object) -> bool:
    """Return whether a model container or raw object requires SAM-HQ generation."""

    if isinstance(model, LoadedSAMModel):
        return model.model_id.startswith("sam_hq") or model.model_id == "mobile_sam"
    model_name = getattr(model, "model_name", "")
    return isinstance(model_name, str) and (
        model_name.startswith("sam_hq") or model_name == "mobile_sam"
    )


def _clamp_confidence(value: float) -> float:
    """Return one confidence value constrained to the public SEGS range."""

    return max(0.0, min(1.0, value))
