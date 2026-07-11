# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt SAM-compatible model objects into box segmentation."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray

from .loaded_models import LoadedSAMModel, unwrap_sam_model
from .model_device_manager import TorchModelDeviceManager, external_model_inference
from .progress import create_comfy_phase_progress
from .sam_loader import SAM_HQ_RUNTIME_PACKAGE


class SAMBoxSegmenter(Protocol):
    """Segment image boxes into one mask per box."""

    def segment_boxes(
        self,
        sam_model: object,
        image: torch.Tensor,
        boxes: torch.Tensor,
        threshold: float,
        execution_device: str,
    ) -> tuple[torch.Tensor, ...]:
        """Return one HW mask per input box."""


class SAMModelSegmenter:
    """Segment boxes with SimpleSyrup, Impact, or raw SAM model objects."""

    def segment_boxes(
        self,
        sam_model: object,
        image: torch.Tensor,
        boxes: torch.Tensor,
        threshold: float,
        execution_device: str,
    ) -> tuple[torch.Tensor, ...]:
        """Return one HW mask per provided box."""

        if boxes.numel() == 0:
            return ()

        progress = create_comfy_phase_progress(
            operation="sam_inference",
            subject=_sam_subject(sam_model),
            total_phases=3,
        )
        progress.advance("preparing_device")
        try:
            progress.advance("running_inference")
            masks = self._segment_boxes(
                sam_model,
                image,
                boxes,
                threshold,
                execution_device,
            )
        except Exception:
            progress.advance("failed")
            raise
        progress.advance("completed")
        return masks

    def _segment_boxes(
        self,
        sam_model: object,
        image: torch.Tensor,
        boxes: torch.Tensor,
        threshold: float,
        execution_device: str,
    ) -> tuple[torch.Tensor, ...]:
        """Run SAM adaptation after progress ownership is established."""

        model = unwrap_sam_model(sam_model)
        image_array = _tensor_to_rgb_array(image)
        wrapper = getattr(model, "sam_wrapper", None)
        if wrapper is not None:
            return _predict_with_wrapper(wrapper, image_array, boxes, threshold)

        predict = getattr(model, "predict", None)
        if callable(predict):
            return _predict_with_wrapper(model, image_array, boxes, threshold)

        if (
            isinstance(sam_model, LoadedSAMModel)
            and sam_model.managed_model is not None
        ):
            use_hq_predictor = _uses_sam_hq_predictor(sam_model)
            with TorchModelDeviceManager().inference(
                sam_model.managed_model,
                execution_device,
            ) as loaded:
                return _predict_with_raw_segment_anything(
                    loaded.model,
                    image_array,
                    boxes,
                    loaded.device,
                    use_hq_predictor=use_hq_predictor,
                )

        with external_model_inference(model, execution_device) as loaded:
            return _predict_with_raw_segment_anything(
                loaded.model,
                image_array,
                boxes,
                loaded.device,
                use_hq_predictor=_uses_sam_hq_predictor(model),
            )


def _sam_subject(model: object) -> str:
    """Return non-sensitive model identity for progress diagnostics."""

    if isinstance(model, LoadedSAMModel):
        return model.model_id
    model_name = getattr(model, "model_name", None)
    return (
        model_name
        if isinstance(model_name, str) and model_name
        else type(model).__name__
    )


def _predict_with_wrapper(
    wrapper: object,
    image_array: NDArray[np.uint8],
    boxes: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, ...]:
    """Segment boxes with Impact-style wrapper objects."""

    prepare_device = getattr(wrapper, "prepare_device", None)
    release_device = getattr(wrapper, "release_device", None)
    if callable(prepare_device):
        prepare_device()
    try:
        masks: list[torch.Tensor] = []
        predict = getattr(wrapper, "predict", None)
        if not callable(predict):
            raise TypeError(
                "SAM_MODEL is not compatible with Prompt SEGS w/ SAM. Expected an "
                "object with sam_wrapper, predict(...), or a raw Segment Anything "
                "model."
            )
        for box in boxes.cpu().tolist():
            predicted = predict(image_array, [], [], box, threshold)
            candidate_masks = _normalize_predicted_masks(predicted)
            if candidate_masks:
                masks.append(
                    torch.stack(candidate_masks).float().amax(dim=0).clamp(0.0, 1.0)
                )
            else:
                masks.append(torch.zeros(image_array.shape[:2], dtype=torch.float32))
        return tuple(masks)
    finally:
        if callable(release_device):
            release_device()


def _predict_with_raw_segment_anything(
    model: object,
    image_array: NDArray[np.uint8],
    boxes: torch.Tensor,
    device: torch.device,
    *,
    use_hq_predictor: bool,
) -> tuple[torch.Tensor, ...]:
    """Segment boxes with a Segment Anything predictor API."""

    predictor = _create_raw_sam_predictor(model, use_hq_predictor)
    predictor.set_image(image_array)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes.cpu(),
        image_array.shape[:2],
    )
    masks, _scores, _logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    box_masks = cast(torch.Tensor, masks[:, 0, :, :].detach().cpu().float())
    return tuple(mask.clamp(0.0, 1.0) for mask in box_masks)


def _create_raw_sam_predictor(model: object, use_hq_predictor: bool) -> Any:
    """Create the predictor matching the loaded SAM model family."""

    try:
        if use_hq_predictor:
            predictor_module = importlib.import_module(
                f"{SAM_HQ_RUNTIME_PACKAGE}.predictor"
            )
            predictor_class = cast(Any, predictor_module).SamPredictorHQ
            return predictor_class(model, True)

        segment_anything = importlib.import_module("segment_anything")
        return segment_anything.SamPredictor(model)
    except ImportError as error:
        raise RuntimeError(
            "SAM_MODEL is not compatible with Prompt SEGS w/ SAM. Expected an object "
            "with sam_wrapper, predict(...), or a raw Segment Anything model with "
            "its required runtime package installed."
        ) from error


def _uses_sam_hq_predictor(model: object) -> bool:
    """Return whether a model container or raw object needs the HQ predictor."""

    if isinstance(model, LoadedSAMModel):
        return model.model_id.startswith("sam_hq")
    model_name = getattr(model, "model_name", "")
    return isinstance(model_name, str) and model_name.startswith("sam_hq")


def _normalize_predicted_masks(predicted: object) -> list[torch.Tensor]:
    """Normalize wrapper prediction output into HW float tensors."""

    if predicted is None:
        return []
    raw_items = predicted if isinstance(predicted, list | tuple) else [predicted]
    masks: list[torch.Tensor] = []
    for item in raw_items:
        tensor = torch.as_tensor(item, dtype=torch.float32).detach().cpu()
        while tensor.ndim > 2 and 1 in tensor.shape:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 3:
            tensor = tensor.amax(dim=0)
        if tensor.ndim != 2:
            raise ValueError(
                f"SAM_MODEL returned an invalid mask shape: {tuple(tensor.shape)}."
            )
        masks.append(tensor.clamp(0.0, 1.0))
    return masks


def _tensor_to_rgb_array(image: torch.Tensor) -> NDArray[np.uint8]:
    """Convert a ComfyUI HWC image sample to uint8 RGB."""

    array = image.detach().cpu().float().clamp(0.0, 1.0).numpy()
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return (array * 255.0).round().astype(np.uint8)
