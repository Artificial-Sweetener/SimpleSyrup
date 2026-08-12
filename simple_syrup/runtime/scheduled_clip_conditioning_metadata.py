# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Preserve installed encoder metadata across Comfy scheduled CLIP encoding."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Protocol, cast


class _EncodingLock(Protocol):
    """Describe the context-manager surface used to serialize shared encoders."""

    def __enter__(self) -> object:
        """Acquire the encoder lock."""

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> bool | None:
        """Release the encoder lock."""


class _MutableEncoder(Protocol):
    """Describe the one callable temporarily observed during encoding."""

    encode_token_weights: Callable[..., object]


class ScheduledClipConditioningMetadataAdapter:
    """Retain extra metadata from each actual scheduled encoder invocation."""

    def __init__(
        self,
        clip: object,
        *,
        encoding_lock: _EncodingLock | None = None,
    ) -> None:
        """Wrap one derived host CLIP without changing its patcher state."""

        if isinstance(clip, ScheduledClipConditioningMetadataAdapter):
            raise TypeError("Scheduled CLIP metadata adapter cannot wrap itself.")
        self._clip = clip
        self._encoding_lock = encoding_lock or cast(_EncodingLock, RLock())

    def __getattr__(self, name: str) -> object:
        """Delegate the unchanged host CLIP surface."""

        return getattr(self._clip, name)

    def clone(
        self,
        disable_dynamic: bool = False,
    ) -> ScheduledClipConditioningMetadataAdapter:
        """Wrap a native clone while retaining shared-encoder serialization."""

        clone = _required_callable(self._clip, "clone", value_name="CLIP")
        derived = clone(disable_dynamic=disable_dynamic)
        if derived is self._clip:
            raise RuntimeError("Scheduled CLIP clone returned its source object.")
        return ScheduledClipConditioningMetadataAdapter(
            derived,
            encoding_lock=self._encoding_lock,
        )

    def encode_from_tokens_scheduled(
        self,
        tokens: object,
        unprojected: bool = False,
        add_dict: dict[str, object] | None = None,
        show_pbar: bool = True,
    ) -> object:
        """Delegate one scheduled encode and restore its omitted extra metadata."""

        model = getattr(self._clip, "cond_stage_model", None)
        if model is None:
            raise TypeError("Scheduled CLIP does not expose cond_stage_model.")
        encode = _required_callable(
            model,
            "encode_token_weights",
            value_name="scheduled CLIP text encoder",
        )
        scheduled_encode = _required_callable(
            self._clip,
            "encode_from_tokens_scheduled",
            value_name="CLIP",
        )
        mutable_model = cast(_MutableEncoder, model)
        captured: list[dict[str, object]] = []

        def capture(*args: object, **kwargs: object) -> object:
            output = encode(*args, **kwargs)
            captured.append(_additional_metadata(output))
            return output

        with self._encoding_lock:
            namespace = vars(model)
            had_override = "encode_token_weights" in namespace
            previous_override = namespace.get("encode_token_weights")
            mutable_model.encode_token_weights = capture
            try:
                result = scheduled_encode(
                    tokens,
                    unprojected=unprojected,
                    add_dict={} if add_dict is None else add_dict,
                    show_pbar=show_pbar,
                )
            finally:
                if had_override:
                    mutable_model.encode_token_weights = cast(
                        Callable[..., object], previous_override
                    )
                else:
                    del mutable_model.encode_token_weights

        outputs = _scheduled_outputs(result)
        if len(outputs) != len(captured):
            raise ValueError(
                "Scheduled CLIP encoder/result cardinality mismatch: "
                f"captured {len(captured)}, observed {len(outputs)}."
            )
        for metadata, extras in zip(outputs, captured, strict=True):
            for key, value in extras.items():
                if key not in metadata:
                    metadata[key] = value
        return result


def _additional_metadata(output: object) -> dict[str, object]:
    """Narrow one installed encoder's optional third return mapping."""

    if not isinstance(output, tuple | list):
        raise TypeError("Scheduled CLIP encoder output must be a sequence.")
    if len(output) < 3:
        return {}
    extra = output[2]
    if not isinstance(extra, dict):
        raise TypeError("Scheduled CLIP encoder extra metadata must be a dictionary.")
    if any(not isinstance(key, str) for key in extra):
        raise TypeError("Scheduled CLIP encoder metadata keys must be strings.")
    return cast(dict[str, object], dict(extra))


def _scheduled_outputs(result: object) -> tuple[dict[str, object], ...]:
    """Return the mutable metadata mapping for each scheduled output."""

    if not isinstance(result, list):
        raise TypeError("Scheduled CLIP result must be a list.")
    outputs: list[dict[str, object]] = []
    for item in result:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise TypeError(
                "Scheduled CLIP result entries must be cond/metadata pairs."
            )
        metadata = item[1]
        if not isinstance(metadata, dict):
            raise TypeError("Scheduled CLIP result metadata must be a dictionary.")
        if any(not isinstance(key, str) for key in metadata):
            raise TypeError("Scheduled CLIP result metadata keys must be strings.")
        outputs.append(cast(dict[str, object], metadata))
    return tuple(outputs)


def _required_callable(
    value: object,
    name: str,
    *,
    value_name: str,
) -> Callable[..., object]:
    """Return one required dynamic host callable."""

    candidate = getattr(value, name, None)
    if not callable(candidate):
        raise TypeError(f"{value_name} does not expose callable {name}.")
    return cast(Callable[..., object], candidate)
