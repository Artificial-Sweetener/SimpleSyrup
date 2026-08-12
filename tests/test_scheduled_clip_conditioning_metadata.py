"""Prove metadata preservation around installed scheduled CLIP encoding."""

from __future__ import annotations

from typing import Protocol

import pytest
import torch

from simple_syrup.runtime.scheduled_clip_conditioning_metadata import (
    ScheduledClipConditioningMetadataAdapter,
)


class _Encoder:
    """Return one configured encoder result per scheduled invocation."""

    def __init__(self, outputs: list[tuple[object, object, dict[str, object]]]) -> None:
        """Retain ordered installed-encoder outputs."""

        self.outputs = outputs
        self.calls = 0

    def encode_token_weights(
        self,
        tokens: object,
    ) -> tuple[object, object, dict[str, object]]:
        """Return the next exact output without interpreting tokens."""

        output = self.outputs[self.calls]
        self.calls += 1
        return output


class _TokenEncoder(Protocol):
    """Describe the fake encoder surface consumed by the fake CLIP."""

    def encode_token_weights(
        self,
        tokens: object,
    ) -> tuple[object, object, object]:
        """Return one installed-style encoder value."""


class _Clip:
    """Mimic the installed scheduled path that drops encoder extras."""

    def __init__(self, model: _TokenEncoder, schedule_count: int) -> None:
        """Retain a shared encoder and output count."""

        self.cond_stage_model = model
        self.schedule_count = schedule_count
        self.marker = object()
        self.clone_calls: list[bool] = []

    def clone(self, disable_dynamic: bool = False) -> _Clip:
        """Return a host clone sharing the installed text encoder."""

        self.clone_calls.append(disable_dynamic)
        clone = _Clip(self.cond_stage_model, self.schedule_count)
        clone.clone_calls = self.clone_calls
        return clone

    def encode_from_tokens_scheduled(
        self,
        tokens: object,
        *,
        unprojected: bool,
        add_dict: dict[str, object],
        show_pbar: bool,
    ) -> list[list[object]]:
        """Drop the third encoder return like the installed Comfy path."""

        outputs: list[list[object]] = []
        for _ in range(self.schedule_count):
            encoded = self.cond_stage_model.encode_token_weights(tokens)
            metadata = {"pooled_output": encoded[1], **add_dict}
            outputs.append([encoded[0], metadata])
        return outputs


def test_adapter_preserves_each_encoder_metadata_value_by_identity() -> None:
    """Align multiple scheduled outputs with their exact third-return values."""

    first_ids = torch.tensor([1, 2])
    second_ids = torch.tensor([3, 4])
    model = _Encoder(
        [
            ("cond-1", "pooled-1", {"t5xxl_ids": first_ids}),
            ("cond-2", "pooled-2", {"t5xxl_ids": second_ids}),
        ]
    )
    adapter = ScheduledClipConditioningMetadataAdapter(_Clip(model, 2))

    result = adapter.encode_from_tokens_scheduled("tokens")

    assert isinstance(result, list)
    assert result[0][1]["t5xxl_ids"] is first_ids
    assert result[1][1]["t5xxl_ids"] is second_ids
    assert "encode_token_weights" not in vars(model)


def test_adapter_preserves_existing_host_metadata_and_delegates_attributes() -> None:
    """Let future host-preserved keys win while exposing the native CLIP surface."""

    model = _Encoder([("cond", "pooled", {"attention_mask": "captured"})])
    clip = _Clip(model, 1)
    adapter = ScheduledClipConditioningMetadataAdapter(clip)

    result = adapter.encode_from_tokens_scheduled(
        "tokens",
        add_dict={"attention_mask": "host"},
    )

    assert result == [["cond", {"pooled_output": "pooled", "attention_mask": "host"}]]
    assert adapter.marker is clip.marker


def test_adapter_wraps_native_clones_and_retains_shared_encoder() -> None:
    """Keep preservation active when Prompt Control clones the prepared CLIP."""

    model = _Encoder([("cond", "pooled", {})])
    clip = _Clip(model, 1)
    adapter = ScheduledClipConditioningMetadataAdapter(clip)

    clone = adapter.clone(disable_dynamic=True)

    assert isinstance(clone, ScheduledClipConditioningMetadataAdapter)
    assert clone.cond_stage_model is model
    assert clip.clone_calls == [True]


def test_adapter_restores_encoder_after_delegated_failure() -> None:
    """Never leave capture behavior installed when the host encode raises."""

    class _FailingClip(_Clip):
        """Raise after one captured encoder invocation."""

        def encode_from_tokens_scheduled(
            self,
            tokens: object,
            *,
            unprojected: bool,
            add_dict: dict[str, object],
            show_pbar: bool,
        ) -> list[list[object]]:
            """Invoke the encoder once and expose the delegated failure."""

            self.cond_stage_model.encode_token_weights("tokens")
            raise RuntimeError("delegated failure")

    model = _Encoder([("cond", "pooled", {})])
    adapter = ScheduledClipConditioningMetadataAdapter(_FailingClip(model, 1))

    with pytest.raises(RuntimeError, match="delegated failure"):
        adapter.encode_from_tokens_scheduled("tokens")
    assert "encode_token_weights" not in vars(model)


def test_adapter_rejects_cardinality_and_malformed_host_surfaces() -> None:
    """Fail explicitly when captures cannot align with scheduled results."""

    class _NoCaptureClip(_Clip):
        """Return one result without invoking the encoder."""

        def encode_from_tokens_scheduled(
            self,
            tokens: object,
            *,
            unprojected: bool,
            add_dict: dict[str, object],
            show_pbar: bool,
        ) -> list[list[object]]:
            """Return one unpaired scheduled result."""

            return [["cond", {}]]

    model = _Encoder([("cond", "pooled", {})])
    adapter = ScheduledClipConditioningMetadataAdapter(_NoCaptureClip(model, 1))
    with pytest.raises(ValueError, match="cardinality mismatch"):
        adapter.encode_from_tokens_scheduled("tokens")

    missing_model = ScheduledClipConditioningMetadataAdapter(object())
    with pytest.raises(TypeError, match="cond_stage_model"):
        missing_model.encode_from_tokens_scheduled("tokens")


def test_adapter_rejects_malformed_encoder_extras_after_restoration() -> None:
    """Reject a non-mapping third return without retaining the capture callable."""

    class _MalformedEncoder:
        """Return an invalid third value from the installed encoder surface."""

        def encode_token_weights(self, tokens: object) -> tuple[object, object, object]:
            """Return one malformed encoder result."""

            return "cond", "pooled", object()

    model = _MalformedEncoder()
    adapter = ScheduledClipConditioningMetadataAdapter(_Clip(model, 1))
    with pytest.raises(TypeError, match="extra metadata"):
        adapter.encode_from_tokens_scheduled("tokens")
    assert "encode_token_weights" not in vars(model)
