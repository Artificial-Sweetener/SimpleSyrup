# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Integration test for the authored-mask regional prompt workflow."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.nodes.encode_prompt_batch import EncodePromptBatch
from simple_syrup.nodes_v3 import load_mask_batch as load_node_module
from simple_syrup.nodes_v3.ksampler_prompt_by_region import (
    KSamplerPromptByRegionV3,
)
from simple_syrup.nodes_v3.load_mask_batch import LoadMaskBatchV3


class WorkflowEncoder:
    """Encode prompt chunks into standard recognizable conditioning."""

    def encode_batch(
        self,
        clip: Any,
        chunks: tuple[str, ...],
    ) -> ConditioningBatch:
        """Return ordered standard conditioning entries."""

        del clip
        return ConditioningBatch(tuple([[chunk, {}]] for chunk in chunks))


class WorkflowMaskLoader:
    """Return three authored masks in selected-file order."""

    def load(self, files: list[str], channel: str) -> torch.Tensor:
        """Return values that expose positional ordering."""

        assert files == ["left.png", "right.png", "extra.png"]
        assert channel == "red"
        return torch.stack(
            [
                torch.zeros((4, 4)),
                torch.ones((4, 4)),
                torch.full((4, 4), 0.5),
            ]
        )

    def validate(self, files: str | list[str], channel: str) -> None:
        """Accept integration-test widget values."""

        del files, channel

    def fingerprint(self, files: list[str], channel: str) -> str:
        """Provide the loader protocol for the node class."""

        return f"{channel}:{files!r}"

    def available_files(self) -> tuple[str, ...]:
        """Provide the schema protocol for the node class."""

        return ()


class WorkflowSampler:
    """Capture assembled conditioning at the final sampling boundary."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def sample(self, **kwargs: Any) -> dict[str, Any]:
        """Return the input latent after recording the complete request."""

        type(self).calls.append(kwargs)
        return cast(dict[str, Any], kwargs["latent_image"])


def test_sep_prompts_and_disk_masks_flow_into_regional_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public nodes compose with global-first and missing-region behavior."""

    monkeypatch.setattr(EncodePromptBatch, "encoder_class", WorkflowEncoder)
    monkeypatch.setattr(LoadMaskBatchV3, "service_class", WorkflowMaskLoader)
    monkeypatch.setattr(
        load_node_module._comfy_ui, "PreviewMask", lambda mask, cls: None
    )
    monkeypatch.setattr(
        KSamplerPromptByRegionV3,
        "sampling_service_class",
        WorkflowSampler,
    )
    WorkflowSampler.calls = []

    positive, negative = EncodePromptBatch().encode(
        clip=object(),
        positive_prompt="global [SEP] left [SEP] right",
        negative_prompt="global negative [SEP] left negative",
        separator="[SEP]",
    )
    mask_output = LoadMaskBatchV3.execute(
        image=["left.png", "right.png", "extra.png"],
        channel="red",
    )
    assert mask_output.result is not None
    masks = mask_output.result[0]
    latent = {"samples": torch.zeros((2, 4, 1, 1))}

    (output,) = KSamplerPromptByRegionV3.execute(
        model=object(),
        seed=1,
        steps=2,
        cfg=3.0,
        sampler_name="euler",
        scheduler="normal",
        positive=positive,
        negative=negative,
        region_masks=masks,
        regional_prompt_weight=0.8,
        region_mask_feather=0,
        latent_image=latent,
        denoise=1.0,
    )

    assert output is latent
    call = WorkflowSampler.calls[0]
    assembled_positive = call["positive"]
    assembled_negative = call["negative"]
    assert [item[0] for item in assembled_positive] == ["global", "left", "right"]
    assert [item[0] for item in assembled_negative] == [
        "global negative",
        "left negative",
    ]
    assert torch.equal(assembled_positive[1][1]["mask"], masks[0:1])
    assert torch.equal(assembled_positive[2][1]["mask"], masks[1:2])
    assert assembled_positive[1][1]["mask_strength"] == 0.8
    assert assembled_positive[2][1]["mask_strength"] == 0.8
    assert assembled_negative[1][1]["mask_strength"] == 0.8
    assert call["latent_image"]["samples"].shape[0] == 2
