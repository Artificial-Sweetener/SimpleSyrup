# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Load Mask Batch Comfy v3 node."""

from __future__ import annotations

from typing import ClassVar

import torch

from simple_syrup.nodes_v3.load_mask_batch import LoadMaskBatchV3


class FakeService:
    """Provide deterministic loader behavior at the node boundary."""

    loaded: ClassVar[tuple[list[str], str] | None] = None

    def available_files(self) -> tuple[str, ...]:
        """Return node combo choices."""

        return ("b.png", "a.png")

    def validate(self, files: str | list[str], channel: str) -> None:
        """Accept deterministic widget values."""

        del files, channel

    def load(self, files: list[str], channel: str) -> torch.Tensor:
        """Record input order and return a two-mask batch."""

        type(self).loaded = (files, channel)
        return torch.stack([torch.zeros((2, 2)), torch.ones((2, 2))])

    def fingerprint(self, files: list[str], channel: str) -> str:
        """Return an order-visible node fingerprint."""

        return f"{channel}:{'|'.join(files)}"


def test_schema_exposes_native_ordered_mask_multiselect() -> None:
    """The node uses one native multi-value file list and disk uploader."""

    original = LoadMaskBatchV3.service_class
    LoadMaskBatchV3.service_class = FakeService  # type: ignore[assignment]
    try:
        schema = LoadMaskBatchV3.define_schema()
    finally:
        LoadMaskBatchV3.service_class = original

    inputs = {value.id: value for value in schema.inputs}
    image = inputs["image"].as_dict()
    assert schema.node_id == "SimpleSyrup.LoadMaskBatch"
    assert schema.display_name == "Load Mask Batch"
    assert schema.has_intermediate_output is False
    assert image["image_upload"] is True
    assert image["allow_batch"] is True
    assert image["image_folder"] == "input"
    assert image["multiselect"] is True
    assert image["multi_select"] == {
        "placeholder": "Select one or more masks",
        "chip": True,
    }
    assert image["default"] == []
    assert image["options"] == ["b.png", "a.png"]
    assert [output.io_type for output in schema.outputs] == ["MASK"]


def test_execute_preserves_order_and_returns_one_mask_batch() -> None:
    """Execution returns one MASK batch without adding a duplicate UI gallery."""

    original = LoadMaskBatchV3.service_class
    LoadMaskBatchV3.service_class = FakeService  # type: ignore[assignment]
    try:
        output = LoadMaskBatchV3.execute(["b.png", "a.png"], "green")
        fingerprint = LoadMaskBatchV3.fingerprint_inputs(["b.png", "a.png"], "green")
    finally:
        LoadMaskBatchV3.service_class = original

    assert FakeService.loaded == (["b.png", "a.png"], "green")
    assert output.result is not None
    mask_batch = output.result[0]
    assert mask_batch.shape == (2, 2, 2)
    assert output.ui is None
    assert fingerprint == "green:b.png|a.png"


def test_validate_inputs_accepts_native_scalar_or_batch_values() -> None:
    """The regular native combo may serialize one path or an ordered path list."""

    original = LoadMaskBatchV3.service_class
    LoadMaskBatchV3.service_class = FakeService  # type: ignore[assignment]
    try:
        assert LoadMaskBatchV3.validate_inputs("a.png", "alpha") is True
        assert LoadMaskBatchV3.validate_inputs(["b.png", "a.png"], "red") is True
    finally:
        LoadMaskBatchV3.service_class = original
