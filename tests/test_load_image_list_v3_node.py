# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Load Image List Comfy v3 node."""

from __future__ import annotations

from typing import ClassVar

import torch

from simple_syrup.nodes_v3.load_image_list import LoadImageListV3


class FakeService:
    """Provide deterministic image-list behavior at the node boundary."""

    loaded: ClassVar[list[str] | None] = None

    def available_files(self) -> tuple[str, ...]:
        """Return node combo choices."""

        return ("b.png", "a.png")

    def validate(self, files: str | list[str]) -> None:
        """Accept deterministic widget values."""

        del files

    def load(self, files: str | list[str]) -> list[torch.Tensor]:
        """Record order and return independently sized singleton images."""

        ordered = [files] if isinstance(files, str) else files
        type(self).loaded = ordered
        return [torch.zeros((1, 2, 3, 3)), torch.ones((1, 4, 2, 3))]

    def fingerprint(self, files: str | list[str]) -> str:
        """Return an order-visible node fingerprint."""

        ordered = [files] if isinstance(files, str) else files
        return "|".join(ordered)


def test_schema_exposes_native_ordered_image_upload_and_list_output() -> None:
    """The widget persists filenames while the IMAGE socket is a Comfy list."""

    original = LoadImageListV3.service_class
    LoadImageListV3.service_class = FakeService  # type: ignore[assignment]
    try:
        schema = LoadImageListV3.define_schema()
    finally:
        LoadImageListV3.service_class = original

    inputs = {value.id: value for value in schema.inputs}
    image = inputs["image"].as_dict()
    assert schema.node_id == "SimpleSyrup.LoadImageList"
    assert schema.display_name == "Load Image List"
    assert schema.has_intermediate_output is False
    assert image["image_upload"] is True
    assert image["allow_batch"] is True
    assert image["image_folder"] == "input"
    assert image["multiselect"] is True
    assert image["default"] == []
    assert schema.outputs[0].io_type == "IMAGE"
    assert schema.outputs[0].is_output_list is True


def test_execute_returns_exact_ordered_python_list_without_batching() -> None:
    """Comfy receives one singleton tensor for every ordered image position."""

    original = LoadImageListV3.service_class
    LoadImageListV3.service_class = FakeService  # type: ignore[assignment]
    try:
        output = LoadImageListV3.execute(["b.png", "a.png"])
        fingerprint = LoadImageListV3.fingerprint_inputs(["b.png", "a.png"])
    finally:
        LoadImageListV3.service_class = original

    assert FakeService.loaded == ["b.png", "a.png"]
    assert output.result is not None
    images = output.result[0]
    assert isinstance(images, list)
    assert [tuple(image.shape) for image in images] == [(1, 2, 3, 3), (1, 4, 2, 3)]
    assert fingerprint == "b.png|a.png"


def test_validation_surfaces_actionable_service_failures() -> None:
    """Comfy validation receives the service's workflow-facing reason."""

    class RejectingService(FakeService):
        """Reject an empty image list."""

        def validate(self, files: str | list[str]) -> None:
            """Raise one deterministic validation failure."""

            del files
            raise ValueError("Load Image List requires at least one image file.")

    original = LoadImageListV3.service_class
    LoadImageListV3.service_class = RejectingService  # type: ignore[assignment]
    try:
        result: bool | str = LoadImageListV3.validate_inputs([])
    finally:
        LoadImageListV3.service_class = original

    assert result == "Load Image List requires at least one image file."
