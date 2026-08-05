# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node for loading one or many authored masks with native widgets."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..runtime.mask_file_loader import MASK_CHANNELS
from ..services.load_mask_batch_service import LoadMaskBatchService

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_api: Any = None if TYPE_CHECKING else import_module("comfy_api.latest")
_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io


class LoadMaskBatchV3(_ComfyNodeBase):
    """Load an ordered set of authored files as one Comfy MASK batch."""

    service_class: ClassVar[type[LoadMaskBatchService]] = LoadMaskBatchService

    @classmethod
    def define_schema(cls) -> Any:
        """Declare a native image-upload combo that accepts one or many files."""

        choices = list(cls.service_class().available_files())
        return _comfy_io.Schema(
            node_id="SimpleSyrup.LoadMaskBatch",
            display_name="Load Mask Batch",
            category="SimpleSyrup/Loaders",
            description=(
                "Loads one or many authored mask files in selection order as a "
                "single mask batch."
            ),
            search_aliases=["load masks", "mask batch", "regional masks"],
            inputs=[
                _comfy_io.MultiCombo.Input(
                    "image",
                    options=choices,
                    default=[],
                    placeholder="Select one or more masks",
                    chip=True,
                    tooltip=(
                        "Select one or more ordered mask files; each selected "
                        "file becomes one regional mask."
                    ),
                    extra_dict={
                        "image_upload": True,
                        "image_folder": "input",
                        "allow_batch": True,
                    },
                ),
                _comfy_io.Combo.Input(
                    "channel",
                    options=list(MASK_CHANNELS),
                    default="alpha",
                    tooltip=(
                        "Image channel read from every file. Missing alpha "
                        "produces zero coverage at the source image dimensions."
                    ),
                ),
            ],
            outputs=[
                _comfy_io.Mask.Output(
                    "mask",
                    tooltip="Ordered BHW mask batch containing one mask per file.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image: str | list[str], channel: str) -> Any:
        """Load the selected files as one same-sized mask batch."""

        mask_batch = cls.service_class().load(image, channel)
        return _comfy_io.NodeOutput(mask_batch)

    @classmethod
    def validate_inputs(cls, image: str | list[str], channel: str) -> bool | str:
        """Validate native widget values before execution or filesystem reads."""

        try:
            cls.service_class().validate(image, channel)
        except (TypeError, ValueError) as error:
            return str(error)
        return True

    @classmethod
    def fingerprint_inputs(cls, image: str | list[str], channel: str) -> str:
        """Fingerprint the ordered selected files and shared channel."""

        return cls.service_class().fingerprint(image, channel)
