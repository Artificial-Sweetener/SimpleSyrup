# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node for loading an ordered list of authored images."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..services.load_image_list_service import LoadImageListService

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_api: Any = None if TYPE_CHECKING else import_module("comfy_api.latest")
_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io


class LoadImageListV3(_ComfyNodeBase):
    """Load ordered files as a native Comfy IMAGE execution list."""

    service_class: ClassVar[type[LoadImageListService]] = LoadImageListService

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the persisted multi-upload and IMAGE list output."""

        choices = list(cls.service_class().available_files())
        return _comfy_io.Schema(
            node_id="SimpleSyrup.LoadImageList",
            display_name="Load Image List",
            category="SimpleSyrup/Loaders",
            description=(
                "Loads an ordered image list without resizing or batching its items."
            ),
            search_aliases=["load images", "image list", "reference images"],
            inputs=[
                _comfy_io.MultiCombo.Input(
                    "image",
                    options=choices,
                    default=[],
                    placeholder="Select one or more images",
                    chip=True,
                    tooltip=(
                        "Select images in order; each file becomes one independent "
                        "IMAGE list item."
                    ),
                    extra_dict={
                        "image_upload": True,
                        "image_folder": "input",
                        "allow_batch": True,
                    },
                ),
            ],
            outputs=[
                _comfy_io.Image.Output(
                    "images",
                    tooltip=(
                        "Ordered IMAGE list with one independently sized item per file."
                    ),
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def execute(cls, image: str | list[str]) -> Any:
        """Return the selected images as a native Comfy execution list."""

        return _comfy_io.NodeOutput(cls.service_class().load(image))

    @classmethod
    def validate_inputs(cls, image: str | list[str]) -> bool | str:
        """Validate widget values before image loading."""

        try:
            cls.service_class().validate(image)
        except (TypeError, ValueError) as error:
            return str(error)
        return True

    @classmethod
    def fingerprint_inputs(cls, image: str | list[str]) -> str:
        """Fingerprint ordered filenames and file contents."""

        return cls.service_class().fingerprint(image)
