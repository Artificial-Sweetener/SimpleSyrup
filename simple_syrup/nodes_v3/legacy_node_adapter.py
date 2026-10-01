# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Translate maintained implementation contracts into native Comfy v3 schemas."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        hidden: ClassVar[Any]
        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io

_HIDDEN_INPUTS = {
    "PROMPT": "prompt",
    "DYNPROMPT": "dynprompt",
    "EXTRA_PNGINFO": "extra_pnginfo",
    "UNIQUE_ID": "unique_id",
}


class LegacyNodeV3Adapter(_ComfyNodeBase):
    """Build a v3 schema and execution bridge for a legacy implementation class."""

    LEGACY_NODE_CLASS: ClassVar[type[Any]]
    NODE_ID: ClassVar[str]
    DISPLAY_NAME: ClassVar[str]
    ENABLE_EXPAND: ClassVar[bool] = False
    WORKFLOW_INPUT_ORDER: ClassVar[tuple[str, ...] | None] = None

    @classmethod
    def define_schema(cls) -> Any:
        """Declare a v3 schema from the implementation class contract."""

        legacy = cls.LEGACY_NODE_CLASS
        return _comfy_io.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY_NAME,
            category=str(getattr(legacy, "CATEGORY", "SimpleSyrup")),
            description=str(getattr(legacy, "DESCRIPTION", "")),
            search_aliases=list(getattr(legacy, "SEARCH_ALIASES", [])),
            inputs=_v3_inputs(
                legacy.INPUT_TYPES(),
                workflow_order=cls.WORKFLOW_INPUT_ORDER,
            ),
            outputs=_v3_outputs(legacy),
            hidden=_v3_hidden_inputs(legacy.INPUT_TYPES()),
            is_input_list=bool(getattr(legacy, "INPUT_IS_LIST", False)),
            is_output_node=bool(getattr(legacy, "OUTPUT_NODE", False)),
            enable_expand=cls.ENABLE_EXPAND,
        )

    @classmethod
    def execute(cls, **kwargs: object) -> Any:
        """Run the wrapped implementation with v3-provided inputs."""

        values = dict(kwargs)
        for name, hidden_attr in _legacy_hidden_inputs(
            cls.LEGACY_NODE_CLASS.INPUT_TYPES()
        ).items():
            if name not in values:
                values[name] = getattr(cls.hidden, hidden_attr)

        function_name = str(cls.LEGACY_NODE_CLASS.FUNCTION)
        implementation = cls.LEGACY_NODE_CLASS()
        function = getattr(implementation, function_name)
        return function(**values)


def _v3_inputs(
    input_types: Mapping[str, Mapping[str, object]],
    *,
    workflow_order: tuple[str, ...] | None = None,
) -> list[Any]:
    """Return v3 inputs while preserving any explicit persisted socket order."""

    declarations: dict[str, tuple[object, bool]] = {}
    for section_name, optional in (("required", False), ("optional", True)):
        section = input_types.get(section_name, {})
        for name, declaration in section.items():
            if name in declarations:
                raise ValueError(f"legacy input {name} is declared more than once.")
            declarations[name] = (declaration, optional)
    order = tuple(declarations) if workflow_order is None else workflow_order
    if len(order) != len(set(order)) or set(order) != set(declarations):
        raise ValueError("legacy workflow input order must name every input once.")
    return [
        _v3_input(name, declarations[name][0], optional=declarations[name][1])
        for name in order
    ]


def _v3_input(name: str, declaration: object, *, optional: bool) -> Any:
    """Return one v3 input declaration from a legacy field declaration."""

    if not isinstance(declaration, tuple) or not declaration:
        raise TypeError(f"legacy input {name} declaration must be a tuple.")

    io_declaration = declaration[0]
    options = _input_options(declaration)
    tooltip = _string_option(options, "tooltip")
    advanced = _bool_option(options, "advanced")
    raw_link = _bool_option(options, "rawLink") or _bool_option(options, "raw_link")
    force_input = _bool_option(options, "forceInput") or _bool_option(
        options, "force_input"
    )

    if isinstance(io_declaration, (list, tuple)):
        return _comfy_io.Combo.Input(
            name,
            options=list(io_declaration),
            optional=optional,
            default=options.get("default"),
            control_after_generate=options.get("control_after_generate"),
            tooltip=tooltip,
            raw_link=raw_link,
            advanced=advanced,
        )

    if not isinstance(io_declaration, str):
        raise TypeError(f"legacy input {name} type must be a string or options list.")

    input_type = io_declaration
    input_class = _io_class(input_type)
    common_options = {
        "optional": optional,
        "tooltip": tooltip,
        "raw_link": raw_link,
        "advanced": advanced,
    }

    if input_type == "INT":
        return input_class.Input(
            name,
            default=options.get("default"),
            min=options.get("min"),
            max=options.get("max"),
            step=options.get("step"),
            control_after_generate=options.get("control_after_generate"),
            **common_options,
        )
    if input_type == "FLOAT":
        return input_class.Input(
            name,
            default=options.get("default"),
            min=options.get("min"),
            max=options.get("max"),
            step=options.get("step"),
            round=options.get("round"),
            **common_options,
        )
    if input_type == "STRING":
        return input_class.Input(
            name,
            default=options.get("default"),
            multiline=bool(options.get("multiline", False)),
            force_input=force_input,
            **common_options,
        )
    if input_type == "BOOLEAN":
        return input_class.Input(
            name,
            default=options.get("default"),
            label_on=options.get("label_on"),
            label_off=options.get("label_off"),
            **common_options,
        )

    return input_class.Input(name, **common_options)


def _v3_outputs(legacy: type[Any]) -> list[Any]:
    """Return v3 output declarations from legacy return metadata."""

    return_types = tuple(getattr(legacy, "RETURN_TYPES", ()))
    return_names = getattr(legacy, "RETURN_NAMES", None)
    output_tooltips = tuple(getattr(legacy, "OUTPUT_TOOLTIPS", ()))
    output_is_list = tuple(
        getattr(legacy, "OUTPUT_IS_LIST", (False,) * len(return_types))
    )
    outputs: list[Any] = []
    for index, io_type in enumerate(return_types):
        output_name = None
        if isinstance(return_names, tuple) and index < len(return_names):
            output_name = str(return_names[index])
        tooltip = None
        if index < len(output_tooltips):
            tooltip = str(output_tooltips[index])
        is_output_list = index < len(output_is_list) and bool(output_is_list[index])
        outputs.append(
            _io_class(str(io_type)).Output(
                output_name,
                tooltip=tooltip,
                is_output_list=is_output_list,
            )
        )
    return outputs


def _v3_hidden_inputs(input_types: Mapping[str, Mapping[str, object]]) -> list[Any]:
    """Return v3 hidden declarations requested by legacy hidden inputs."""

    hidden_values = set(_legacy_hidden_inputs(input_types).values())
    return [getattr(_comfy_io.Hidden, value) for value in sorted(hidden_values)]


def _legacy_hidden_inputs(
    input_types: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    """Return legacy hidden input names mapped to v3 hidden holder attributes."""

    hidden_inputs: dict[str, str] = {}
    for name, sentinel in input_types.get("hidden", {}).items():
        if isinstance(sentinel, str) and sentinel in _HIDDEN_INPUTS:
            hidden_inputs[name] = _HIDDEN_INPUTS[sentinel]
    return hidden_inputs


def _io_class(io_type: str) -> Any:
    """Return the v3 IO class for a legacy Comfy type string."""

    known_types = {
        "BOOLEAN": _comfy_io.Boolean,
        "INT": _comfy_io.Int,
        "FLOAT": _comfy_io.Float,
        "STRING": _comfy_io.String,
        "IMAGE": _comfy_io.Image,
        "MASK": _comfy_io.Mask,
        "LATENT": _comfy_io.Latent,
        "MODEL": _comfy_io.Model,
        "CLIP": _comfy_io.Clip,
        "VAE": _comfy_io.Vae,
        "CONDITIONING": _comfy_io.Conditioning,
        "SEGS": _comfy_io.SEGS,
    }
    return known_types.get(io_type, _comfy_io.Custom(io_type))


def _input_options(declaration: tuple[object, ...]) -> dict[str, object]:
    """Return an input options dictionary from a legacy declaration."""

    if len(declaration) < 2 or not isinstance(declaration[1], dict):
        return {}
    return dict(declaration[1])


def _string_option(options: Mapping[str, object], name: str) -> str | None:
    """Return a string option when present."""

    value = options.get(name)
    if isinstance(value, str):
        return value
    return None


def _bool_option(options: Mapping[str, object], name: str) -> bool | None:
    """Return a boolean option when present."""

    value = options.get(name)
    if isinstance(value, bool):
        return value
    return None
