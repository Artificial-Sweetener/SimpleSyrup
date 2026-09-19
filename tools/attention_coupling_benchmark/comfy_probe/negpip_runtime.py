# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Instrument live NegPiP attention callbacks without changing their results."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

import torch
from comfy.model_patcher import ModelPatcher

from simple_syrup.runtime.negpip.anima import (
    TRANSFORMER_MASK_KEY as ANIMA_MASK_KEY,
)
from simple_syrup.runtime.negpip.krea2 import (
    TRANSFORMER_MASK_KEY as KREA_MASK_KEY,
)

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for benchmark-only Comfy v3 nodes."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyNodeBase = _comfy_api.io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io


@dataclass
class _NegpipProbeState:
    """Accumulate live callback evidence for one managed workflow."""

    family: str
    patch_name: str
    callback_name: str
    attention_calls: int = 0
    negative_mask_calls: int = 0
    invariant_failures: list[str] = field(default_factory=list)
    input_value_shape: list[int] | None = None
    output_value_shape: list[int] | None = None
    mask_shape: list[int] | None = None
    text_length: int | None = None
    negative_token_count: int = 0
    negative_token_positions: list[int] = field(default_factory=list)
    negative_token_locations: list[list[int]] = field(default_factory=list)


_STATES: dict[str, _NegpipProbeState] = {}
_STATE_LOCK = threading.Lock()


class InstrumentNegpipModelV3(_ComfyNodeBase):
    """Wrap one installed NegPiP callback and validate live tensor semantics."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare benchmark-only MODEL instrumentation."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.InstrumentNegpipModel",
            display_name="Benchmark Instrument NegPiP Model",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Model.Input("model"),
                _comfy_io.String.Input("run_id"),
            ],
            outputs=[_comfy_io.Model.Output("model")],
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, model: object, run_id: str) -> Any:
        """Clone MODEL and replace its owned callback with an observing delegate."""

        if not isinstance(model, ModelPatcher):
            raise TypeError("NegPiP instrumentation requires a Comfy MODEL.")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("NegPiP instrumentation run ID must not be empty.")
        if model.model_options.get("ppm_negpip") is not True:
            raise ValueError("NegPiP instrumentation requires a patched MODEL.")
        cloned = model.clone()
        patches = _transformer_patches(cloned)
        patch_name, family, callback = _owned_negpip_callback(patches)
        state = _NegpipProbeState(
            family=family,
            patch_name=patch_name,
            callback_name=f"{callback.__module__}.{callback.__qualname__}",
        )
        with _STATE_LOCK:
            if run_id in _STATES:
                raise ValueError(f"NegPiP probe run is already active: {run_id!r}.")
            _STATES[run_id] = state

        def observe(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            *args: object,
            **kwargs: object,
        ) -> object:
            """Delegate one callback and record its exact family invariant."""

            result = callback(query, key, value, *args, **kwargs)
            options = _extra_options(args, kwargs)
            try:
                _observe_result(
                    state,
                    query=query,
                    key=key,
                    value=value,
                    result=result,
                    options=options,
                )
            except (TypeError, ValueError) as error:
                with _STATE_LOCK:
                    state.invariant_failures.append(str(error))
            return result

        replacement = list(patches[patch_name])
        replacement[replacement.index(callback)] = observe
        patches[patch_name] = replacement
        return _comfy_io.NodeOutput(cloned)


class ReadNegpipRuntimeV3(_ComfyNodeBase):
    """Publish and enforce completed live NegPiP callback evidence."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare a latent-synchronized evidence output."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.ReadNegpipRuntime",
            display_name="Benchmark Read NegPiP Runtime",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Latent.Input("latent"),
                _comfy_io.String.Input("run_id"),
            ],
            outputs=[
                _comfy_io.Latent.Output("latent"),
                _comfy_io.String.Output("evidence_json"),
            ],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, latent: dict[str, Any], run_id: str) -> Any:
        """Require observed negative-mask execution and return stable evidence."""

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with _STATE_LOCK:
            state = _STATES.pop(run_id, None)
        if state is None:
            raise ValueError(f"NegPiP probe run was not instrumented: {run_id!r}.")
        if state.attention_calls < 1:
            raise ValueError("NegPiP attention callback was not executed.")
        if state.negative_mask_calls < 1:
            raise ValueError("NegPiP callback never observed a negative token.")
        if state.invariant_failures:
            raise ValueError(
                "NegPiP live tensor invariants failed: "
                + "; ".join(state.invariant_failures[:3])
            )
        evidence = {
            "run_id": run_id,
            "family": state.family,
            "patch_name": state.patch_name,
            "callback_name": state.callback_name,
            "attention_calls": state.attention_calls,
            "negative_mask_calls": state.negative_mask_calls,
            "input_value_shape": state.input_value_shape,
            "output_value_shape": state.output_value_shape,
            "mask_shape": state.mask_shape,
            "text_length": state.text_length,
            "negative_token_count": state.negative_token_count,
            "negative_token_positions": state.negative_token_positions,
            "negative_token_locations": state.negative_token_locations,
            "invariant_failures": state.invariant_failures,
        }
        encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
        return _comfy_io.NodeOutput(
            latent,
            encoded,
            ui={"negpip_runtime_evidence": [evidence]},
        )


def _transformer_patches(model: ModelPatcher) -> dict[str, list[object]]:
    """Return the cloned MODEL's mutable transformer patch mapping."""

    options = model.model_options.get("transformer_options")
    if not isinstance(options, dict):
        raise TypeError("NegPiP MODEL transformer_options must be a dictionary.")
    patches = options.get("patches")
    if not isinstance(patches, dict):
        raise TypeError("NegPiP MODEL patches must be a dictionary.")
    return cast(dict[str, list[object]], patches)


def _owned_negpip_callback(
    patches: dict[str, list[object]],
) -> tuple[str, str, Any]:
    """Resolve exactly one owned family callback from a patch list."""

    identities = {
        ("src.negpip.unet_negpip", "sdxl_attn2_negpip"): (
            "attn2_patch",
            "standard",
        ),
        ("simple_syrup.runtime.negpip.standard", "standard_attn2_negpip"): (
            "attn2_patch",
            "standard",
        ),
        ("src.negpip.anima_negpip", "cosmos_attn2_negpip"): (
            "attn2_patch",
            "anima",
        ),
        ("simple_syrup.runtime.negpip.anima", "anima_attn2_negpip"): (
            "attn2_patch",
            "anima",
        ),
        ("simple_syrup.runtime.negpip.krea2", "krea2_attn1_negpip"): (
            "attn1_patch",
            "krea2",
        ),
    }
    matches: list[tuple[str, str, Any]] = []
    observed: list[str] = []
    for patch_name, callbacks in patches.items():
        if not isinstance(callbacks, list):
            raise TypeError("NegPiP transformer patches must be callback lists.")
        for callback in callbacks:
            module = getattr(callback, "__module__", None)
            qualname = getattr(callback, "__qualname__", None)
            observed.append(f"{patch_name}:{module}.{qualname}")
            resolved = None
            if isinstance(module, str) and isinstance(qualname, str):
                resolved = next(
                    (
                        value
                        for (
                            module_suffix,
                            expected_qualname,
                        ), value in identities.items()
                        if (
                            module == module_suffix
                            or module.endswith(f".{module_suffix}")
                        )
                        and qualname == expected_qualname
                    ),
                    None,
                )
            if resolved is not None:
                matches.append((*resolved, callback))
    if len(matches) != 1:
        raise ValueError(
            "NegPiP instrumentation requires exactly one owned family callback; "
            f"observed {observed!r}."
        )
    return matches[0]


def _extra_options(
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> dict[str, Any]:
    """Read Comfy's positional or keyword attention option mapping."""

    options = kwargs.get("extra_options")
    if options is None and args:
        options = args[-1]
    if not isinstance(options, dict):
        return {}
    return cast(dict[str, Any], options)


def _observe_result(
    state: _NegpipProbeState,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    result: object,
    options: dict[str, Any],
) -> None:
    """Validate one family-specific callback result against its live inputs."""

    if state.family == "standard":
        _observe_standard(state, query, key, value, result)
        return
    _observe_masked(state, query, key, value, result, options)


def _observe_standard(
    state: _NegpipProbeState,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    result: object,
) -> None:
    """Verify the standard interleaved split and count signed value pairs."""

    if not isinstance(result, tuple) or len(result) != 3:
        raise TypeError("Standard NegPiP must return a Q/K/V tuple.")
    output_query, output_key, output_value = result
    if output_query is not query:
        raise ValueError("Standard NegPiP changed attention queries.")
    if not isinstance(output_key, torch.Tensor) or not isinstance(
        output_value, torch.Tensor
    ):
        raise TypeError("Standard NegPiP must return tensor keys and values.")
    if not torch.equal(output_key, key[:, 0::2]):
        raise ValueError("Standard NegPiP did not select magnitude key positions.")
    if not torch.equal(output_value, value[:, 1::2]):
        raise ValueError("Standard NegPiP did not select signed value positions.")
    pair_delta = value[:, 0::2] - value[:, 1::2]
    signed_pairs = torch.any(pair_delta != 0, dim=-1)
    negative_locations = signed_pairs.nonzero().tolist()
    negative_positions = sorted({int(location[1]) for location in negative_locations})
    _record_observation(
        state,
        value,
        output_value,
        negative=bool(negative_locations),
        negative_positions=negative_positions,
        negative_locations=negative_locations,
    )


def _observe_masked(
    state: _NegpipProbeState,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    result: object,
    options: dict[str, Any],
) -> None:
    """Verify Anima or Krea applies a binary sign mask only to values."""

    if not isinstance(result, dict):
        raise TypeError("Masked NegPiP must return an attention tensor dictionary.")
    if result.get("q") is not query or result.get("k") is not key:
        raise ValueError("Masked NegPiP changed attention queries or keys.")
    output_value = result.get("v")
    if not isinstance(output_value, torch.Tensor):
        raise TypeError("Masked NegPiP must return tensor values.")
    mask_key = ANIMA_MASK_KEY if state.family == "anima" else KREA_MASK_KEY
    multiplier = options.get(mask_key)
    if not isinstance(multiplier, torch.Tensor):
        raise TypeError("Masked NegPiP callback did not receive its sign tensor.")
    negative_mask = multiplier[:, :, 0] < 0
    negative_locations = negative_mask.nonzero().tolist()
    negative = bool(negative_locations)
    negative_positions = sorted({int(location[1]) for location in negative_locations})
    state.mask_shape = list(multiplier.shape)
    if state.family == "anima":
        expected = value * multiplier
    else:
        image_slice = options.get("img_slice")
        if not isinstance(image_slice, (list, tuple)) or len(image_slice) != 2:
            raise ValueError("Krea NegPiP did not receive its text/image boundary.")
        text_length = image_slice[0]
        if not isinstance(text_length, int):
            raise TypeError("Krea NegPiP text boundary must be an integer.")
        state.text_length = text_length
        expected = value.clone()
        expected[:, :, :text_length] *= multiplier.to(value).unsqueeze(1)
        if not torch.equal(output_value[:, :, text_length:], value[:, :, text_length:]):
            raise ValueError("Krea NegPiP changed image or reference values.")
    if not torch.equal(output_value, expected):
        raise ValueError("Masked NegPiP values do not match the live sign tensor.")
    _record_observation(
        state,
        value,
        output_value,
        negative=negative,
        negative_positions=negative_positions,
        negative_locations=negative_locations,
    )


def _record_observation(
    state: _NegpipProbeState,
    source: torch.Tensor,
    output: torch.Tensor,
    *,
    negative: bool,
    negative_positions: list[int],
    negative_locations: list[list[int]],
) -> None:
    """Record one proven callback execution under the process-local lock."""

    with _STATE_LOCK:
        state.attention_calls += 1
        state.negative_mask_calls += int(negative)
        if len(negative_positions) > state.negative_token_count:
            state.negative_token_count = len(negative_positions)
            state.negative_token_positions = negative_positions
            state.negative_token_locations = negative_locations
        if state.input_value_shape is None:
            state.input_value_shape = list(source.shape)
            state.output_value_shape = list(output.shape)
