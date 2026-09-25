# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify universal CFG and positive-only ComfyUI guider selection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from comfy import samplers as comfy_samplers

from simple_syrup.runtime import guided_sampling


class _RecordingComfySample:
    """Record the unchanged CFG sampling boundary."""

    def __init__(self, result: torch.Tensor) -> None:
        """Retain the recognizable CFG result."""

        self.result = result
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def sample_custom(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Record and return the configured CFG result."""

        self.calls.append((args, kwargs))
        return self.result


class _RecordingGuider:
    """Represent ComfyUI's core guider used by BasicGuider."""

    instances: ClassVar[list[_RecordingGuider]] = []

    def __init__(self, model: object) -> None:
        """Record the model and Comfy's positive-only guider defaults."""

        self.model = model
        self.cfg = 1.0
        self.conds: dict[str, object] | None = None
        self.sample_call: tuple[tuple[Any, ...], dict[str, Any]] | None = None
        type(self).instances.append(self)

    def inner_set_conds(self, conds: dict[str, object]) -> None:
        """Record the exact conditioning branches registered for sampling."""

        self.conds = conds

    def sample(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Return a distinctive tensor after recording guider execution."""

        self.sample_call = (args, kwargs)
        return torch.ones((1, 4, 2, 2), dtype=torch.float64)


class _InstalledComfyModel:
    """Provide the model-patcher surface used while Comfy registers conditions."""

    def __init__(self) -> None:
        """Create static model options for the installed core guider."""

        self.model_options: dict[str, object] = {}

    def is_dynamic(self) -> bool:
        """Keep condition registration on this test model patcher."""

        return False


def _arguments(negative: object | None) -> dict[str, Any]:
    """Build one recognizable shared sampling request."""

    return {
        "model": object(),
        "noise": torch.zeros((1, 4, 2, 2)),
        "cfg": 7.5,
        "sampler": object(),
        "sigmas": torch.tensor([1.0, 0.0]),
        "positive": [[torch.ones((1, 1, 1)), {}]],
        "negative": negative,
        "latent_image": torch.zeros((1, 4, 2, 2)),
        "noise_mask": torch.ones((1, 2, 2)),
        "callback": object(),
        "disable_pbar": True,
        "seed": 42,
    }


def test_connected_negative_preserves_sample_custom_cfg_call() -> None:
    """Keep the established CFG path byte-for-byte at its Comfy boundary."""

    expected = torch.full((1, 4, 2, 2), 3.0)
    comfy_sample = _RecordingComfySample(expected)
    negative = [[torch.zeros((1, 1, 1)), {}]]
    arguments = _arguments(negative)

    result = guided_sampling.sample_with_optional_negative(
        comfy_sample=comfy_sample,
        **arguments,
    )

    assert result is expected
    assert len(comfy_sample.calls) == 1
    positional, keywords = comfy_sample.calls[0]
    assert positional == (
        arguments["model"],
        arguments["noise"],
        arguments["cfg"],
        arguments["sampler"],
        arguments["sigmas"],
        arguments["positive"],
        negative,
        arguments["latent_image"],
    )
    assert keywords == {
        "noise_mask": arguments["noise_mask"],
        "callback": arguments["callback"],
        "disable_pbar": True,
        "seed": 42,
    }


def test_disconnected_negative_uses_comfy_positive_only_guider(
    monkeypatch: Any,
) -> None:
    """Register only positive conditioning through Comfy's BasicGuider behavior."""

    comfy_sample = _RecordingComfySample(torch.empty(0))
    arguments = _arguments(None)
    _RecordingGuider.instances = []

    def fake_import(name: str) -> object:
        """Provide only the two Comfy modules used by positive-only sampling."""

        if name == "comfy.samplers":
            return SimpleNamespace(CFGGuider=_RecordingGuider)
        if name == "comfy.model_management":
            return SimpleNamespace(
                intermediate_device=lambda: torch.device("cpu"),
                intermediate_dtype=lambda: torch.float32,
            )
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(guided_sampling, "import_module", fake_import)

    result = guided_sampling.sample_with_optional_negative(
        comfy_sample=comfy_sample,
        **arguments,
    )

    assert not comfy_sample.calls
    assert result.dtype is torch.float32
    assert len(_RecordingGuider.instances) == 1
    guider = _RecordingGuider.instances[0]
    assert guider.model is arguments["model"]
    assert guider.cfg == 1.0
    assert guider.conds == {"positive": arguments["positive"]}
    assert guider.sample_call == (
        (
            arguments["noise"],
            arguments["latent_image"],
            arguments["sampler"],
            arguments["sigmas"],
        ),
        {
            "denoise_mask": arguments["noise_mask"],
            "callback": arguments["callback"],
            "disable_pbar": True,
            "seed": 42,
        },
    )


def test_installed_comfy_guider_registers_only_positive_conditioning(
    monkeypatch: Any,
) -> None:
    """Exercise the production adapter against ComfyUI's installed core guider."""

    captured: dict[str, object] = {}

    def fake_sample(
        guider: Any,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Capture Comfy's registered conditions before GPU sampling begins."""

        del args, kwargs
        captured["condition_names"] = tuple(guider.original_conds)
        captured["cfg"] = guider.cfg
        return torch.ones((1, 4, 2, 2))

    monkeypatch.setattr(comfy_samplers.CFGGuider, "sample", fake_sample)
    arguments = _arguments(None)
    arguments["model"] = _InstalledComfyModel()

    result = guided_sampling.sample_with_optional_negative(
        comfy_sample=_RecordingComfySample(torch.empty(0)),
        **arguments,
    )

    assert result.shape == (1, 4, 2, 2)
    assert captured == {"condition_names": ("positive",), "cfg": 1.0}
