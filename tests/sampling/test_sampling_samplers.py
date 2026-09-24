# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for KSampler Extras sampler runtime helpers."""

from __future__ import annotations

import comfy.samplers
import pytest

from simple_syrup.runtime.a1111_sampling import sample_euler_ancestral_a1111
from simple_syrup.runtime.sampling_samplers import (
    available_samplers,
    resolve_sampler,
)


class FakeKSampler:
    """Capture a sampler function while exposing ComfyUI's sampler protocol."""

    def __init__(self, sampler_function: object) -> None:
        """Create a fake KSampler wrapper."""

        self.sampler_function = sampler_function

    def sample(self, *args: object, **kwargs: object) -> object:
        """Provide the sampler protocol expected by runtime code."""

        del args, kwargs
        return None


def test_available_samplers_includes_core_and_extras() -> None:
    """Sampler options combine ComfyUI core names with SimpleSyrup extras."""

    samplers = available_samplers()

    for sampler in comfy.samplers.KSampler.SAMPLERS:
        assert sampler in samplers
    assert samplers[: len(comfy.samplers.KSampler.SAMPLERS)] == tuple(
        comfy.samplers.KSampler.SAMPLERS
    )
    assert samplers[-1] == "euler_a_a1111"


def test_available_samplers_deduplicates_local_extra_when_globally_patched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local A1111 sampler is shown once if another extension patched ComfyUI."""

    monkeypatch.setattr(
        comfy.samplers.KSampler,
        "SAMPLERS",
        tuple(comfy.samplers.KSampler.SAMPLERS) + ("euler_a_a1111",),
    )

    samplers = available_samplers()

    assert samplers.count("euler_a_a1111") == 1
    assert "euler_a_a1111" in samplers


def test_available_samplers_includes_lcm() -> None:
    """LCM is exposed because it is already a core ComfyUI sampler."""

    assert "lcm" in available_samplers()


def test_resolve_sampler_returns_comfy_sampler_object() -> None:
    """Valid sampler names resolve to executable ComfyUI sampler objects."""

    sampler = resolve_sampler("lcm")

    assert callable(sampler.sample)


def test_resolve_euler_a_a1111_returns_local_sampler_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The A1111 sampler resolves locally without ComfyUI sampler_object lookup."""

    def fail_sampler_object(sampler_name: str) -> object:
        """Fail if local sampler resolution delegates to ComfyUI by name."""

        del sampler_name
        raise AssertionError("euler_a_a1111 should not call sampler_object")

    monkeypatch.setattr(comfy.samplers, "KSAMPLER", FakeKSampler)
    monkeypatch.setattr(comfy.samplers, "sampler_object", fail_sampler_object)

    sampler = resolve_sampler("euler_a_a1111")

    assert callable(sampler.sample)
    assert isinstance(sampler, FakeKSampler)
    assert sampler.sampler_function is sample_euler_ancestral_a1111


def test_resolve_sampler_rejects_unknown_sampler() -> None:
    """Unsupported sampler names fail before sampling begins."""

    with pytest.raises(ValueError, match="Unsupported sampler 'not-real'"):
        resolve_sampler("not-real")
