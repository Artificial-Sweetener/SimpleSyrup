# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove capture preparation precedes native validation for every queue caller."""

from __future__ import annotations

import asyncio
import inspect
import sys
from copy import deepcopy
from types import ModuleType
from typing import Any

import pytest

from simple_syrup.runtime.noise_reuse_validation import (
    NativePromptValidator,
    prepare_noise_reuse_prompt,
    register_noise_reuse_validation,
)


class ValidationModule(ModuleType):
    """Describe the isolated host module's replaceable validation function."""

    validate_prompt: NativePromptValidator


async def validate(
    execution: ValidationModule,
    prompt_id: str,
    prompt: dict[str, Any],
    partial_targets: list[str] | None,
) -> tuple[bool, object, object, object]:
    """Await the external validator whether it returns a coroutine or awaitable."""
    return await execution.validate_prompt(prompt_id, prompt, partial_targets)


def capture_graph() -> dict[str, Any]:
    """Describe direct source lineage and a live receiving capture dependency."""
    return {
        "source": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["loader", 0],
                "latent_image": ["empty", 0],
                "sampler_name": "euler_ancestral",
            },
        },
        "reuse": {
            "class_type": "SimpleSyrup.NoiseReuseOptions",
            "inputs": {"enabled": True},
        },
        "receive": {
            "class_type": "SimpleSyrup.KSampler",
            "inputs": {
                "model": ["loader", 0],
                "latent_image": ["source", 0],
                "options": ["reuse", 0],
                "steps": 12,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 0.15,
            },
        },
    }


@pytest.fixture
def validation_host(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ValidationModule, list[tuple[str, dict[str, Any], list[str] | None]]]:
    """Isolate only Comfy's external validation and registration boundaries."""
    calls: list[tuple[str, dict[str, Any], list[str] | None]] = []

    async def validate_prompt(
        prompt_id: str,
        prompt: dict[str, Any],
        partial_execution_list: list[str] | None,
    ) -> tuple[bool, object, object, object]:
        """Record the graph exactly as the native validator receives it."""
        calls.append((prompt_id, deepcopy(prompt), partial_execution_list))
        return True, None, ["receive"], {}

    execution = ValidationModule("execution")
    execution.validate_prompt = validate_prompt
    nodes = ModuleType("nodes")
    nodes.NODE_CLASS_MAPPINGS = {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "execution", execution)
    monkeypatch.setitem(sys.modules, "nodes", nodes)
    return execution, calls


@pytest.mark.parametrize("partial_targets", [None, ["receive"]])
def test_direct_validation_prepares_capture_without_http_handler(
    validation_host: tuple[
        ValidationModule, list[tuple[str, dict[str, Any], list[str] | None]]
    ],
    partial_targets: list[str] | None,
) -> None:
    """Cover both full Cube queueing and Comfy partial-output validation."""
    execution, calls = validation_host
    register_noise_reuse_validation()
    prompt = capture_graph()
    result = asyncio.run(validate(execution, "run", prompt, partial_targets))
    assert result == (True, None, ["receive"], {})
    assert len(calls) == 1
    prompt_id, validated, targets = calls[0]
    assert prompt_id == "run" and targets is partial_targets
    assert validated == prompt
    assert validated["source"]["inputs"]["model"] == ["source.trajectory_capture", 0]
    assert validated["receive"]["inputs"]["latent_image"] == [
        "source.sampling_provenance",
        0,
    ]
    assert validated["receive.noise_capture_request"]["inputs"]["denoise"] == 0.15


def test_registration_and_history_requeue_do_not_duplicate_capture(
    validation_host: tuple[
        ValidationModule, list[tuple[str, dict[str, Any], list[str] | None]]
    ],
) -> None:
    """Preserve graph identity across registration, pre-preparation and history."""
    execution, calls = validation_host
    register_noise_reuse_validation()
    validator = execution.validate_prompt
    register_noise_reuse_validation()
    assert execution.validate_prompt is validator
    prompt = capture_graph()
    prepare_noise_reuse_prompt(prompt)
    prepared = deepcopy(prompt)
    for prompt_id in ("first", "requeue"):
        asyncio.run(validate(execution, prompt_id, prompt, None))
    assert prompt == prepared and len(calls) == 2
    assert all(validated == prepared for _, validated, _ in calls)


def test_registration_preserves_async_signature_and_wrapped_validator(
    validation_host: tuple[
        ValidationModule, list[tuple[str, dict[str, Any], list[str] | None]]
    ],
) -> None:
    """Keep native function inspection and extension chaining compatible."""
    execution, _ = validation_host
    native = execution.validate_prompt
    register_noise_reuse_validation()
    wrapped = execution.validate_prompt
    assert inspect.iscoroutinefunction(wrapped)
    assert inspect.signature(wrapped) == inspect.signature(native)
    assert inspect.unwrap(wrapped) is native


def test_changed_receiving_controls_remain_source_cache_dependencies(
    validation_host: tuple[
        ValidationModule, list[tuple[str, dict[str, Any], list[str] | None]]
    ],
) -> None:
    """Keep fresh submissions sensitive to downstream denoise and live steps."""
    execution, calls = validation_host
    register_noise_reuse_validation()
    for denoise in (0.15, 0.2):
        prompt = capture_graph()
        prompt["receive"]["inputs"].update(denoise=denoise, steps=["live_steps", 0])
        asyncio.run(validate(execution, "run", prompt, None))
    requests = [
        graph["receive.noise_capture_request"]["inputs"] for _, graph, _ in calls
    ]
    assert [request["denoise"] for request in requests] == [0.15, 0.2]
    assert all(request["steps"] == ["live_steps", 0] for request in requests)


def test_unrelated_graph_and_validation_rejection_are_preserved(
    validation_host: tuple[
        ValidationModule, list[tuple[str, dict[str, Any], list[str] | None]]
    ],
) -> None:
    """Leave ordinary workflows and native validation error payloads intact."""
    execution, _ = validation_host
    prompt = {"output": {"class_type": "SaveImage", "inputs": {}}}
    before = deepcopy(prompt)
    rejection: tuple[bool, object, object, object] = (
        False,
        {"type": "native_error"},
        [],
        {"output": {"errors": []}},
    )

    async def reject(
        prompt_id: str, prompt: dict[str, Any], partial_execution_list: list[str] | None
    ) -> tuple[bool, object, object, object]:
        """Return the host's validation rejection without extension translation."""
        return rejection

    execution.validate_prompt = reject
    register_noise_reuse_validation()
    assert asyncio.run(validate(execution, "run", prompt, None)) is rejection
    assert prompt == before


def test_native_validation_exception_propagates(
    validation_host: tuple[
        ValidationModule, list[tuple[str, dict[str, Any], list[str] | None]]
    ],
) -> None:
    """Keep unexpected host failures visible to the queue that called it."""
    execution, _ = validation_host

    async def fail(
        prompt_id: str, prompt: dict[str, Any], partial_execution_list: list[str] | None
    ) -> tuple[bool, object, object, object]:
        """Represent a failed external native validator."""
        raise RuntimeError("native validation failed")

    execution.validate_prompt = fail
    register_noise_reuse_validation()
    with pytest.raises(RuntimeError, match="native validation failed"):
        asyncio.run(validate(execution, "run", capture_graph(), None))


def test_preparation_failure_stops_before_validation(
    validation_host: tuple[
        ValidationModule, list[tuple[str, dict[str, Any], list[str] | None]]
    ],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reject an unexpected planner failure rather than silently skipping capture."""
    execution, calls = validation_host

    def fail(prompt: dict[str, Any]) -> None:
        """Represent failure at the graph preparation boundary."""
        raise RuntimeError("capture planning failed")

    monkeypatch.setattr(
        "simple_syrup.runtime.noise_reuse_validation.prepare_noise_reuse_prompt", fail
    )
    register_noise_reuse_validation()
    with pytest.raises(RuntimeError, match="capture planning failed"):
        asyncio.run(validate(execution, "run", capture_graph(), None))
    assert calls == []
    assert "capture planning failed" in caplog.text


def test_registration_without_comfy_runtime_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow tooling imports when Comfy's execution module is unavailable."""

    def unavailable(name: str) -> ModuleType:
        """Represent an absent external host installation."""
        raise ModuleNotFoundError("No module named execution", name="execution")

    monkeypatch.setattr(
        "simple_syrup.runtime.noise_reuse_validation.import_module", unavailable
    )
    register_noise_reuse_validation()
