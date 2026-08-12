"""Verify benchmark-only LoRA execution observation nodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from tools.attention_coupling_benchmark.comfy_probe.lora_execution_probe import (
    InstrumentLoraModelV3,
    ReadLoraMetricsV3,
)


@dataclass
class _Hook:
    """Provide mutable schedule strength for transition observation."""

    hook_ref: str = "hook-ref"
    _strength_model: float = 0.75
    strength: float = 1.0

    @property
    def strength_model(self) -> float:
        """Return the effective scheduled model strength."""

        return self._strength_model * self.strength


@dataclass
class _HookGroup:
    """Expose ordered fake hooks."""

    hooks: list[_Hook]


class _FakeModel:
    """Provide the ModelPatcher surface observed by the LoRA probe."""

    def __init__(self) -> None:
        """Initialize one static patch and one registered hook."""

        self.model_options: dict[str, Any] = {}
        self.patches = {"diffusion_model.a.weight": [(0.75, object(), 1.0, None, None)]}
        self.hook = _Hook()
        self.current_hooks = _HookGroup([self.hook])
        self.hook_patches = {
            self.hook.hook_ref: {"diffusion_model.a.weight": [object()]}
        }
        self.wrapper: Any = None
        self.callbacks: dict[str, Any] = {}

    def clone(self) -> _FakeModel:
        """Return an independent patcher shell sharing the fake hook."""

        cloned = _FakeModel()
        cloned.patches = self.patches
        cloned.hook = self.hook
        cloned.current_hooks = self.current_hooks
        cloned.hook_patches = self.hook_patches
        return cloned

    def set_model_unet_function_wrapper(self, wrapper: Any) -> None:
        """Record the installed observation wrapper."""

        self.wrapper = wrapper

    def add_callback_with_key(self, call_type: str, key: str, callback: Any) -> None:
        """Record one keyed ModelPatcher callback by event type."""

        del key
        self.callbacks[call_type] = callback

    def trigger_registered_hooks(self) -> None:
        """Simulate Comfy registering hooks on the runtime clone."""

        self.callbacks["on_register_all_hook_patches"](
            self, self.current_hooks, {}, {}, self.current_hooks
        )

    def trigger_applied_hooks(self) -> None:
        """Simulate Comfy applying the current hook schedule."""

        self.callbacks["on_apply_hooks"](self, self.current_hooks)


def test_probe_records_patch_order_schedule_transitions_and_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture exact evidence while returning the delegated tensor unchanged."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    instrumented = InstrumentLoraModelV3.execute(
        _FakeModel(), "lora-run", '["adapter-a"]', True
    )
    (model,) = instrumented.result
    tensor = torch.zeros((1, 2, 2, 2), dtype=torch.float16)
    arguments = {
        "input": tensor,
        "timestep": torch.tensor([1.0]),
        "c": {},
    }
    model.trigger_registered_hooks()
    model.trigger_applied_hooks()

    def apply_model(
        input_x: torch.Tensor,
        timestep: torch.Tensor,
        **conditioning: object,
    ) -> torch.Tensor:
        """Return a recognizable denoiser output."""

        del timestep, conditioning
        return input_x + 1

    assert torch.equal(model.wrapper(apply_model, arguments), tensor + 1)
    model.hook.strength = 0.5
    model.trigger_applied_hooks()
    arguments["timestep"] = torch.tensor([0.5])
    assert torch.equal(model.wrapper(apply_model, arguments), tensor + 1)

    measured = ReadLoraMetricsV3.execute({"samples": tensor}, "lora-run")
    metrics = measured.ui["lora_benchmark_metrics"][0]

    assert metrics["model_call_count"] == 2
    assert metrics["static_patches"]["target_count"] == 1
    assert metrics["hook_patches"][0]["identity"] == "adapter-a"
    assert [
        item["effective_strengths"] for item in metrics["schedule_observations"]
    ] == [
        [0.75],
        [0.375],
    ]
    assert len(metrics["denoiser_outputs"]) == 2
    assert json.loads(measured.result[1])["run_id"] == "lora-run"


def test_probe_schemas_are_dev_only_and_identity_input_fails_closed() -> None:
    """Expose dedicated node contracts and reject duplicate identities."""

    instrument = InstrumentLoraModelV3.define_schema()
    metrics = ReadLoraMetricsV3.define_schema()
    assert instrument.node_id == "SimpleSyrupBenchmark.InstrumentLoraModel"
    assert instrument.is_dev_only is True
    assert metrics.node_id == "SimpleSyrupBenchmark.ReadLoraMetrics"
    assert metrics.is_output_node is True

    with pytest.raises(ValueError, match="unique"):
        InstrumentLoraModelV3.execute(
            _FakeModel(), "bad-run", '["adapter-a", "adapter-a"]', False
        )
