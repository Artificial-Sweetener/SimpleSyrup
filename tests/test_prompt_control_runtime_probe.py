"""Verify sampling-time Prompt Control UUID and WeightHook observation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from tools.attention_coupling_benchmark.comfy_probe.prompt_control_runtime import (
    InstrumentPromptControlModelV3,
    ReadPromptControlRuntimeV3,
)


@dataclass
class _Keyframe:
    """Provide one native keyframe shape."""

    start_percent: float
    strength: float
    guarantee_steps: int = 1


@dataclass
class _Keyframes:
    """Expose keyframes and current schedule strength."""

    keyframes: list[_Keyframe]


@dataclass
class _Hook:
    """Provide mutable WeightHook-compatible fields."""

    _strength_model: float = 0.75
    _strength_clip: float = 0.25
    current: float = 0.0
    hook_keyframe: _Keyframes = field(
        default_factory=lambda: _Keyframes([_Keyframe(0.0, 0.0), _Keyframe(0.25, 1.0)])
    )
    hook_ref: object = field(default_factory=object)
    hook_id: str | None = None
    hook_scope: object = None

    @property
    def strength_model(self) -> float:
        """Return the mutable effective model strength."""

        return self._strength_model * self.current

    @property
    def strength_clip(self) -> float:
        """Return the mutable effective CLIP strength."""

        return self._strength_clip * self.current


@dataclass
class _Hooks:
    """Expose ordered hooks."""

    hooks: list[_Hook]


class _Model:
    """Provide the keyed ModelPatcher and wrapper surface."""

    def __init__(self) -> None:
        """Initialize callbacks and one scheduled hook."""

        self.model_options: dict[str, Any] = {}
        self.hook = _Hook()
        self.hooks = _Hooks([self.hook])
        self.callbacks: dict[str, Any] = {}
        self.wrapper: Any = None

    def clone(self) -> _Model:
        """Return a shell sharing the hook for test transitions."""

        clone = _Model()
        clone.hook = self.hook
        clone.hooks = self.hooks
        return clone

    def add_callback_with_key(self, event: str, key: str, callback: Any) -> None:
        """Record a callback by event."""

        del key
        self.callbacks[event] = callback

    def set_model_unet_function_wrapper(self, wrapper: Any) -> None:
        """Record the preserving wrapper."""

        self.wrapper = wrapper

    def apply_hooks(self) -> None:
        """Trigger Comfy's scheduled-hook callback shape."""

        self.callbacks["on_apply_hooks"](self, self.hooks)

    def register_hooks(self) -> None:
        """Trigger Comfy's registration callback shape."""

        self.callbacks["on_register_all_hook_patches"](
            self, self.hooks, {}, {}, self.hooks
        )


def test_runtime_probe_records_stable_uuids_and_schedule_transitions() -> None:
    """Observe actual call metadata without changing the returned tensor."""

    instrumented = InstrumentPromptControlModelV3.execute(
        _Model(), "pc-run", '["adapter-a"]'
    )
    (model,) = instrumented.result
    model.register_hooks()
    model.apply_hooks()
    tensor = torch.zeros((1, 2, 2, 2))
    arguments = {
        "input": tensor,
        "timestep": torch.tensor([1.0]),
        "c": {
            "transformer_options": {
                "uuids": ["uuid-a", "uuid-b"],
                "cond_or_uncond": [0, 1],
            }
        },
    }

    def apply_model(
        input_x: torch.Tensor, timestep: torch.Tensor, **conditioning: object
    ) -> torch.Tensor:
        """Return a recognizable tensor."""

        del timestep, conditioning
        return input_x + 1

    assert torch.equal(model.wrapper(apply_model, arguments), tensor + 1)
    model.hook.current = 1.0
    model.apply_hooks()
    arguments["timestep"] = torch.tensor([0.5])
    assert torch.equal(model.wrapper(apply_model, arguments), tensor + 1)
    measured = ReadPromptControlRuntimeV3.execute({"samples": tensor}, "pc-run")
    runtime = measured.ui["prompt_control_runtime"][0]
    assert runtime["model_call_count"] == 2
    assert runtime["calls"][0]["uuids"] == ["uuid-a", "uuid-b"]
    assert [
        item["effective_strengths"] for item in runtime["schedule_transitions"]
    ] == [
        [0.0],
        [0.75],
    ]


def test_runtime_probe_schemas_are_isolated_dev_nodes() -> None:
    """Expose only dedicated instrumentation and output contracts."""

    instrument = InstrumentPromptControlModelV3.define_schema()
    reader = ReadPromptControlRuntimeV3.define_schema()
    assert instrument.node_id == "SimpleSyrupBenchmark.InstrumentPromptControlModel"
    assert instrument.is_dev_only is True
    assert reader.node_id == "SimpleSyrupBenchmark.ReadPromptControlRuntime"
    assert reader.is_output_node is True
