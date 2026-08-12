"""Verify the benchmark-only Comfy model-call and resource probe."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tools.attention_coupling_benchmark.comfy_probe.sampling_metrics import (
    InstrumentModelV3,
    ReadMetricsV3,
)


class _FakeModel:
    """Provide the model-patcher surface required by the probe."""

    def __init__(self) -> None:
        """Initialize empty model options and no installed wrapper."""

        self.model_options: dict[str, Any] = {}
        self.wrapper: Any = None

    def clone(self) -> _FakeModel:
        """Return an independent fake model with copied options."""

        cloned = _FakeModel()
        cloned.model_options = self.model_options.copy()
        return cloned

    def set_model_unet_function_wrapper(self, wrapper: Any) -> None:
        """Record the wrapper installed on the cloned model."""

        self.wrapper = wrapper


def test_probe_counts_actual_wrapper_calls_and_returns_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count each underlying model call and pass the measured latent through."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    instrumented = InstrumentModelV3.execute(_FakeModel(), "run-1")
    (model,) = instrumented.result
    first_batch = torch.zeros((1, 1, 2, 2))
    second_batch = torch.zeros((3, 1, 2, 2))

    def apply_model(
        input_x: torch.Tensor,
        timestep: torch.Tensor,
        **conditioning: object,
    ) -> torch.Tensor:
        """Return a recognizable raw-model result."""

        del timestep, conditioning
        return input_x + 1

    arguments = {
        "input": first_batch,
        "timestep": torch.ones((1,)),
        "c": {},
    }
    assert torch.equal(model.wrapper(apply_model, arguments), first_batch + 1)
    arguments["input"] = second_batch
    assert torch.equal(model.wrapper(apply_model, arguments), second_batch + 1)
    latent = {"samples": first_batch}

    measured = ReadMetricsV3.execute(latent, "run-1")

    assert measured.result[0] is latent
    assert measured.ui["benchmark_metrics"][0]["model_call_count"] == 2
    assert measured.ui["benchmark_metrics"][0]["model_input_batch_sizes"] == [1, 3]
    assert measured.ui["benchmark_metrics"][0]["model_input_batch_elements"] == 4
    assert measured.ui["benchmark_metrics"][0]["peak_vram_bytes"] == 0
    assert measured.ui["benchmark_metrics"][0]["runtime_ms"] >= 0


def test_probe_rejects_model_calls_without_tensor_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before delegation when the host wrapper omits its tensor input."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    instrumented = InstrumentModelV3.execute(_FakeModel(), "run-invalid-input")
    (model,) = instrumented.result

    with pytest.raises(TypeError, match="input must be a tensor"):
        model.wrapper(lambda *_args, **_kwargs: torch.zeros(1), {"input": object()})

    ReadMetricsV3.execute({"samples": torch.zeros(1)}, "run-invalid-input")


def test_probe_schemas_remain_benchmark_only_v3_nodes() -> None:
    """Expose only the two dev-only v3 probe contracts."""

    instrument = InstrumentModelV3.define_schema()
    metrics = ReadMetricsV3.define_schema()

    assert instrument.node_id == "SimpleSyrupBenchmark.InstrumentModel"
    assert instrument.is_dev_only is True
    assert metrics.node_id == "SimpleSyrupBenchmark.ReadMetrics"
    assert metrics.is_dev_only is True
    assert metrics.is_output_node is True
