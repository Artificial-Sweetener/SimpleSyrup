"""Verify installed Sage override delegation and observation."""

from __future__ import annotations

import pytest
import torch

from tools.anima_regional_lora_performance import attention_backend as backend_module
from tools.anima_regional_lora_performance.attention_backend import (
    SagePerformanceAttentionOverride,
)


def test_sage_override_records_exact_shape_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve q/k/v/heads while publishing no tensor contents."""

    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def sage(*args: object, **kwargs: object) -> torch.Tensor:
        """Capture installed-call arguments and return a tensor result."""

        captured.append((args, kwargs))
        return torch.ones((1, 2, 4))

    monkeypatch.setattr(backend_module, "attention_sage", sage)
    override = SagePerformanceAttentionOverride()
    query = torch.zeros((1, 2, 4))
    key = torch.zeros((1, 3, 4))
    value = torch.zeros((1, 3, 4))

    output = override(lambda *args, **kwargs: torch.zeros(1), query, key, value, 2)

    assert tuple(output.shape) == (1, 2, 4)
    assert captured == [((query, key, value, 2), {})]
    assert override.call_count == 1
    assert override.calls[0].query_shape == (1, 2, 4)
    assert override.calls[0].key_shape == (1, 3, 4)
    assert override.calls[0].heads == 2


def test_sage_override_rejects_malformed_wrapper_call() -> None:
    """Fail before installed attention when q/k/v/heads are unavailable."""

    with pytest.raises(TypeError, match="requires q, k, v, and heads"):
        SagePerformanceAttentionOverride()(lambda: torch.zeros(1))
