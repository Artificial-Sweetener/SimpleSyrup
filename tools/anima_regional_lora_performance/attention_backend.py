"""Adapt and observe the installed Sage attention override boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from comfy.ldm.modules.attention import attention_sage


@dataclass(frozen=True, slots=True)
class PerformanceAttentionCall:
    """Retain one tensor-only attention invocation signature."""

    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    heads: int
    dtype: str
    device: str


class SagePerformanceAttentionOverride:
    """Delegate to installed Sage attention while observing exact call shapes."""

    def __init__(self) -> None:
        """Create an empty per-profile invocation ledger."""

        self._calls: list[PerformanceAttentionCall] = []

    def __call__(
        self,
        original: Callable[..., torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Record the wrapper call and execute installed Sage attention."""

        if not callable(original):
            raise TypeError("Sage performance override requires an original callable.")
        if len(args) < 4:
            raise TypeError("Sage performance override requires q, k, v, and heads.")
        query, key, value, heads = args[:4]
        if not all(isinstance(tensor, torch.Tensor) for tensor in (query, key, value)):
            raise TypeError("Sage performance override requires tensor q, k, and v.")
        if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
            raise TypeError("Sage performance override requires a positive head count.")
        query_tensor = query
        key_tensor = key
        value_tensor = value
        self._calls.append(
            PerformanceAttentionCall(
                tuple(query_tensor.shape),
                tuple(key_tensor.shape),
                tuple(value_tensor.shape),
                heads,
                str(query_tensor.dtype),
                str(query_tensor.device),
            )
        )
        output = attention_sage(*args, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Installed Sage attention returned a non-tensor value.")
        return output

    @property
    def call_count(self) -> int:
        """Return the exact observed override invocation count."""

        return len(self._calls)

    @property
    def calls(self) -> tuple[PerformanceAttentionCall, ...]:
        """Return an immutable snapshot of observed tensor signatures."""

        return tuple(self._calls)
