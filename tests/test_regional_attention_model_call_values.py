"""Prove exact model-call sigma and conditioning UUID normalization."""

from __future__ import annotations

from uuid import uuid1, uuid4

import pytest
import torch

from simple_syrup.runtime.regional_attention_model_call_values import (
    conditioning_uuid_strings,
    regional_attention_model_call_values,
    uniform_model_call_sigma,
)


def test_model_call_values_preserve_view_major_uuid_repetition() -> None:
    """Retain exact repeated UUID order produced by spatial view batching."""

    first = uuid4()
    second = uuid4()
    options = {
        "sigmas": torch.full((4,), 0.5),
        "uuids": [first, str(second), first, str(second)],
    }

    values = regional_attention_model_call_values(options)

    assert values.sampling_sigma == 0.5
    assert values.conditioning_uuids == (
        str(first),
        str(second),
        str(first),
        str(second),
    )


def test_model_call_values_accept_absent_optional_evidence() -> None:
    """Represent unavailable call metadata without inventing identities."""

    assert regional_attention_model_call_values(None).sampling_sigma is None
    assert regional_attention_model_call_values(None).conditioning_uuids == ()
    assert regional_attention_model_call_values({}).sampling_sigma is None
    assert regional_attention_model_call_values({}).conditioning_uuids == ()


@pytest.mark.parametrize(
    "value",
    [object(), ["not-a-uuid"], [uuid1()]],
)
def test_conditioning_uuid_strings_reject_invalid_host_values(value: object) -> None:
    """Require an ordered list or tuple containing only UUIDv4 values."""

    with pytest.raises((TypeError, ValueError), match="UUID|list or tuple"):
        conditioning_uuid_strings(value)


def test_uniform_sigma_accepts_scalar_and_broadcast_views() -> None:
    """Use scalar and zero-stride broadcast fast paths without materialization."""

    scalar = torch.tensor(2.5)
    broadcast = torch.tensor([1.25]).expand(8)

    assert uniform_model_call_sigma(scalar) == 2.5
    assert uniform_model_call_sigma(broadcast) == 1.25


@pytest.mark.parametrize(
    "value",
    [torch.tensor([]), torch.tensor([1.0, 0.5]), torch.tensor([float("nan")])],
)
def test_uniform_sigma_rejects_empty_nonuniform_and_nonfinite_values(
    value: torch.Tensor,
) -> None:
    """Fail before malformed sampling state enters diagnostics or schedules."""

    with pytest.raises((TypeError, ValueError), match="nonempty|uniform|finite"):
        uniform_model_call_sigma(value)
