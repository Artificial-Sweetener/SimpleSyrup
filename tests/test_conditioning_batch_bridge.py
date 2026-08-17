# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify canonical conditioning-batch normalization across host namespaces."""

from __future__ import annotations

import pytest

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from tools.attention_coupling_benchmark.comfy_probe.conditioning_batch_bridge import (
    normalize_conditioning_batch,
)


def test_local_batch_is_preserved_and_plain_conditioning_passes_through() -> None:
    """Avoid copying local domain values or reinterpreting ordinary conditioning."""

    batch = ConditioningBatch((object(),))
    conditioning = object()

    assert normalize_conditioning_batch(batch) is batch
    assert normalize_conditioning_batch(conditioning) is conditioning


def test_canonical_host_namespace_is_rebuilt_as_the_local_domain_type() -> None:
    """Bridge only the same immutable domain surface loaded by Comfy's host."""

    host_type = type(
        "ConditioningBatch",
        (),
        {
            "__module__": (
                "custom_nodes.SimpleSyrup.simple_syrup.domain.conditioning_batch"
            )
        },
    )
    host_value = host_type()
    entries = (object(), object())
    host_value.entries = entries

    normalized = normalize_conditioning_batch(host_value)

    assert isinstance(normalized, ConditioningBatch)
    assert normalized.entries == entries


@pytest.mark.parametrize(
    ("name", "module", "entries", "message"),
    [
        (
            "OtherBatch",
            "simple_syrup.domain.conditioning_batch",
            (object(),),
            "runtime type",
        ),
        (
            "ConditioningBatch",
            "example.conditioning_batch",
            (object(),),
            "runtime type",
        ),
        (
            "ConditioningBatch",
            "simple_syrup.domain.conditioning_batch",
            [object()],
            "immutable tuple",
        ),
        (
            "ConditioningBatch",
            "simple_syrup.domain.conditioning_batch",
            (),
            "must not be empty",
        ),
    ],
)
def test_malformed_or_noncanonical_batch_surfaces_fail_closed(
    name: str,
    module: str,
    entries: object,
    message: str,
) -> None:
    """Reject lookalikes and mutable or empty canonical surfaces."""

    value_type = type(name, (), {"__module__": module})
    value = value_type()
    value.entries = entries

    with pytest.raises((TypeError, ValueError), match=message):
        normalize_conditioning_batch(value)
