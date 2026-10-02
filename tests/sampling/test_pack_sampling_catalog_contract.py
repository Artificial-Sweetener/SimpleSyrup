# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Enforce automatic pack-wide sampler and scheduler menu availability."""

from __future__ import annotations

from typing import Any, Protocol, cast

import comfy.samplers
import pytest

from simple_syrup import nodes_v3
from simple_syrup.runtime import sampling_samplers, sampling_schedulers


class SamplingNode(Protocol):
    """Describe the host-facing schema boundary inspected by this contract."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare sampling controls without executing a workflow."""


def sampling_nodes() -> tuple[type[SamplingNode], ...]:
    """Discover every exported KSampler and detailer without a manual inventory."""
    nodes = tuple(
        cast(type[SamplingNode], node)
        for node in nodes_v3.get_nodes()
        if node.__name__.startswith(("KSampler", "Detail"))
    )
    assert nodes, "The pack must export sampling nodes."
    return nodes


@pytest.mark.parametrize("prompt_control_available", [False, True])
def test_every_exported_sampling_node_uses_complete_catalogs(
    monkeypatch: pytest.MonkeyPatch,
    prompt_control_available: bool,
) -> None:
    """Keep all registered sampling menus aligned in both integration states."""
    monkeypatch.setattr(
        nodes_v3, "prompt_control_is_available", lambda: prompt_control_available
    )
    for node in sampling_nodes():
        schema = node.define_schema()
        inputs = {item.id: item for item in schema.inputs}
        assert tuple(inputs["sampler_name"].options) == (
            sampling_samplers.available_samplers()
        ), schema.node_id
        assert tuple(inputs["scheduler"].options) == (
            sampling_schedulers.available_schedulers()
        ), schema.node_id


@pytest.mark.parametrize("source", ["pack", "res4lyf", "host"])
def test_catalog_additions_propagate_without_node_edits(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    """Rebuild every menu from changed catalogs, preserving all other contracts."""
    monkeypatch.setattr(nodes_v3, "prompt_control_is_available", lambda: False)
    nodes = sampling_nodes()
    original = {node: node.define_schema() for node in nodes}
    sampler_name = "catalog_test_sampler"
    scheduler_name = "catalog_test_scheduler"

    if source == "host":
        monkeypatch.setattr(
            comfy.samplers.KSampler,
            "SAMPLERS",
            (*comfy.samplers.KSampler.SAMPLERS, sampler_name, sampler_name),
        )
        monkeypatch.setattr(
            comfy.samplers.KSampler,
            "SCHEDULERS",
            (*comfy.samplers.KSampler.SCHEDULERS, scheduler_name, scheduler_name),
        )
    else:
        catalog = "RES4LYF_SAMPLER_NAMES" if source == "res4lyf" else "EXTRA_SAMPLERS"
        monkeypatch.setattr(
            sampling_samplers,
            catalog,
            (*getattr(sampling_samplers, catalog), sampler_name, sampler_name),
        )
        monkeypatch.setattr(
            sampling_schedulers,
            "EXTRA_SCHEDULERS",
            (*sampling_schedulers.EXTRA_SCHEDULERS, scheduler_name, scheduler_name),
        )

    for node in nodes:
        schema = node.define_schema()
        previous = original[node]
        assert schema.node_id == previous.node_id
        assert [item.as_dict() for item in schema.outputs] == [
            item.as_dict() for item in previous.outputs
        ]
        assert [item.id for item in schema.inputs] == [
            item.id for item in previous.inputs
        ]
        inputs = {item.id: item for item in schema.inputs}
        assert tuple(inputs["sampler_name"].options) == (
            sampling_samplers.available_samplers()
        ), schema.node_id
        assert tuple(inputs["scheduler"].options) == (
            sampling_schedulers.available_schedulers()
        ), schema.node_id
        assert inputs["sampler_name"].options.count(sampler_name) == 1
        assert inputs["scheduler"].options.count(scheduler_name) == 1
        for current, prior in zip(schema.inputs, previous.inputs, strict=True):
            assert current.optional == prior.optional
            assert getattr(current, "default", None) == getattr(prior, "default", None)
