# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Load WD14 Tagger Comfy v3 wrapper."""

from __future__ import annotations

from typing import Any

from simple_syrup.nodes.wd14_tagger_loader import WD14TaggerLoader
from simple_syrup.nodes_v3.wd14_tagger_loader import WD14TaggerLoaderV3


def test_wd14_tagger_loader_v3_schema() -> None:
    """The v3 loader schema exposes the WD14 tagger loader contract."""

    schema = WD14TaggerLoaderV3.define_schema()

    assert schema.node_id == "SimpleSyrup.WD14TaggerLoader"
    assert schema.display_name == "Load WD14 Tagger"
    assert [input_item.id for input_item in schema.inputs] == ["wd14_model"]
    assert schema.inputs[0].default == "wd-eva02-large-tagger-v3"
    assert [output.id for output in schema.outputs] == ["wd14_tagger"]
    assert schema.outputs[0].io_type == "WD14_TAGGER"


def test_wd14_tagger_loader_v3_execute_forwards_to_legacy_loader(
    monkeypatch: Any,
) -> None:
    """The v3 loader wrapper forwards execution to the legacy loader."""

    expected = object()

    class FakeService:
        """Service double for the legacy loader."""

        def __init__(self) -> None:
            """Initialize captured kwargs."""

            self.kwargs: dict[str, object] = {}

        def load_model(self, **kwargs: object) -> object:
            """Return a fixed loaded tagger object."""

            self.kwargs = kwargs
            return expected

    fake_service = FakeService()
    monkeypatch.setattr(WD14TaggerLoader, "_service", fake_service)

    assert WD14TaggerLoaderV3.execute("wd-eva02-large-tagger-v3") == (expected,)
    assert fake_service.kwargs["wd14_model"] == "wd-eva02-large-tagger-v3"
