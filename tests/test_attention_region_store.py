# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test attention capture session publication and cleanup ownership."""

from __future__ import annotations

import pytest

from simple_syrup.domain.attention_region_maps import CapturedAttentionMap
from simple_syrup.runtime.attention_region_store import (
    AttentionCaptureSession,
    AttentionRegionCaptureStore,
)


class _Session:
    """Provide the minimal downstream capture protocol."""

    @property
    def status_message(self) -> str:
        """Return a stable lifecycle-test status."""

        return "test session"

    def maps_for(self, request_node_id: str) -> tuple[CapturedAttentionMap, ...]:
        """Return no maps for a lifecycle-only test session."""

        del request_node_id
        return ()


def test_shared_session_is_consumed_once_per_coalesced_request() -> None:
    """Keep sibling request bindings alive until each public node executes."""

    store = AttentionRegionCaptureStore()
    session: AttentionCaptureSession = _Session()
    store.publish(("all", "search"), session)

    assert store.consume("search") is session
    assert store.consume("search") is None
    assert store.consume("all") is session


def test_store_rejects_overlapping_live_prompt_state() -> None:
    """Detect request-id collisions rather than routing maps across executions."""

    store = AttentionRegionCaptureStore()
    store.publish(("search",), _Session())

    with pytest.raises(RuntimeError, match="already active"):
        store.publish(("search",), _Session())
