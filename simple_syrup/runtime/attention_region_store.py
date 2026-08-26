# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own prompt-scoped attention capture sessions until public nodes consume them."""

from __future__ import annotations

from threading import RLock
from typing import Protocol

from ..domain.attention_region_maps import CapturedAttentionMap


class AttentionCaptureSession(Protocol):
    """Expose the read surface required by downstream attention nodes."""

    @property
    def status_message(self) -> str:
        """Return one concise user-facing capture status."""

    def maps_for(
        self,
        request_node_id: str,
    ) -> tuple[CapturedAttentionMap, ...]:
        """Return captured maps associated with one public request node."""


class AttentionRegionCaptureStore:
    """Publish and consume request sessions with deterministic cleanup."""

    def __init__(self) -> None:
        """Create an empty thread-safe session index."""

        self._sessions: dict[str, AttentionCaptureSession] = {}
        self._lock = RLock()

    def publish(
        self,
        request_node_ids: tuple[str, ...],
        session: AttentionCaptureSession,
    ) -> None:
        """Bind every unique request id to one shared capture session."""

        if not request_node_ids or request_node_ids != tuple(
            sorted(set(request_node_ids))
        ):
            raise ValueError(
                "Attention capture request ids must be unique and ordered."
            )
        with self._lock:
            collisions = tuple(
                node_id for node_id in request_node_ids if node_id in self._sessions
            )
            if collisions:
                raise RuntimeError(
                    "Attention capture request state is already active: "
                    + ", ".join(collisions)
                )
            for node_id in request_node_ids:
                self._sessions[node_id] = session

    def consume(self, request_node_id: str) -> AttentionCaptureSession | None:
        """Remove and return one request binding without affecting its siblings."""

        with self._lock:
            return self._sessions.pop(request_node_id, None)

    def clear(self) -> None:
        """Remove all sessions after an interrupted or completed prompt."""

        with self._lock:
            self._sessions.clear()


ATTENTION_REGION_CAPTURE_STORE = AttentionRegionCaptureStore()
