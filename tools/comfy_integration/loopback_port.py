# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reserve and validate non-default loopback TCP ports."""

from __future__ import annotations

import socket

PROTECTED_COMFY_PORTS = frozenset({8188, 8297})


class LoopbackPortReservation:
    """Own an exclusive loopback socket until the managed launcher is ready."""

    def __init__(self, listener: socket.socket) -> None:
        """Retain the bound socket and its OS-assigned port."""

        self._listener: socket.socket | None = listener
        self.port = int(listener.getsockname()[1])

    def release(self) -> None:
        """Release the reservation immediately before the child launch attempt."""

        listener = self._listener
        if listener is not None:
            self._listener = None
            listener.close()

    def __enter__(self) -> LoopbackPortReservation:
        """Expose the live reservation to its lifecycle owner."""

        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Release the reservation on every owner exit path."""

        del exc_type, exc, traceback
        self.release()


def reserve_loopback_port() -> LoopbackPortReservation:
    """Return one live ephemeral reservation outside protected Comfy ports."""

    for _ in range(16):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            port = int(listener.getsockname()[1])
        except BaseException:
            listener.close()
            raise
        if port not in PROTECTED_COMFY_PORTS:
            return LoopbackPortReservation(listener)
        listener.close()
    raise RuntimeError("Unable to reserve a non-default loopback port.")


def validate_loopback_port(port: int) -> None:
    """Reject protected, privileged, or invalid managed-server ports."""

    if (
        isinstance(port, bool)
        or not 1024 <= port <= 65535
        or port in PROTECTED_COMFY_PORTS
    ):
        raise ValueError(
            "Managed Comfy port must be non-protected and between 1024 and 65535."
        )


def is_loopback_port_available(port: int) -> bool:
    """Return whether the managed loopback port can be bound after cleanup."""

    validate_loopback_port(port)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
    except OSError:
        return False
    return True
