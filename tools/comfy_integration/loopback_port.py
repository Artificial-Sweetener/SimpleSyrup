# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Select and validate unused non-default loopback TCP ports."""

from __future__ import annotations

import socket

PROTECTED_COMFY_PORTS = frozenset({8188, 8297})


def select_unused_loopback_port() -> int:
    """Return one unused ephemeral port outside protected Comfy instances."""

    for _ in range(16):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        if port not in PROTECTED_COMFY_PORTS:
            return port
    raise RuntimeError("Unable to select a non-default loopback port.")


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
