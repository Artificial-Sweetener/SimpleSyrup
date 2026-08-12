"""Test isolated loopback-port selection and validation."""

from __future__ import annotations

import socket

import pytest

from tools.comfy_integration.loopback_port import (
    is_loopback_port_available,
    select_unused_loopback_port,
    validate_loopback_port,
)


def test_selected_port_is_nondefault_and_unused() -> None:
    """Prove selection returns a bindable non-default loopback port."""

    port = select_unused_loopback_port()

    assert port not in {8188, 8297}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))
        assert not is_loopback_port_available(port)
    assert is_loopback_port_available(port)


@pytest.mark.parametrize("port", [True, 0, 1023, 8188, 8297, 65536])
def test_invalid_managed_ports_are_rejected(port: int) -> None:
    """Reject unsafe and default managed-server ports."""

    with pytest.raises(ValueError, match="non-protected"):
        validate_loopback_port(port)
