"""Provide RES4LYF solver logging without registering its ComfyUI nodes."""

from __future__ import annotations

import logging

LOGGER = logging.getLogger("SimpleSyrup.RES4LYF")


def RESplain(*values: object, debug: bool = False, **_kwargs: object) -> None:
    """Log solver diagnostics at the upstream-requested verbosity."""

    message = " ".join(str(value) for value in values)
    LOGGER.log(logging.DEBUG if debug else logging.INFO, message)


def is_debug_logging_enabled() -> bool:
    """Report whether solver debug diagnostics are enabled."""

    return LOGGER.isEnabledFor(logging.DEBUG)


def get_display_sampler_category() -> bool:
    """Keep upstream sampler names stable without UI category mutation."""

    return False
