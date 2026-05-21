# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Logging helpers for SimpleSyrup modules."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a SimpleSyrup child logger without configuring global logging."""

    return logging.getLogger(f"simple_syrup.{name}")
