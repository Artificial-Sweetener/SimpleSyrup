# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest configuration for SimpleSyrup tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PROJECT_ROOT.parents[1]
CUSTOM_NODES_ROOT = PROJECT_ROOT.parent

for path in (PROJECT_ROOT, COMFY_ROOT, CUSTOM_NODES_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

if os.environ.get("SIMPLE_SYRUP_TEST_COMFY_CPU") == "1":
    from comfy.cli_args import args

    args.cpu = True
