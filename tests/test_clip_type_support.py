# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ComfyUI CLIP-type compatibility validation."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from simple_syrup.runtime.clip_type_support import ComfyClipTypeSupport


def test_clip_type_support_accepts_installed_krea2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The current Comfy KREA2 enum satisfies loader preflight."""

    comfy_sd = ModuleType("comfy.sd")
    comfy_sd.CLIPType = SimpleNamespace(KREA2=object())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)

    ComfyClipTypeSupport().require("KREA2")


def test_clip_type_support_reports_actionable_update_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older Comfy builds fail explicitly before any artifact download."""

    comfy_sd = ModuleType("comfy.sd")
    comfy_sd.CLIPType = SimpleNamespace()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)

    with pytest.raises(RuntimeError, match="Update ComfyUI"):
        ComfyClipTypeSupport().require("KREA2")
