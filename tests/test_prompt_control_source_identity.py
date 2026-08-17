# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the pinned installed Prompt Control source identity owner."""

from pathlib import Path

import pytest

from tools.comfy_integration.default_paths import default_custom_node_root
from tools.prompt_control_characterization.source_identity import (
    inspect_source,
    validate_pinned_source,
)


def test_installed_prompt_control_matches_pinned_tracked_tree() -> None:
    """Pin version, path count, and canonical path-plus-content digest."""

    identity = inspect_source(default_custom_node_root("ComfyUI-Prompt-Control"))
    validate_pinned_source(identity)
    assert identity.version == "3.0.0-beta.3"
    assert identity.tracked_path_count == 53


def test_source_identity_rejects_non_install(tmp_path: Path) -> None:
    """Fail closed when required install identity files are absent."""

    with pytest.raises(ValueError, match=".tracking"):
        inspect_source(tmp_path)
