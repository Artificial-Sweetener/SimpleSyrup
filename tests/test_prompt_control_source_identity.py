"""Verify the pinned installed Prompt Control source identity owner."""

from pathlib import Path

import pytest

from tools.prompt_control_characterization.source_identity import (
    inspect_source,
    validate_pinned_source,
)


def test_installed_prompt_control_matches_pinned_tracked_tree() -> None:
    """Pin version, path count, and canonical path-plus-content digest."""

    identity = inspect_source(Path(r"<COMFY_ROOT>\custom_nodes\ComfyUI-Prompt-Control"))
    validate_pinned_source(identity)
    assert identity.version == "3.0.0-beta.3"
    assert identity.tracked_path_count == 53


def test_source_identity_rejects_non_install(tmp_path: Path) -> None:
    """Fail closed when required install identity files are absent."""

    with pytest.raises(ValueError, match=".tracking"):
        inspect_source(tmp_path)
