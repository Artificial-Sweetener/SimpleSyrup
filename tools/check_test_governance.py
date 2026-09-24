# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check deterministic test governance and reviewed execution state."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.test_governance.validation import validate_test_governance


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the repository's exact current test-governance state."""

    if argv:
        raise ValueError("The test-governance checker accepts no arguments")
    root = Path(__file__).resolve().parents[1]
    result = validate_test_governance(root)
    for diagnostic in result.diagnostics:
        print(diagnostic.render())
    errors = [item for item in result.diagnostics if item.severity == "error"]
    if errors:
        print(f"FAILED: Found {len(errors)} test-governance errors.")
        return 1
    print(
        "SUCCESS: Test governance is valid "
        f"({len(result.candidates)} reviewed candidates)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
