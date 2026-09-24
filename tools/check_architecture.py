# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check repository architecture governance."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.architecture_governance.import_boundaries import validate_import_boundaries
from tools.architecture_governance.soft_reviews import validate_soft_reviews
from tools.architecture_governance.validation import validate_repository


def main(argv: Sequence[str] | None = None) -> int:
    """Validate current architecture state."""

    if argv:
        raise ValueError("The architecture checker accepts no arguments")
    root = Path(__file__).resolve().parents[1]
    diagnostics = [
        *validate_repository(root),
        *validate_import_boundaries(root),
        *validate_soft_reviews(root),
    ]
    for diagnostic in diagnostics:
        print(diagnostic.render())
    errors = [item for item in diagnostics if item.severity == "error"]
    if errors:
        print(f"FAILED: Found {len(errors)} architecture governance errors.")
        return 1
    warning_count = len(diagnostics)
    print(f"SUCCESS: Architecture is valid ({warning_count} structural warnings).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
