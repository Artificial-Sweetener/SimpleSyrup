# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the focused SDXL related-token localization diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from .client import execute
from .concept_diagnostic import (
    build_related_concept_workflow,
    compose_related_concept_sheet,
)

OUTPUT_ROOT = Path("E:/ComfyUI/output")
PROOF_ROOT = OUTPUT_ROOT / "simple_syrup_attention_cohesion_proof"
SERVER = "http://127.0.0.1:8207"


def main() -> None:
    """Execute one generation and persist its related-token comparison."""

    prefix = "simple_syrup_attention_cohesion_proof/sdxl_related_tokens"
    source = (
        OUTPUT_ROOT
        / "simple_syrup_attention_acceptance_v2/sdxl/final_many_image_00002_.png"
    )
    outputs = execute(
        SERVER,
        build_related_concept_workflow(source, output_prefix=prefix),
        OUTPUT_ROOT,
    )
    sheet = PROOF_ROOT / "sdxl_related_token_diagnostic.png"
    compose_related_concept_sheet(
        image_path=outputs["900"],
        hair_path=outputs["101"],
        pink_hair_path=outputs["1101"],
        twintails_path=outputs["1131"],
        combined_path=outputs["1141"],
        destination=sheet,
    )
    manifest = {
        "sheet": str(sheet),
        "outputs": {node_id: str(path) for node_id, path in outputs.items()},
    }
    destination = PROOF_ROOT / "sdxl_related_token_diagnostic.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
