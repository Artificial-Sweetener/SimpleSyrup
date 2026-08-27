# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the Anima compound-outfit token diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from .client import execute
from .phrase_diagnostic import (
    build_phrase_diagnostic_workflow,
    compose_phrase_diagnostic_sheet,
)
from .run import OUTPUT_ROOT, PROOF_ROOT, SERVER

CONCEPTS = ("pink", "blue", "witch", "outfit", "pink and blue witch outfit")
OUTFIT_STRENGTHS = (0.15, 0.30, 0.50, 0.70, 0.85)


def main() -> None:
    """Execute and persist the fixed Anima compound-phrase comparison."""

    source = Path("E:/devprojects/attention-atlas/fixtures/source/pink-witch.json")
    prefix = "simple_syrup_attention_cohesion_proof/anima_outfit_tokens"
    outputs = execute(
        SERVER,
        build_phrase_diagnostic_workflow(
            source,
            output_prefix=prefix,
            concepts=CONCEPTS,
        ),
        OUTPUT_ROOT,
    )
    concept_paths = tuple(
        (concept, outputs[str(3000 + index)])
        for index, concept in enumerate(CONCEPTS, start=1)
    )
    sheet = PROOF_ROOT / "anima_outfit_token_diagnostic.png"
    compose_phrase_diagnostic_sheet(
        title="Hassaku Anima - Compound Outfit Token Diagnostic",
        image_path=outputs["900"],
        concept_paths=concept_paths,
        destination=sheet,
    )
    manifest = {
        "sheet": str(sheet),
        "outputs": {node_id: str(path) for node_id, path in outputs.items()},
    }
    destination = PROOF_ROOT / "anima_outfit_token_diagnostic.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(destination)

    strength_prefix = "simple_syrup_attention_cohesion_proof/anima_outfit_strength"
    strength_outputs = execute(
        SERVER,
        build_phrase_diagnostic_workflow(
            source,
            output_prefix=strength_prefix,
            concepts=(CONCEPTS[-1],) * len(OUTFIT_STRENGTHS),
            strengths=OUTFIT_STRENGTHS,
        ),
        OUTPUT_ROOT,
    )
    strength_paths = tuple(
        (f"strength {strength:.2f}", strength_outputs[str(3000 + index)])
        for index, strength in enumerate(OUTFIT_STRENGTHS, start=1)
    )
    strength_sheet = PROOF_ROOT / "anima_outfit_strength_diagnostic.png"
    compose_phrase_diagnostic_sheet(
        title="Hassaku Anima - Compound Outfit Strength Sweep",
        image_path=strength_outputs["900"],
        concept_paths=strength_paths,
        destination=strength_sheet,
    )
    strength_manifest = {
        "sheet": str(strength_sheet),
        "outputs": {node_id: str(path) for node_id, path in strength_outputs.items()},
    }
    strength_destination = PROOF_ROOT / "anima_outfit_strength_diagnostic.json"
    strength_destination.write_text(
        json.dumps(strength_manifest, indent=2),
        encoding="utf-8",
    )
    print(strength_destination)


if __name__ == "__main__":
    main()
