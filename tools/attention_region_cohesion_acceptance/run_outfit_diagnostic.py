# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the Anima compound-outfit token diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

from .client import execute
from .fidelity_diagnostic import compose_mask_diagnostic_sheet
from .phrase_diagnostic import (
    build_phrase_diagnostic_workflow,
    compose_phrase_diagnostic_sheet,
)
from .run import OUTPUT_ROOT, PROOF_ROOT, SERVER
from .source_workflow import build_source_workflow
from .workflow import configure_mask

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

    default_graph = build_source_workflow(
        source,
        output_prefix="simple_syrup_attention_cohesion_proof/anima_outfit_default",
        concept=CONCEPTS[-1],
    )
    configure_mask(
        default_graph,
        request_id="14",
        save_id="1131",
        concept=CONCEPTS[-1],
        strength=0.15,
        consensus=0.25,
        split=0.0,
        minimum_size=512,
        keep_only=1,
        solidity=0.75,
        evidence_mode="concept isolation",
        prefix="simple_syrup_attention_cohesion_proof/anima_outfit_default/mask",
        edge_feather=8,
    )
    default_graph = {
        node_id: node
        for node_id, node in default_graph.items()
        if node_id in {"1", "2", "3", "4", "5", "6", "14", "900", "1130", "1131"}
    }
    default_outputs = execute(
        SERVER,
        default_graph,
        OUTPUT_ROOT,
    )
    default_sheet = PROOF_ROOT / "anima_outfit_default_mask.png"
    compose_mask_diagnostic_sheet(
        title="Hassaku Anima - Default Compound Outfit Mask",
        subtitle=(
            "Strength 0.15; consensus 0.25; minimum region 512; "
            "largest cohesive instance; solidity 0.75; feather 8"
        ),
        image_path=default_outputs["900"],
        stage_paths=(("DEFAULT MASK", default_outputs["1131"]),),
        destination=default_sheet,
    )
    default_destination = PROOF_ROOT / "anima_outfit_default_mask.json"
    default_destination.write_text(
        json.dumps(
            {
                "sheet": str(default_sheet),
                "outputs": {
                    node_id: str(path) for node_id, path in default_outputs.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(default_destination)


if __name__ == "__main__":
    main()
