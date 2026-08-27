# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the fixed Anima relation-and-object attention diagnostic."""

from __future__ import annotations

import json

from .artifacts import compose_before_after_sheet
from .benchmark_workflow import build_benchmark_proof
from .client import execute
from .fidelity_diagnostic import (
    build_split_diagnostic_workflow,
    compose_mask_diagnostic_sheet,
)
from .phrase_diagnostic import (
    build_saved_phrase_diagnostic_workflow,
    compose_phrase_diagnostic_sheet,
)
from .run import OUTPUT_ROOT, PROOF_ROOT, SERVER

CONCEPTS = ("cat", "holding", "holding cat")
SPLIT_SENSITIVITIES = (0.0, 0.15, 0.35, 0.65)
STRENGTHS = (0.02, 0.04, 0.06, 0.09)


def main() -> None:
    """Compare native object, relation, and phrase evidence on one generation."""

    source = (
        OUTPUT_ROOT
        / "simple_syrup_attention_acceptance_v2/anima/final_many_image_00002_.png"
    )
    prefix = "simple_syrup_attention_cohesion_proof/anima_cat_native_evidence"
    outputs = execute(
        SERVER,
        build_saved_phrase_diagnostic_workflow(
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
    sheet = PROOF_ROOT / "anima_cat_native_evidence_diagnostic.png"
    compose_phrase_diagnostic_sheet(
        title="Hassaku Anima - Native Cat Phrase Evidence",
        image_path=outputs["900"],
        concept_paths=concept_paths,
        destination=sheet,
    )
    destination = PROOF_ROOT / "anima_cat_native_evidence_diagnostic.json"
    destination.write_text(
        json.dumps(
            {
                "sheet": str(sheet),
                "outputs": {node_id: str(path) for node_id, path in outputs.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(destination)

    split_prefix = "simple_syrup_attention_cohesion_proof/anima_cat_split"
    split_outputs = execute(
        SERVER,
        build_split_diagnostic_workflow(
            source,
            output_prefix=split_prefix,
            concept="holding cat",
            sensitivities=SPLIT_SENSITIVITIES,
        ),
        OUTPUT_ROOT,
    )
    split_sheet = PROOF_ROOT / "anima_cat_split_diagnostic.png"
    compose_mask_diagnostic_sheet(
        title="Hassaku Anima - Cat Instance Split Diagnostic",
        subtitle=(
            "Same phrase evidence; strength 0.15; consensus 0.25; "
            "minimum region 512; keep largest one"
        ),
        image_path=split_outputs["900"],
        stage_paths=tuple(
            (
                f"SPLIT {sensitivity:.2f}",
                split_outputs[save_id],
            )
            for sensitivity, save_id in zip(
                SPLIT_SENSITIVITIES,
                ("101", "1101", "1131", "1141"),
                strict=True,
            )
        ),
        destination=split_sheet,
    )
    split_destination = PROOF_ROOT / "anima_cat_split_diagnostic.json"
    split_destination.write_text(
        json.dumps(
            {
                "sheet": str(split_sheet),
                "outputs": {
                    node_id: str(path) for node_id, path in split_outputs.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(split_destination)

    strength_prefix = "simple_syrup_attention_cohesion_proof/anima_cat_strength"
    strength_outputs = execute(
        SERVER,
        build_saved_phrase_diagnostic_workflow(
            source,
            output_prefix=strength_prefix,
            concepts=("holding cat",) * len(STRENGTHS),
            strengths=STRENGTHS,
        ),
        OUTPUT_ROOT,
    )
    strength_sheet = PROOF_ROOT / "anima_cat_strength_diagnostic.png"
    compose_mask_diagnostic_sheet(
        title="Hassaku Anima - Cat Strength Diagnostic",
        subtitle=(
            "Same phrase evidence; no splitting, component filtering, solidity, "
            "or feathering"
        ),
        image_path=strength_outputs["900"],
        stage_paths=tuple(
            (
                f"STRENGTH {strength:.2f}",
                strength_outputs[str(3000 + index)],
            )
            for index, strength in enumerate(STRENGTHS, start=1)
        ),
        destination=strength_sheet,
    )
    strength_destination = PROOF_ROOT / "anima_cat_strength_diagnostic.json"
    strength_destination.write_text(
        json.dumps(
            {
                "sheet": str(strength_sheet),
                "outputs": {
                    node_id: str(path) for node_id, path in strength_outputs.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(strength_destination)

    corrected_graph = build_benchmark_proof(
        source,
        concept="holding cat",
        seed=992_702,
        output_prefix="simple_syrup_attention_cohesion_proof/anima_cat_corrected",
    )
    corrected_outputs = execute(SERVER, corrected_graph, OUTPUT_ROOT)
    comparison_sheet = PROOF_ROOT / "anima_cat_before_after.png"
    compose_before_after_sheet(
        title="Hassaku Anima - Cat Mask Regression",
        subtitle=(
            "Same model, prompt, seed, and generated pixels; corrected concept "
            "support and non-destructive default splitting"
        ),
        image_path=corrected_outputs["900"],
        before_path=PROOF_ROOT / "anima_cat_before_phrase_fix.png",
        after_path=corrected_outputs["1131"],
        destination=comparison_sheet,
    )
    comparison_destination = PROOF_ROOT / "anima_cat_before_after.json"
    comparison_destination.write_text(
        json.dumps(
            {
                "sheet": str(comparison_sheet),
                "before": str(PROOF_ROOT / "anima_cat_before_phrase_fix.png"),
                "after": str(corrected_outputs["1131"]),
                "image": str(corrected_outputs["900"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(comparison_destination)


if __name__ == "__main__":
    main()
