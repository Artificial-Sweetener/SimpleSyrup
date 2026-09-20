# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact user-prompt global LoRA proof graph mutations."""

from __future__ import annotations

from tools.run_global_prompt_lora_proof import (
    ProofCase,
    build_case_graph,
    cases,
    render_global_first_prompt,
)


def test_prompt_renderer_preserves_named_separators_and_places_both_scopes() -> None:
    """Keep global-first layout while preserving the user's named regions."""

    rendered = render_global_first_prompt(
        "global text[SEP|Taffy]left text[SEP|Anise]right text",
        global_tag="<lora:shared:0.5>",
        regional_tag="<lora:shared:0.8>",
    )

    assert rendered == (
        "<lora:shared:0.5>\nglobal text"
        "[SEP|Taffy]<lora:shared:0.8>\nleft text"
        "[SEP|Anise]right text"
    )


def test_matrix_covers_same_lora_turbo_tiled_and_contextual() -> None:
    """Keep all requested managed proof variants explicit and ordered."""

    definitions = cases()

    assert [case.case_id for case in definitions] == [
        "sdxl-global-and-regional-same-lora",
        "anima-global-and-regional-same-lora",
        "anima-turbo-global-arcane-regional-tiled",
        "anima-turbo-global-arcane-regional-contextual",
    ]
    assert definitions[0].global_tag == definitions[0].regional_tag
    assert "ArcaneViolet" in definitions[1].global_tag
    assert "ArcaneViolet" in definitions[1].regional_tag
    assert [case.turbo for case in definitions] == [False, False, True, True]
    assert [case.contextual for case in definitions] == [False, False, False, True]
    assert definitions[0].region_mask_feather == 10
    assert {case.region_mask_feather for case in definitions[1:]} == {64}


def test_turbo_contextual_graph_uses_appropriate_sampling_contract() -> None:
    """Apply Turbo's low-step CFG-one contract to full and contextual stages."""

    template = _anima_template()
    case = ProofCase(
        "fixture",
        "anima",
        "<lora:turbo:0.7>",
        "<lora:regional:0.8>",
        turbo=True,
        contextual=True,
    )

    graph, save_ids = build_case_graph(template, case, run_id="run")

    assert save_ids == ("proof:source", "proof:refinement")
    for node_id in (
        "anima-prompt-region:ksampler",
        "anima-diffusion-upscale:ksampler",
    ):
        inputs = graph[node_id]["inputs"]
        assert isinstance(inputs, dict)
        assert inputs["steps"] == 10
        assert inputs["cfg"] == 1.0
        assert inputs["sampler_name"] == "euler"
        assert inputs["scheduler"] == "simple"
        assert inputs["region_mask_feather"] == 64
    refinement = graph["anima-diffusion-upscale:ksampler"]
    assert refinement["class_type"] == "SimpleSyrup.KSamplerAttentionCouplingContextual"
    refinement_inputs = refinement["inputs"]
    assert isinstance(refinement_inputs, dict)
    assert "latent_context_size" in refinement_inputs
    assert "latent_tile_width" not in refinement_inputs


def test_replayed_multiselect_mask_values_are_literal_wrapped() -> None:
    """Keep executed multiselect lists from being reinterpreted as graph links."""

    template = _anima_template()
    template["mask-loader"] = {
        "class_type": "SimpleSyrup.LoadMaskBatch",
        "inputs": {"image": ["left.png", "right.png"], "channel": "red"},
    }
    case = ProofCase("fixture", "anima", "<lora:g:1>", "<lora:r:1>")

    graph, _save_ids = build_case_graph(template, case, run_id="run")

    assert graph["mask-loader"]["inputs"] == {
        "image": {"__value__": ["left.png", "right.png"]},
        "channel": "red",
    }


def _anima_template() -> dict[str, dict[str, object]]:
    """Return one minimal expanded Anima graph accepted by the mutator."""

    return {
        "anima-prompt-region:positive_prompt": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": "global[SEP|Taffy]left[SEP|Anise]right"},
        },
        "anima-diffusion-upscale:positive_prompt": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": "global[SEP|Taffy]left[SEP|Anise]right"},
        },
        "anima-prompt-region:ksampler": {
            "class_type": "SimpleSyrup.KSamplerAttentionCoupling",
            "inputs": {},
        },
        "anima-diffusion-upscale:ksampler": {
            "class_type": "SimpleSyrup.KSamplerAttentionCouplingTiled",
            "inputs": {
                "latent_tile_width": 128,
                "latent_tile_height": 128,
                "latent_tile_overlap": 16,
                "latent_tile_batch_size": 4,
            },
        },
        "anima-prompt-region:vae_decode": {
            "class_type": "VAEDecode",
            "inputs": {},
        },
        "anima-diffusion-upscale:vae_decode": {
            "class_type": "VAEDecode",
            "inputs": {},
        },
        "__sugarcubes_cube_output__:fixture": {
            "class_type": "SugarCubes.CubeOutput",
            "inputs": {},
        },
    }
