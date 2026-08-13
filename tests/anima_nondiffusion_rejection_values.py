# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build exact synthetic P9.6 host rejection evidence for focused tests."""

from __future__ import annotations

from tools.anima_regional_lora_admission_integration.graph_contract import (
    PUBLIC_NODE_ID,
)
from tools.anima_regional_lora_admission_integration.history import (
    RegionalLoraAdmissionError,
)
from tools.anima_regional_lora_admission_integration.workflow import (
    BuiltRegionalLoraAdmissionWorkflow,
)
from tools.anima_regional_nondiffusion_rejection_integration.matrix import (
    AnimaNondiffusionRejectionCase,
)
from tools.comfy_api import JsonObject


def synthetic_rejection_message(case: AnimaNondiffusionRejectionCase) -> str:
    """Return one exact aggregate production-shaped exception message."""

    reason = case.expected_error_fragments[-1]
    if case.case_id == "turbo-mixed-diffusion-llm":
        targets = [
            "diffusion_model.llm_adapter.blocks.0.cross_attn.k_proj",
            *(
                f"diffusion_model.llm_adapter.blocks.{index // 10}.synthetic.{index}"
                for index in range(1, 59)
            ),
            "diffusion_model.llm_adapter.blocks.5.self_attn.v_proj",
        ]
        issue_lines = tuple(
            f"- adapter 0 'anima-turbo-v0.2' key '{target}': {reason}"
            for target in targets
        )
    elif case.case_id == "llm-adapter-only":
        issue_lines = (
            "- adapter 0 'llm-adapter-target' key "
            "'diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj': "
            f"{reason}",
        )
    else:
        index = case.expected_issue_adapter_index
        identity = case.fixture_adapter_identity
        assert identity is not None
        text_reason = case.expected_error_fragments[-3]
        issue_lines = (
            f"- adapter {index} {identity!r} key "
            "'lora_te_model.layers.0.self_attn.q_proj': "
            f"{text_reason}",
            f"- adapter {index} {identity!r} key 'vae.decoder.conv_in': {reason}",
        )
    return "Regional Anima LoRA plan admission failed before sampling:\n" + "\n".join(
        issue_lines
    )


def synthetic_rejection_history(
    case: AnimaNondiffusionRejectionCase,
    workflow: BuiltRegionalLoraAdmissionWorkflow,
) -> JsonObject:
    """Return one exact terminal Comfy execution-error history value."""

    return {
        "outputs": {},
        "status": {
            "status_str": "error",
            "messages": [
                [
                    "execution_error",
                    {
                        "node_id": workflow.sampler_node_id,
                        "node_type": PUBLIC_NODE_ID,
                        "exception_type": (
                            "simple_syrup.runtime.regional_lora."
                            "AnimaRegionalLoraPlanAdmissionError"
                        ),
                        "exception_message": synthetic_rejection_message(case),
                        "executed": ["1", "2"],
                    },
                ]
            ],
        },
    }


def synthetic_rejection(
    case: AnimaNondiffusionRejectionCase,
    workflow: BuiltRegionalLoraAdmissionWorkflow,
) -> RegionalLoraAdmissionError:
    """Return the already narrowed form of the synthetic host error."""

    return RegionalLoraAdmissionError(
        workflow.sampler_node_id,
        PUBLIC_NODE_ID,
        "simple_syrup.runtime.regional_lora.AnimaRegionalLoraPlanAdmissionError",
        synthetic_rejection_message(case),
        ("1", "2"),
    )
