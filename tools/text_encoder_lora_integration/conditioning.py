"""Build native global and Prompt-Control regional CLIP-LoRA graphs."""

from __future__ import annotations

from tools.anima_attention_coupling_conditioning import BuiltAnimaConditioning
from tools.anima_attention_coupling_prompts import (
    GLOBAL_PROMPT,
    NEGATIVE_PROMPT,
    PINNED_ADAPTER_A,
    REGIONAL_PROMPTS,
)
from tools.anima_workflow_graph import AnimaWorkflowGraph, NodeReference

from .fixture import PINNED_TEXT_ENCODER_LORA_FIXTURE
from .matrix import TextEncoderLoraCase

TEXT_ENCODER_STRENGTH = 0.75
ADAPTER_A_MODEL_STRENGTH = 0.8


class TextEncoderLoraConditioningWorkflow:
    """Encode one P9.4 case through public Comfy and Prompt Control nodes."""

    def __init__(self, case: TextEncoderLoraCase) -> None:
        """Retain one immutable matrix case."""

        if not isinstance(case, TextEncoderLoraCase):
            raise TypeError("P9.4 conditioning requires a matrix case.")
        self._case = case

    def add(
        self,
        graph: AnimaWorkflowGraph,
        *,
        model: NodeReference,
        clip: NodeReference,
    ) -> BuiltAnimaConditioning:
        """Return model and already-encoded regional conditioning links."""

        if self._case.global_text_lora:
            applied = graph.add(
                "LoraLoader",
                model=model,
                clip=clip,
                lora_name=PINNED_TEXT_ENCODER_LORA_FIXTURE.lora_name,
                strength_model=0.0,
                strength_clip=TEXT_ENCODER_STRENGTH,
            )
            model = [applied, 0]
            clip = [applied, 1]
        encoded = graph.add(
            "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
            model=model,
            clip=clip,
            positive_prompt=self._positive_prompt(),
            negative_prompt=NEGATIVE_PROMPT,
        )
        return BuiltAnimaConditioning([encoded, 0], [encoded, 1], [encoded, 2])

    def _positive_prompt(self) -> str:
        """Render explicit model and CLIP strengths for the left region."""

        left_tags: list[str] = []
        if self._case.regional_text_lora:
            left_tags.append(
                f"<lora:{PINNED_TEXT_ENCODER_LORA_FIXTURE.lora_name}:"
                f"0:{TEXT_ENCODER_STRENGTH:g}>"
            )
        if self._case.regional_model_lora:
            left_tags.append(f"<lora:{PINNED_ADAPTER_A}:{ADAPTER_A_MODEL_STRENGTH:g}:0>")
        left = " ".join((REGIONAL_PROMPTS[0], *left_tags))
        return "[SEP]".join((GLOBAL_PROMPT, left, REGIONAL_PROMPTS[1]))
