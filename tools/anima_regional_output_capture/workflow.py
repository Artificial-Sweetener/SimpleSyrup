# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build fixed API workflows for labeled regional Anima output evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass

from tools.comfy_api import JsonObject
from tools.comfy_integration.anima_fixture_selections import (
    ANIMA_DIFFUSION_SELECTION,
    ANIMA_TEXT_ENCODER_SELECTION,
    ANIMA_VAE_SELECTION,
)

from .matrix import (
    CFG,
    HEIGHT,
    NEGATIVE_PROMPT,
    SAMPLER,
    SCHEDULER,
    SEED,
    STEPS,
    WIDTH,
    VisualOutputProfile,
    VisualProfileKind,
)


@dataclass(frozen=True, slots=True)
class BuiltVisualOutputWorkflow:
    """Return one graph and its sole saved-image node identity."""

    prompt: dict[str, JsonObject]
    save_node_id: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every exact node contract required by this graph."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


class VisualOutputWorkflowBuilder:
    """Translate one immutable profile into a loader-to-image graph."""

    def build(
        self,
        profile: VisualOutputProfile,
        *,
        run_id: str,
        mask_names: tuple[str, ...],
    ) -> BuiltVisualOutputWorkflow:
        """Build one deterministic reference or optimized regional workflow."""

        graph = _Graph()
        loader = graph.add(
            "SimpleSyrup.SimpleLoadAnima",
            diffusion_model=ANIMA_DIFFUSION_SELECTION,
            quantization="Original",
            diffusion_weight_dtype="default",
            text_encoder=ANIMA_TEXT_ENCODER_SELECTION,
            text_encoder_device="default",
            vae=ANIMA_VAE_SELECTION,
        )
        model: object = [loader, 0]
        for adapter in profile.global_adapters:
            applied = graph.add(
                "LoraLoaderModelOnly",
                model=model,
                lora_name=adapter.lora_name,
                strength_model=adapter.strength,
            )
            model = [applied, 0]
        latent = graph.add(
            "EmptyCosmosLatentVideo",
            width=WIDTH,
            height=HEIGHT,
            length=1,
            batch_size=1,
        )
        if profile.kind is VisualProfileKind.GLOBAL_REFERENCE:
            positive = graph.add(
                "CLIPTextEncode", clip=[loader, 1], text=profile.global_prompt
            )
            negative = graph.add(
                "CLIPTextEncode", clip=[loader, 1], text=NEGATIVE_PROMPT
            )
            sampler_model = model
            sampler_positive: object = [positive, 0]
            sampler_negative: object = [negative, 0]
        else:
            encoded = graph.add(
                "SimpleSyrup.EncodePromptBatch",
                clip=[loader, 1],
                positive_prompt="[SEP]".join(
                    (profile.global_prompt, *profile.regional_prompts)
                ),
                negative_prompt="[SEP]".join(
                    (NEGATIVE_PROMPT,) * (len(profile.regional_prompts) + 1)
                ),
                separator="[SEP]",
            )
            masks = self._mask_node(graph, profile, mask_names)
            patched = graph.add(
                "SimpleSyrupBenchmark.StaticAnimaRegionalProfile",
                model=model,
                positive=[encoded, 0],
                negative=[encoded, 1],
                region_masks=[masks, 0],
                latent=[latent, 0],
                regional_adapters_json=self._regional_json(profile),
            )
            sampler_model = [patched, 0]
            sampler_positive = [patched, 1]
            sampler_negative = [patched, 2]
        sampled = graph.add(
            "KSampler",
            model=sampler_model,
            seed=SEED,
            steps=STEPS,
            cfg=CFG,
            sampler_name=SAMPLER,
            scheduler=SCHEDULER,
            positive=sampler_positive,
            negative=sampler_negative,
            latent_image=[latent, 0],
            denoise=1.0,
        )
        decoded = graph.add("VAEDecode", samples=[sampled, 0], vae=[loader, 2])
        saved = graph.add(
            "SaveImage",
            images=[decoded, 0],
            filename_prefix=f"simple_syrup_benchmark/{run_id}/{profile.profile_id}",
        )
        return BuiltVisualOutputWorkflow(graph.prompt, saved)

    @staticmethod
    def _mask_node(
        graph: _Graph,
        profile: VisualOutputProfile,
        mask_names: tuple[str, ...],
    ) -> str:
        """Build an all-one control or load exact ordered split masks."""

        if profile.mask_case_id == "all-one":
            if mask_names:
                raise ValueError("All-one visual profile must not supply mask files.")
            return graph.add("SolidMask", value=1.0, width=WIDTH, height=HEIGHT)
        if len(mask_names) != len(profile.regional_prompts):
            raise ValueError("Visual profile mask count must match regional prompts.")
        return graph.add(
            "SimpleSyrup.LoadMaskBatch",
            image={"__value__": list(mask_names)},
            channel="red",
        )

    @staticmethod
    def _regional_json(profile: VisualOutputProfile) -> str:
        """Serialize exact adapter order and region ownership for the probe."""

        return json.dumps(
            [
                {
                    "region_index": adapter.region_index,
                    "branch": adapter.branch.value,
                    "lora_name": adapter.lora_name,
                    "strength": adapter.strength,
                }
                for adapter in profile.regional_adapters
            ],
            separators=(",", ":"),
        )


class _Graph:
    """Own deterministic numeric node identities for one visual graph."""

    def __init__(self) -> None:
        """Initialize an empty graph."""

        self.prompt: dict[str, JsonObject] = {}

    def add(self, class_type: str, **inputs: object) -> str:
        """Append one node and return its stable numeric identity."""

        node_id = str(len(self.prompt) + 1)
        self.prompt[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id
