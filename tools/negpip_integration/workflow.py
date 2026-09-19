# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build focused loader-to-sampler automatic NegPiP workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tools.anima_workflow_graph import AnimaWorkflowGraph
from tools.comfy_api import JsonObject

CONTROL_PROMPT = (
    "studio portrait of a person wearing a bright red jacket, plain gray background"
)
NEGATIVE_WEIGHT_LABEL = "bright (red:-1.0) jacket"
NEGATIVE_WEIGHT_PROMPT = (
    f"studio portrait of a person wearing a {NEGATIVE_WEIGHT_LABEL}, "
    "plain gray background"
)
REFINER_BASE_PROMPT = (
    "studio portrait of a person wearing a jacket, plain gray background"
)
REFINER_SWITCH_STEP = 16


class NegpipLiveFamily(StrEnum):
    """Identify every materially distinct supported live model path."""

    SD1 = "sd1"
    SDXL = "sdxl"
    SDXL_REFINER = "sdxl_refiner"
    ANIMA = "anima"
    KREA2 = "krea2"


@dataclass(frozen=True, slots=True)
class NegpipFixtureSelections:
    """Name exact model selections exposed to the isolated Comfy server."""

    sd1_checkpoint: str
    sdxl_checkpoint: str
    sdxl_refiner_checkpoint: str
    anima_diffusion: str
    anima_text_encoder: str
    krea2_diffusion: str
    krea2_text_encoder: str
    qwen_image_vae: str


@dataclass(frozen=True, slots=True)
class BuiltNegpipLiveWorkflow:
    """Retain one graph and its exact evidence node identities."""

    prompt: dict[str, JsonObject]
    family: NegpipLiveFamily
    run_id: str
    runtime_node_id: str | None
    modifier_node_id: str
    conditioning_node_id: str | None
    image_node_id: str
    mode: str

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every Comfy node contract named by this workflow."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


class NegpipLiveWorkflowBuilder:
    """Create triggered sampling and untriggered gate-control workflows."""

    def __init__(self, selections: NegpipFixtureSelections) -> None:
        """Retain exact managed model aliases."""

        self._selections = selections

    def build(
        self,
        family: NegpipLiveFamily,
        *,
        run_id: str,
        trigger: bool,
        baseline_ppm: bool = False,
    ) -> BuiltNegpipLiveWorkflow:
        """Build one public Schedule & Encode graph with live evidence."""

        if baseline_ppm and not trigger:
            raise ValueError("The PPM baseline workflow requires a negative weight.")

        graph = AnimaWorkflowGraph()
        model, clip, vae = self._loader(graph, family)
        if baseline_ppm:
            baseline = graph.add("CLIPNegPip", model=model, clip=clip)
            model = [baseline, 0]
            clip = [baseline, 1]
        prompt = NEGATIVE_WEIGHT_PROMPT if trigger else CONTROL_PROMPT
        scheduled = graph.add(
            "SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl",
            model=model,
            clip=clip,
            positive_prompt=prompt,
            negative_prompt="blurry, low quality",
        )
        modifier = graph.add(
            "SimpleSyrupBenchmark.SnapshotModelModifier",
            model=[scheduled, 0],
            run_id=f"{run_id}:modifier",
        )
        conditioning: str | None = None
        sampling_model: list[str | int] = [modifier, 0]
        if trigger:
            conditioning = graph.add(
                "SimpleSyrupBenchmark.SnapshotConditioningBatch",
                positive=[scheduled, 1],
                negative=[scheduled, 2],
                run_id=f"{run_id}:conditioning",
            )
            instrumented = graph.add(
                "SimpleSyrupBenchmark.InstrumentNegpipModel",
                model=[modifier, 0],
                run_id=run_id,
            )
            sampling_model = [instrumented, 0]
        if family is NegpipLiveFamily.SDXL_REFINER:
            latent, vae = self._sdxl_base_stage(graph)
        else:
            latent = self._latent(graph, family)
        steps, cfg, sampler_name, scheduler = self._sampling(family)
        if family is NegpipLiveFamily.SDXL_REFINER:
            sampled = graph.add(
                "KSamplerAdvanced",
                model=sampling_model,
                add_noise="disable",
                noise_seed=4_205_191,
                steps=24,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=[scheduled, 1],
                negative=[scheduled, 2],
                latent_image=latent,
                start_at_step=REFINER_SWITCH_STEP,
                end_at_step=24,
                return_with_leftover_noise="disable",
            )
        else:
            sampled = graph.add(
                "KSampler",
                model=sampling_model,
                seed=4_205_191,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=[scheduled, 1],
                negative=[scheduled, 2],
                latent_image=latent,
                denoise=1.0,
            )
        runtime: str | None = None
        decoded_latent: list[str | int] = [sampled, 0]
        if trigger:
            runtime = graph.add(
                "SimpleSyrupBenchmark.ReadNegpipRuntime",
                latent=[sampled, 0],
                run_id=run_id,
            )
            decoded_latent = [runtime, 0]
        decoded = graph.add("VAEDecode", samples=decoded_latent, vae=vae)
        saved = graph.add(
            "SaveImage",
            images=[decoded, 0],
            filename_prefix=(
                "simple_syrup_negpip_proof/"
                + run_id.replace(":", "-").replace("\\", "-")
            ),
        )
        return BuiltNegpipLiveWorkflow(
            graph.prompt,
            family,
            run_id,
            runtime,
            modifier,
            conditioning,
            saved,
            ("ppm_baseline" if baseline_ppm else "negative" if trigger else "control"),
        )

    def _loader(
        self,
        graph: AnimaWorkflowGraph,
        family: NegpipLiveFamily,
    ) -> tuple[list[str | int], list[str | int], list[str | int]]:
        """Add the exact family loader and return MODEL/CLIP/VAE references."""

        if family in {
            NegpipLiveFamily.SD1,
            NegpipLiveFamily.SDXL,
            NegpipLiveFamily.SDXL_REFINER,
        }:
            selections = {
                NegpipLiveFamily.SD1: self._selections.sd1_checkpoint,
                NegpipLiveFamily.SDXL: self._selections.sdxl_checkpoint,
                NegpipLiveFamily.SDXL_REFINER: (
                    self._selections.sdxl_refiner_checkpoint
                ),
            }
            selection = selections[family]
            loader = graph.add("CheckpointLoaderSimple", ckpt_name=selection)
            return [loader, 0], [loader, 1], [loader, 2]
        if family is NegpipLiveFamily.ANIMA:
            loader = graph.add(
                "SimpleSyrup.SimpleLoadAnima",
                diffusion_model=self._selections.anima_diffusion,
                quantization="Original",
                diffusion_weight_dtype="default",
                text_encoder=self._selections.anima_text_encoder,
                text_encoder_device="default",
                vae=self._selections.qwen_image_vae,
            )
            return [loader, 0], [loader, 1], [loader, 2]
        loader = graph.add(
            "UNETLoader",
            unet_name=self._selections.krea2_diffusion,
            weight_dtype="default",
        )
        clip_loader = graph.add(
            "CLIPLoader",
            clip_name=self._selections.krea2_text_encoder,
            type="krea2",
            device="default",
        )
        vae_loader = graph.add("VAELoader", vae_name=self._selections.qwen_image_vae)
        return [loader, 0], [clip_loader, 0], [vae_loader, 0]

    @staticmethod
    def _latent(
        graph: AnimaWorkflowGraph,
        family: NegpipLiveFamily,
    ) -> list[str | int]:
        """Add the family's native smallest practical image latent."""

        if family is NegpipLiveFamily.ANIMA:
            node = graph.add(
                "EmptyCosmosLatentVideo",
                width=512,
                height=512,
                length=1,
                batch_size=1,
            )
        elif family is NegpipLiveFamily.KREA2:
            node = graph.add(
                "EmptySD3LatentImage",
                width=512,
                height=512,
                batch_size=1,
            )
        else:
            node = graph.add(
                "EmptyLatentImage",
                width=512,
                height=512,
                batch_size=1,
            )
        return [node, 0]

    def _sdxl_base_stage(
        self,
        graph: AnimaWorkflowGraph,
    ) -> tuple[list[str | int], list[str | int]]:
        """Generate a valid high-noise SDXL latent for the refiner proof stage."""

        base = graph.add(
            "CheckpointLoaderSimple",
            ckpt_name=self._selections.sdxl_checkpoint,
        )
        positive = graph.add(
            "CLIPTextEncode",
            clip=[base, 1],
            text=REFINER_BASE_PROMPT,
        )
        negative = graph.add(
            "CLIPTextEncode",
            clip=[base, 1],
            text="blurry, low quality",
        )
        latent = graph.add(
            "EmptyLatentImage",
            width=512,
            height=512,
            batch_size=1,
        )
        sampled = graph.add(
            "KSamplerAdvanced",
            model=[base, 0],
            add_noise="enable",
            noise_seed=4_205_191,
            steps=24,
            cfg=5.0,
            sampler_name="euler",
            scheduler="normal",
            positive=[positive, 0],
            negative=[negative, 0],
            latent_image=[latent, 0],
            start_at_step=0,
            end_at_step=REFINER_SWITCH_STEP,
            return_with_leftover_noise="enable",
        )
        return [sampled, 0], [base, 2]

    @staticmethod
    def _sampling(
        family: NegpipLiveFamily,
    ) -> tuple[int, float, str, str]:
        """Return practical multi-step settings for visible family output."""

        if family is NegpipLiveFamily.ANIMA:
            return 16, 1.0, "er_sde", "simple"
        if family is NegpipLiveFamily.KREA2:
            return 16, 1.0, "euler", "simple"
        if family is NegpipLiveFamily.SD1:
            return 20, 7.0, "euler", "normal"
        return 20, 5.0, "euler", "normal"
