# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build API-format Comfy workflows for fixed regional benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass

from .manifest_types import (
    BenchmarkCase,
    BenchmarkManifest,
    BenchmarkRun,
    ExecutionProfile,
    JsonObject,
)

Workflow = dict[str, JsonObject]


@dataclass(frozen=True)
class BuiltWorkflow:
    """Return one prompt graph and the node IDs needed for result parsing."""

    prompt: Workflow
    metrics_node_id: str
    save_node_id: str


class BenchmarkWorkflowBuilder:
    """Translate validated benchmark records into current public Comfy nodes."""

    def build(
        self,
        manifest: BenchmarkManifest,
        run: BenchmarkRun,
        mask_names: tuple[str, ...],
    ) -> BuiltWorkflow:
        """Build one deterministic loader-to-save workflow."""

        case = self._case(manifest, run.case_id)
        execution = self._execution(manifest, run.execution_id)
        if len(mask_names) != len(case.masks):
            raise ValueError(
                "Materialized mask count does not match the benchmark case."
            )

        prompt: Workflow = {
            "1": self._node(
                "SimpleSyrup.SimpleLoadAnima",
                diffusion_model="Anima\\diffusion-model.safetensors",
                quantization="Original",
                diffusion_weight_dtype="default",
                text_encoder="qwen\\qwen_3_06b_base.safetensors",
                text_encoder_device="default",
                vae="qwen\\qwen_image_vae.safetensors",
            ),
            "2": self._node(
                "SimpleSyrup.EncodePromptBatch",
                clip=["1", 1],
                positive_prompt=self._positive_prompt(case),
                negative_prompt=manifest.sampling.negative_prompt,
                separator="[SEP]",
            ),
            "3": self._mask_node(mask_names, manifest),
            "4": self._node(
                "EmptyCosmosLatentVideo",
                width=manifest.sampling.width,
                height=manifest.sampling.height,
                length=1,
                batch_size=1,
            ),
            "5": self._node(
                "SimpleSyrupBenchmark.InstrumentModel",
                model=["1", 0],
                run_id=run.artifact_id,
            ),
            "6": self._sampler_node(manifest, case, run, execution),
            "7": self._node(
                "SimpleSyrupBenchmark.ReadMetrics",
                latent=["6", 0],
                run_id=run.artifact_id,
            ),
            "8": self._node("VAEDecode", samples=["7", 0], vae=["1", 2]),
            "9": self._node(
                "SaveImage",
                images=["8", 0],
                filename_prefix=(
                    f"simple_syrup_benchmark/{manifest.benchmark_id}/{run.artifact_id}"
                ),
            ),
        }
        return BuiltWorkflow(prompt=prompt, metrics_node_id="7", save_node_id="9")

    def _sampler_node(
        self,
        manifest: BenchmarkManifest,
        case: BenchmarkCase,
        run: BenchmarkRun,
        execution: ExecutionProfile,
    ) -> JsonObject:
        """Build the selected full, tiled, or Contextual regional sampler."""

        inputs: JsonObject = {
            "model": ["5", 0],
            "seed": run.seed,
            "steps": manifest.sampling.steps,
            "cfg": manifest.sampling.cfg,
            "sampler_name": manifest.sampling.sampler,
            "scheduler": manifest.sampling.scheduler,
            "positive": ["2", 0],
            "negative": ["2", 1],
            "region_masks": ["3", 0],
            "regional_prompt_weight": case.regional_prompt_weight,
            "region_mask_feather": case.region_mask_feather,
            "latent_image": ["4", 0],
            "denoise": manifest.sampling.denoise,
        }
        if execution.spatial_mode == "full":
            return {
                "class_type": "SimpleSyrup.KSamplerPromptByRegion",
                "inputs": inputs,
            }
        if execution.spatial_mode in {"multidiffusion", "mixture_of_diffusers"}:
            inputs.update(
                {
                    "diffusion_mode": execution.spatial_mode,
                    "latent_tile_width": execution.controls["tile_width"],
                    "latent_tile_height": execution.controls["tile_height"],
                    "latent_tile_overlap": execution.controls["overlap"],
                    "latent_tile_batch_size": execution.controls["tile_batch_size"],
                }
            )
            return {
                "class_type": "SimpleSyrup.KSamplerPromptByTiledRegion",
                "inputs": inputs,
            }
        inputs.update(execution.controls)
        return {
            "class_type": "SimpleSyrup.KSamplerContextualDiffusion",
            "inputs": inputs,
        }

    @staticmethod
    def _mask_node(
        mask_names: tuple[str, ...], manifest: BenchmarkManifest
    ) -> JsonObject:
        """Load authored masks or provide a zero technical mask for global-only."""

        if mask_names:
            return {
                "class_type": "SimpleSyrup.LoadMaskBatch",
                "inputs": {
                    "image": {"__value__": list(mask_names)},
                    "channel": "red",
                },
            }
        return {
            "class_type": "SolidMask",
            "inputs": {
                "value": 0.0,
                "width": manifest.sampling.width,
                "height": manifest.sampling.height,
            },
        }

    @staticmethod
    def _positive_prompt(case: BenchmarkCase) -> str:
        """Serialize the global-first prompt batch without changing prompt text."""

        return "[SEP]".join((case.global_prompt, *case.regional_prompts))

    @staticmethod
    def _case(manifest: BenchmarkManifest, case_id: str) -> BenchmarkCase:
        """Resolve one manifest case by stable ID."""

        return next(case for case in manifest.cases if case.case_id == case_id)

    @staticmethod
    def _execution(
        manifest: BenchmarkManifest,
        execution_id: str,
    ) -> ExecutionProfile:
        """Resolve one manifest execution profile by stable ID."""

        return next(
            execution
            for execution in manifest.executions
            if execution.execution_id == execution_id
        )

    @staticmethod
    def _node(class_type: str, **inputs: object) -> JsonObject:
        """Build one API-format node record."""

        return {"class_type": class_type, "inputs": inputs}
