"""Build the public full, tiled, and Contextual SDXL workflow graph."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject

from .matrix import (
    CFG,
    MODES,
    NEGATIVE_PROMPTS,
    POSITIVE_PROMPTS,
    REFINEMENT_DENOISE,
    REFINEMENT_STEPS,
    SAMPLER,
    SCHEDULER,
    SEED,
    SOURCE_HEIGHT,
    SOURCE_STEPS,
    SOURCE_WIDTH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    TILE_BATCH_SIZE,
    TILE_OVERLAP,
    TILE_SIZE,
)


@dataclass(frozen=True, slots=True)
class SdxlWorkflowOutputs:
    """Retain output-node identities for one labeled sampler mode."""

    save_node_id: str
    metrics_node_id: str
    diagnostics_node_id: str


@dataclass(frozen=True, slots=True)
class SdxlSamplerBranchResult:
    """Retain one sampled latent and its already-created decoded image."""

    latent: list[str | int]
    decoded: list[str | int]


@dataclass(frozen=True, slots=True)
class BuiltSdxlAttentionCouplingWorkflow:
    """Return one API graph and exact labeled output-node identities."""

    prompt: dict[str, JsonObject]
    outputs: dict[str, SdxlWorkflowOutputs]

    @property
    def required_node_ids(self) -> frozenset[str]:
        """Return every class required from clean managed-node discovery."""

        return frozenset(str(node["class_type"]) for node in self.prompt.values())


def build_sdxl_attention_coupling_workflow(
    *,
    run_id: str,
    checkpoint_name: str,
    mask_names: tuple[str, str],
) -> BuiltSdxlAttentionCouplingWorkflow:
    """Build one source generation and two 1.5x refinement branches."""

    graph = _Graph()
    loader = graph.add("CheckpointLoaderSimple", ckpt_name=checkpoint_name)
    positive = _conditioning_batch(
        graph,
        clip=[loader, 1],
        prompts=POSITIVE_PROMPTS,
    )
    negative = _conditioning_batch(
        graph,
        clip=[loader, 1],
        prompts=NEGATIVE_PROMPTS,
    )
    masks = graph.add(
        "SimpleSyrup.LoadMaskBatch",
        image={"__value__": list(mask_names)},
        channel="red",
    )
    source_latent = graph.add(
        "EmptyLatentImage",
        width=SOURCE_WIDTH,
        height=SOURCE_HEIGHT,
        batch_size=1,
    )
    outputs: dict[str, SdxlWorkflowOutputs] = {}
    full_branch = _add_sampler_branch(
        graph,
        mode_id=MODES[0].mode_id,
        node_id=MODES[0].node_id,
        model=[loader, 0],
        positive=positive,
        negative=negative,
        masks=[masks, 0],
        latent=[source_latent, 0],
        vae=[loader, 2],
        run_id=run_id,
        steps=SOURCE_STEPS,
        denoise=1.0,
        sampler_inputs={},
        outputs=outputs,
    )
    upscaled_image = graph.add(
        "ImageScale",
        image=full_branch.decoded,
        upscale_method="lanczos",
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
        crop="disabled",
    )
    upscaled_latent = graph.add(
        "VAEEncode",
        pixels=[upscaled_image, 0],
        vae=[loader, 2],
    )
    _add_sampler_branch(
        graph,
        mode_id=MODES[1].mode_id,
        node_id=MODES[1].node_id,
        model=[loader, 0],
        positive=positive,
        negative=negative,
        masks=[masks, 0],
        latent=[upscaled_latent, 0],
        vae=[loader, 2],
        run_id=run_id,
        steps=REFINEMENT_STEPS,
        denoise=REFINEMENT_DENOISE,
        sampler_inputs={
            "diffusion_mode": "multidiffusion",
            "latent_tile_width": TILE_SIZE,
            "latent_tile_height": TILE_SIZE,
            "latent_tile_overlap": TILE_OVERLAP,
            "latent_tile_batch_size": TILE_BATCH_SIZE,
        },
        outputs=outputs,
    )
    _add_sampler_branch(
        graph,
        mode_id=MODES[2].mode_id,
        node_id=MODES[2].node_id,
        model=[loader, 0],
        positive=positive,
        negative=negative,
        masks=[masks, 0],
        latent=[upscaled_latent, 0],
        vae=[loader, 2],
        run_id=run_id,
        steps=REFINEMENT_STEPS,
        denoise=REFINEMENT_DENOISE,
        sampler_inputs={
            "diffusion_mode": "multidiffusion",
            "latent_context_size": TILE_SIZE,
            "latent_context_overlap": TILE_OVERLAP,
            "latent_context_batch_size": TILE_BATCH_SIZE,
            "global_weight": 1.0,
            "global_steps": 1,
            "global_decay": 0.5,
        },
        outputs=outputs,
    )
    return BuiltSdxlAttentionCouplingWorkflow(graph.prompt, outputs)


def _conditioning_batch(
    graph: _Graph,
    *,
    clip: list[str | int],
    prompts: tuple[str, ...],
) -> list[str | int]:
    """Encode and pack one global-first conditioning batch."""

    encoded = tuple(
        graph.add("CLIPTextEncode", clip=clip, text=prompt) for prompt in prompts
    )
    batch = graph.add(
        "SimpleSyrup.ConditioningBatchStart",
        conditioning=[encoded[0], 0],
    )
    for node_id in encoded[1:]:
        batch = graph.add(
            "SimpleSyrup.ConditioningBatchAppend",
            batch=[batch, 0],
            conditioning=[node_id, 0],
        )
    return [batch, 0]


def _add_sampler_branch(
    graph: _Graph,
    *,
    mode_id: str,
    node_id: str,
    model: list[str | int],
    positive: list[str | int],
    negative: list[str | int],
    masks: list[str | int],
    latent: list[str | int],
    vae: list[str | int],
    run_id: str,
    steps: int,
    denoise: float,
    sampler_inputs: dict[str, object],
    outputs: dict[str, SdxlWorkflowOutputs],
) -> SdxlSamplerBranchResult:
    """Add one instrumented public sampler, diagnostics, decode, and save chain."""

    metrics_run_id = f"{run_id}:{mode_id}:metrics"
    diagnostics_run_id = f"{run_id}:{mode_id}:diagnostics"
    instrumented = graph.add(
        "SimpleSyrupBenchmark.InstrumentModel",
        model=model,
        run_id=metrics_run_id,
    )
    captured = graph.add(
        "SimpleSyrupBenchmark.CaptureRegionalDiagnostics",
        model=[instrumented, 0],
        run_id=diagnostics_run_id,
    )
    sampled = graph.add(
        node_id,
        model=[captured, 0],
        seed=SEED,
        steps=steps,
        cfg=CFG,
        sampler_name=SAMPLER,
        scheduler=SCHEDULER,
        positive=positive,
        negative=negative,
        region_masks=masks,
        regional_prompt_weight=1.0,
        region_mask_feather=32,
        latent_image=latent,
        denoise=denoise,
        **sampler_inputs,
    )
    metrics = graph.add(
        "SimpleSyrupBenchmark.ReadMetrics",
        latent=[sampled, 0],
        run_id=metrics_run_id,
    )
    diagnostics = graph.add(
        "SimpleSyrupBenchmark.ReadRegionalDiagnostics",
        latent=[metrics, 0],
        run_id=diagnostics_run_id,
    )
    decoded = graph.add("VAEDecode", samples=[diagnostics, 0], vae=vae)
    saved = graph.add(
        "SaveImage",
        images=[decoded, 0],
        filename_prefix=f"simple_syrup_p8_5_sdxl/{run_id}/{mode_id}",
    )
    outputs[mode_id] = SdxlWorkflowOutputs(saved, metrics, diagnostics)
    return SdxlSamplerBranchResult([diagnostics, 0], [decoded, 0])


class _Graph:
    """Own deterministic numeric API node identities."""

    def __init__(self) -> None:
        """Initialize one empty prompt graph."""

        self.prompt: dict[str, JsonObject] = {}

    def add(self, class_type: str, **inputs: object) -> str:
        """Append one node and return its stable numeric identity."""

        node_id = str(len(self.prompt) + 1)
        self.prompt[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id
