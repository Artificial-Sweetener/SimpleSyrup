# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared tooltip text for ComfyUI node declarations."""

from __future__ import annotations

CHECKPOINT_MODEL_INPUT = (
    "Checkpoint file to load. This supplies the base MODEL, CLIP, and checkpoint "
    "VAE for the workflow."
)
CHECKPOINT_VAE_INPUT = (
    "VAE to output with the checkpoint. Use the checkpoint VAE to keep the model's "
    "own decoder, or choose another VAE to replace it."
)
CLIP_SKIP_INPUT = (
    "Use ComfyUI's clip-skip behavior for prompt encoding. Enable it for models "
    "that expect the next-to-last CLIP layer."
)

MODEL_OUTPUT = "Loaded diffusion model for downstream MODEL inputs."
CLIP_OUTPUT = "Loaded text encoder for downstream CLIP inputs."
VAE_OUTPUT = "Loaded VAE used to encode images to latents and decode latents to images."

VAE_OPTIONS_USE_TILING = (
    "Use ComfyUI's tiled VAE node. Disabled uses normal ComfyUI VAE behavior, "
    "including its automatic tiled retry after out-of-memory."
)
VAE_OPTIONS_ENCODE_PIXELS = "Image to encode into latent space."
VAE_OPTIONS_DECODE_SAMPLES = "Latent samples to decode into an image."
VAE_OPTIONS_VAE = "VAE used for the selected encode or decode operation."
VAE_OPTIONS_TILE_SIZE = (
    "Tile size in pixels. Larger tiles are faster but use more memory."
)
VAE_OPTIONS_OVERLAP = (
    "Overlap between tiles in pixels. Larger overlaps reduce seams but do more work."
)
VAE_OPTIONS_ENCODE_TEMPORAL_SIZE = "For video VAEs, number of frames to encode at once."
VAE_OPTIONS_DECODE_TEMPORAL_SIZE = "For video VAEs, number of frames to decode at once."
VAE_OPTIONS_TEMPORAL_OVERLAP = (
    "For video VAEs, number of overlapping frames between temporal tiles."
)
VAE_OPTIONS_LATENT_OUTPUT = "Latent produced by ComfyUI's selected VAE encode node."
VAE_OPTIONS_IMAGE_OUTPUT = "Image produced by ComfyUI's selected VAE decode node."

SAM_MODEL_OUTPUT = "Loaded SAM model for prompt-based mask and SEGS creation."
GROUNDING_DINO_MODEL_OUTPUT = (
    "Loaded GroundingDINO model for finding prompt-matched boxes in images."
)
VITMATTE_MODEL_OUTPUT = "Loaded ViTMatte model for refining mask edges."
WD14_TAGGER_OUTPUT = "Loaded WD14 tagger for generating prompt tags from image crops."

SAM_MODEL_INPUT = "SAM model choice used to create masks from detected boxes."
GROUNDING_DINO_MODEL_INPUT = (
    "GroundingDINO model choice used to find boxes that match a text prompt."
)
GROUNDING_DINO_TEXT_ENCODER_INPUT = (
    "BERT text encoder paired with GroundingDINO for prompt matching."
)
VITMATTE_MODEL_INPUT = "ViTMatte model choice used for mask edge refinement."
WD14_MODEL_INPUT = "WD14 tagger model choice used to generate tags from image crops."

SAMPLING_MODEL = "Diffusion model used to denoise the input latent."
SAMPLING_SEED = (
    "Seed used to create sampling noise. Reusing it with matching settings makes "
    "results repeatable."
)
SAMPLING_STEPS = (
    "Number of denoising steps. More steps can add refinement but take longer."
)
SAMPLING_CFG = (
    "Prompt guidance strength. Higher values follow the positive prompt more "
    "strongly but can look overcooked."
)
SAMPLER_NAME = "Sampling algorithm. It affects the image's look, speed, and stability."
SCHEDULER = (
    "Noise schedule used during sampling. It changes how quickly structure and "
    "detail form."
)
POSITIVE_CONDITIONING = "Positive conditioning that guides what the sampler should add."
NEGATIVE_CONDITIONING = (
    "Negative conditioning that guides what the sampler should avoid."
)
LATENT_IMAGE = "Latent input whose samples will be denoised."
DENOISE_STRENGTH = (
    "Sampling strength. Lower values preserve the input more; higher values allow "
    "larger changes."
)
DENOISED_LATENT_OUTPUT = "Denoised latent for VAE decode or more latent processing."

LATENT_TILE_WIDTH = (
    "Width of each latent tile. Larger tiles see more context but use more memory."
)
LATENT_TILE_HEIGHT = (
    "Height of each latent tile. Larger tiles see more context but use more memory."
)
LATENT_TILE_OVERLAP = (
    "Overlap between latent tiles. Larger overlaps reduce seams but increase "
    "sampling work."
)
LATENT_TILE_BATCH_SIZE = (
    "Number of latent tiles sampled together. Higher values can be faster but use "
    "more memory."
)

DETAIL_IMAGE = (
    "Source image containing the regions to improve. Detailed crops are blended "
    "back into this image."
)
DETAIL_SEGS = "SEGS regions that choose which parts of the image are detailed."
DETAIL_MODEL = "Diffusion model used to resample each detailed crop."
DETAIL_VAE = "VAE used to encode crops to latents and decode the edited crops."
DETAIL_POSITIVE = (
    "Positive conditioning for detailing. A conditioning batch is matched to SEGS "
    "order."
)
DETAIL_NEGATIVE = (
    "Negative conditioning for detailing. A conditioning batch is matched to SEGS "
    "order."
)
DETAIL_SCALE_FACTOR = (
    "Crop enlargement multiplier. Larger values give the sampler more detail room "
    "but use more memory."
)
SCALE_FACTOR_VALUE = (
    "Scaling multiplier. 1.0 keeps the target at its current size; larger values "
    "scale it up, with a maximum of 5.0x."
)
DETAIL_UPSCALE_METHOD = (
    "Resize method for scaled crops. Sharper methods preserve detail but can show "
    "more ringing."
)
DETAIL_CLAMP_SIZE = (
    "Maximum crop size in pixels after scaling. Use 0 to leave crop size unclamped."
)
DETAIL_FEATHER = (
    "Mask edge softness in pixels. Higher values blend edits more gently into the "
    "image."
)
DETAIL_NOISE_MASK = (
    "Limit sampling noise to the selected mask area so unchanged pixels stay more "
    "stable."
)
DETAIL_NOISE_MASK_FEATHER = (
    "Noise mask edge softness in pixels. Higher values make the sampled area fade "
    "out more gradually."
)
DETAIL_TILED_ENCODE = (
    "Encode large crops in tiles. This lowers memory use but is usually slower."
)
DETAIL_TILED_DECODE = (
    "Decode large crops in tiles. This lowers memory use but is usually slower."
)
DETAIL_IMAGE_OUTPUT = "Image with the detailed regions blended back into place."
SCALE_FACTOR_OUTPUT = "Multiplier used to scale a connected target."

REGIONAL_GLOBAL_NEGATIVE = (
    "Negative conditioning applied across the full regional pass."
)
REGIONAL_GLOBAL_POSITIVE = (
    "Positive conditioning that gives full-image context to the regional pass."
)
REGIONAL_POSITIVE_BATCH = (
    "Per-region positive conditioning matched to the incoming SEGS order."
)

TILE_IMAGE = "Image to split into tile SEGS for tagging or downstream workflows."
TILE_CLIP = "CLIP model used to encode each generated tile prompt."
TILE_WD14_TAGGER = "WD14 tagger that reads each tile crop and suggests prompt tags."
TILE_UNIVERSAL_POSITIVE = (
    "Positive prompt text added before every generated tile tag prompt."
)
TILE_BBOX_SIZE = "Target tile box size in pixels."
TILE_CROP_FACTOR = (
    "Tile crop expansion. Larger values include more surrounding context for tags."
)
TILE_MIN_OVERLAP = (
    "Minimum pixel overlap between tile regions. Higher values reduce gaps but "
    "repeat more image area."
)
TILE_FILTER_SEGS_DILATION = (
    "Grow or shrink tile masks before filtering. Positive values expand masks; "
    "negative values contract them."
)
TILE_MASK_IRREGULARITY = (
    "Organic variation added to tile masks. Higher values make masks less rectangular."
)
TILE_IRREGULAR_MASK_MODE = (
    "How irregular masks are generated. Reuse is steadier; random varies each "
    "tile; quality modes do more work."
)
TILE_THRESHOLD = (
    "Minimum WD14 confidence for general tags. Higher values keep fewer, more "
    "certain tags."
)
TILE_CHARACTER_THRESHOLD = (
    "Minimum WD14 confidence for character tags. Higher values keep fewer, more "
    "certain character tags."
)
TILE_REPLACE_UNDERSCORE = (
    "Replace underscores with spaces so generated tags read more naturally."
)
TILE_TRAILING_COMMA = (
    "Add a comma after generated tag text for easier prompt composition."
)
TILE_EXCLUDE_TAGS = "Comma-separated tags removed from generated tile prompts."
TILE_SEGS_OUTPUT = "Generated tile SEGS in the same order as the conditioning batch."
TILE_POSITIVE_OUTPUT = (
    "Positive conditioning from WD14 tile tags, matched to SEGS order."
)
