# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Clone and patch supported MODEL/CLIP pairs for automatic NegPiP."""

# NegPiP behavior is adapted from ComfyUI-ppm and its credited predecessors.
# See third_party/manifest.toml and third_party/NOTICE.md.

from __future__ import annotations

import logging
from functools import partial

from comfy.model_base import SDXL, Anima, BaseModel, Krea2, SDXLRefiner
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP

from ..runtime.clip_patcher_mutations import (
    ClipBooleanOptionMutation,
    ClipCallableObjectPatchMutation,
    ClipTokenizerMutation,
)
from ..runtime.model_patcher_mutations import (
    ModelAttentionPatchMutation,
    ModelBooleanOptionMutation,
    ModelCallableObjectPatchMutation,
    ModelDiffusionWrapperMutation,
    ModelInteropDiffusionWrapperMutation,
)
from ..runtime.negpip.anima import (
    WRAPPER_KEY as ANIMA_WRAPPER_KEY,
)
from ..runtime.negpip.anima import (
    anima_attn2_negpip,
    anima_diffusion_negpip_wrapper,
    anima_extra_conds_negpip_wrapper,
)
from ..runtime.negpip.krea2 import (
    CLIP_MARKER,
    KREA_TOKEN_KEY,
    Krea2NegpipTokenizer,
    encode_krea2_token_weights_negpip,
    krea2_attn1_negpip,
    krea2_diffusion_negpip_wrapper,
    krea2_extra_conds_negpip_wrapper,
)
from ..runtime.negpip.krea2 import (
    WRAPPER_KEY as KREA_WRAPPER_KEY,
)
from ..runtime.negpip.standard import (
    encode_token_weights_negpip,
    standard_attn2_negpip,
)
from ..runtime.patcher_lifecycle import PATCHER_LIFECYCLE

LOGGER = logging.getLogger(__name__)
MODEL_MARKER = "ppm_negpip"
SUPPORTED_STANDARD_ENCODERS = ("clip_g", "clip_l", "t5xxl", "llama", "qwen3_06b")


class NegpipModelService:
    """Apply exactly one family-specific NegPiP patch set when supported."""

    def prepare(self, model: object, clip: object) -> tuple[object, object]:
        """Return a patched clone pair or the original unsupported pair unchanged."""

        if not isinstance(model, ModelPatcher) or not isinstance(clip, CLIP):
            raise TypeError("Automatic NegPiP requires Comfy MODEL and CLIP objects.")
        marker = model.model_options.get(MODEL_MARKER, False)
        if not isinstance(marker, bool):
            raise TypeError("MODEL ppm_negpip marker must be boolean.")
        if marker:
            LOGGER.debug("Automatic NegPiP reused an already-patched MODEL")
            return model, clip

        model_type = type(model.model)
        if model_type is Krea2:
            return self._prepare_krea2(model, clip)
        if model_type is Anima:
            return self._prepare_anima(model, clip)
        if model_type is BaseModel or issubclass(model_type, (SDXL, SDXLRefiner)):
            return self._prepare_standard(model, clip)
        LOGGER.debug(
            "Automatic NegPiP skipped unsupported model family",
            extra={"model_type": model_type.__qualname__},
        )
        return model, clip

    def _prepare_standard(
        self,
        model: ModelPatcher,
        clip: CLIP,
    ) -> tuple[ModelPatcher, CLIP]:
        """Install PPM-compatible interleaved key/value encoding on SD1 or SDXL."""

        encoders = [
            name
            for name in SUPPORTED_STANDARD_ENCODERS
            if hasattr(clip.patcher.model, name)
        ]
        if not encoders:
            LOGGER.warning("Automatic NegPiP found no supported standard text encoder")
            return model, clip
        prepared_clip = PATCHER_LIFECYCLE.derive_clip(
            clip,
            (
                *(
                    ClipCallableObjectPatchMutation(
                        f"{encoder_name}.encode_token_weights",
                        partial(
                            encode_token_weights_negpip,
                            getattr(clip.patcher.model, encoder_name),
                        ),
                    )
                    for encoder_name in encoders
                ),
                ClipBooleanOptionMutation(MODEL_MARKER, True),
            ),
            operation="automatic standard NegPiP CLIP preparation",
        )
        prepared_model = PATCHER_LIFECYCLE.derive_model(
            model,
            (
                ModelAttentionPatchMutation("attn2", standard_attn2_negpip),
                ModelBooleanOptionMutation(MODEL_MARKER, True),
            ),
            operation="automatic standard NegPiP MODEL preparation",
        )
        return prepared_model, prepared_clip

    def _prepare_anima(
        self,
        model: ModelPatcher,
        clip: CLIP,
    ) -> tuple[ModelPatcher, CLIP]:
        """Install PPM-compatible Anima weight-mask conditions and attention."""

        previous = model.get_model_object("extra_conds")
        prepared_model = PATCHER_LIFECYCLE.derive_model(
            model,
            (
                ModelCallableObjectPatchMutation(
                    "extra_conds",
                    anima_extra_conds_negpip_wrapper(previous),
                ),
                ModelInteropDiffusionWrapperMutation(
                    ANIMA_WRAPPER_KEY,
                    anima_diffusion_negpip_wrapper,
                ),
                ModelAttentionPatchMutation("attn2", anima_attn2_negpip),
                ModelBooleanOptionMutation(MODEL_MARKER, True),
            ),
            operation="automatic Anima NegPiP MODEL preparation",
        )
        prepared_clip = PATCHER_LIFECYCLE.derive_clip(
            clip,
            (ClipBooleanOptionMutation(MODEL_MARKER, True),),
            operation="automatic Anima NegPiP CLIP preparation",
        )
        return prepared_model, prepared_clip

    def _prepare_krea2(
        self,
        model: ModelPatcher,
        clip: CLIP,
    ) -> tuple[ModelPatcher, CLIP]:
        """Install Krea's shape-preserving sign-mask encoder and attention patch."""

        if not hasattr(clip.patcher.model, KREA_TOKEN_KEY):
            LOGGER.warning("Automatic NegPiP found no Krea Qwen3-VL text encoder")
            return model, clip
        outer_encoder = clip.patcher.get_model_object("encode_token_weights")
        prepared_clip = PATCHER_LIFECYCLE.derive_clip(
            clip,
            (
                ClipTokenizerMutation(
                    clip.tokenizer,
                    Krea2NegpipTokenizer(clip.tokenizer),
                ),
                ClipCallableObjectPatchMutation(
                    "encode_token_weights",
                    partial(encode_krea2_token_weights_negpip, outer_encoder),
                ),
                ClipBooleanOptionMutation(CLIP_MARKER, True),
            ),
            operation="automatic Krea 2 NegPiP CLIP preparation",
        )
        previous = model.get_model_object("extra_conds")
        prepared_model = PATCHER_LIFECYCLE.derive_model(
            model,
            (
                ModelCallableObjectPatchMutation(
                    "extra_conds",
                    krea2_extra_conds_negpip_wrapper(previous),
                ),
                ModelDiffusionWrapperMutation(
                    KREA_WRAPPER_KEY,
                    krea2_diffusion_negpip_wrapper,
                ),
                ModelAttentionPatchMutation("attn1", krea2_attn1_negpip),
                ModelBooleanOptionMutation(MODEL_MARKER, True),
            ),
            operation="automatic Krea 2 NegPiP MODEL preparation",
        )
        LOGGER.info(
            "Automatic NegPiP enabled",
            extra={"model_family": "krea2", "encoder": KREA_TOKEN_KEY},
        )
        return prepared_model, prepared_clip


NEGPIP_MODEL_SERVICE = NegpipModelService()
