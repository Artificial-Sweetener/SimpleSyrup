# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt NegPiP value masking to Krea 2's layered Qwen conditioning."""

# NegPiP behavior is adapted from ComfyUI-ppm and its credited predecessors.
# See third_party/manifest.toml and third_party/NOTICE.md.

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from comfy import conds

CLIP_MARKER = "simple_syrup_negpip"
WRAPPER_KEY = "simple_syrup.negpip.krea2"
ENCODER_MASK_KEY = "simple_syrup_negpip_mask"
CONDITION_MASK_KEY = "c_simple_syrup_negpip_mask"
TRANSFORMER_MASK_KEY = "simple_syrup_negpip_mask"
KREA_TOKEN_KEY = "qwen3vl_4b"
IM_START_TOKEN = 151644
USER_TOKEN = 872
NEWLINE_TOKEN = 198
IMAGE_PAD_TOKEN = 151655


class Krea2NegpipTokenizer:
    """Preserve Krea templates while enabling Comfy prompt-weight tokenization."""

    def __init__(self, source: object) -> None:
        """Retain one cloned CLIP's shared source tokenizer without mutating it."""

        self._source = source

    def __getattr__(self, name: str) -> object:
        """Delegate tokenizer metadata and helpers to the installed Krea tokenizer."""

        return getattr(self._source, name)

    def tokenize_with_weights(
        self,
        text: str,
        return_word_ids: bool = False,
        llama_template: str | None = None,
        images: Sequence[torch.Tensor] = (),
        prevent_empty_text: bool = False,
        thinking: bool = True,
        **kwargs: object,
    ) -> dict[str, list[list[tuple[object, ...]]]]:
        """Tokenize the normal Krea template while retaining parsed scalar weights."""

        image = kwargs.pop("image", None)
        if image is not None and not images:
            if not isinstance(image, torch.Tensor):
                raise TypeError("Krea tokenizer image input must be a tensor.")
            images = tuple(image[index : index + 1] for index in range(image.shape[0]))
        skip_template = bool(kwargs.pop("skip_template", False)) or text.startswith(
            "<|im_start|>"
        )
        kwargs.pop("disable_weights", None)
        if prevent_empty_text and text == "":
            text = " "

        if skip_template:
            prepared_text = text
        else:
            template = llama_template
            if template is None:
                template_name = (
                    "llama_template" if not images else "llama_template_images"
                )
                template = getattr(self._source, template_name)
            if not isinstance(template, str):
                raise TypeError("Krea tokenizer template must be text.")
            if len(images) > 1:
                vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
                template = template.replace(
                    vision_block,
                    vision_block * len(images),
                    1,
                )
            prepared_text = template.format(text)
            if not thinking:
                prepared_text += "<think>\n\n</think>\n\n"

        inner = getattr(self._source, KREA_TOKEN_KEY)
        tokens = inner.tokenize_with_weights(
            prepared_text,
            return_word_ids=return_word_ids,
            disable_weights=False,
            **kwargs,
        )
        embedded_count = 0
        for section in tokens:
            for index, pair in enumerate(section):
                token = pair[0]
                if (
                    isinstance(token, (int, float))
                    and token == IMAGE_PAD_TOKEN
                    and embedded_count < len(images)
                ):
                    section[index] = (
                        {
                            "type": "image",
                            "data": images[embedded_count],
                            "original_type": "image",
                        },
                        *pair[1:],
                    )
                    embedded_count += 1
        return {KREA_TOKEN_KEY: tokens}


def encode_krea2_token_weights_negpip(
    original: Callable[..., tuple[object, ...]],
    token_weight_pairs: dict[str, list[list[tuple[object, ...]]]],
    template_end: int = -1,
) -> tuple[object, ...]:
    """Encode absolute Krea magnitudes and publish a post-template sign mask."""

    sections = token_weight_pairs.get(KREA_TOKEN_KEY)
    if not isinstance(sections, list) or len(sections) != 1:
        raise ValueError("Krea NegPiP requires exactly one Qwen token section.")
    source_section = sections[0]
    absolute_section = [
        (pair[0], abs(_token_weight(pair)), *pair[2:]) for pair in source_section
    ]
    absolute_tokens = dict(token_weight_pairs)
    absolute_tokens[KREA_TOKEN_KEY] = [absolute_section]
    encoded = original(absolute_tokens, template_end=template_end)
    if len(encoded) < 3 or not isinstance(encoded[0], torch.Tensor):
        raise TypeError("Krea NegPiP encoder must return tensor conditioning metadata.")
    extra = encoded[2]
    if not isinstance(extra, dict):
        raise TypeError("Krea NegPiP encoder metadata must be a dictionary.")
    cut = _template_end(source_section) if template_end == -1 else template_end
    signs = [
        -1.0 if _token_weight(pair) < 0.0 else 1.0 for pair in source_section[cut:]
    ]
    sequence_length = int(encoded[0].shape[1])
    if len(signs) != sequence_length:
        raise ValueError(
            "Krea NegPiP sign mask does not match post-template conditioning: "
            f"{len(signs)} signs for {sequence_length} tokens."
        )
    prepared_extra = dict(extra)
    prepared_extra[ENCODER_MASK_KEY] = torch.tensor(signs).reshape(1, -1, 1)
    return encoded[0], encoded[1], prepared_extra


def krea2_extra_conds_negpip_wrapper(
    previous_extra_conds: Callable[..., dict[str, object]],
) -> Callable[..., dict[str, object]]:
    """Publish the Krea token-sign mask as a processed model condition."""

    def wrapped_extra_conds(**kwargs: object) -> dict[str, object]:
        """Attach a validated sequence multiplier without altering other conditions."""

        output = previous_extra_conds(**kwargs)
        if not isinstance(output, dict):
            raise TypeError("Krea extra conditions must be a dictionary.")
        multiplier = kwargs.get(ENCODER_MASK_KEY)
        if multiplier is not None:
            if not isinstance(multiplier, torch.Tensor):
                raise TypeError("Krea NegPiP sign mask must be a tensor.")
            if (
                multiplier.ndim != 3
                or multiplier.shape[0] != 1
                or multiplier.shape[2] != 1
            ):
                raise ValueError(
                    "Krea NegPiP sign mask must have shape (1, sequence, 1)."
                )
            output[CONDITION_MASK_KEY] = conds.CONDRegular(multiplier)
        return output

    return wrapped_extra_conds


def krea2_diffusion_negpip_wrapper(
    executor: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object:
    """Move a processed Krea sign mask into call-local transformer options."""

    positional_options = args[5] if len(args) > 5 else None
    transformer_options = (
        positional_options
        if positional_options is not None
        else kwargs.get("transformer_options", {})
    )
    if not isinstance(transformer_options, dict):
        raise TypeError("Krea transformer options must be a dictionary.")
    prepared = transformer_options.copy()
    multiplier = kwargs.get(CONDITION_MASK_KEY)
    if multiplier is not None:
        if not isinstance(multiplier, torch.Tensor):
            raise TypeError("Krea NegPiP processed mask must be a tensor.")
        prepared[TRANSFORMER_MASK_KEY] = multiplier
    if len(args) > 5:
        prepared_args = list(args)
        prepared_args[5] = prepared
        return executor(*prepared_args, **kwargs)
    kwargs["transformer_options"] = prepared
    return executor(*args, **kwargs)


def krea2_attn1_negpip(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pe: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    extra_options: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor | None]:
    """Apply negative signs only to Krea text values in the joint token stream."""

    options = {} if extra_options is None else extra_options
    multiplier = options.get(TRANSFORMER_MASK_KEY)
    if multiplier is None:
        return {"q": query, "k": key, "v": value, "pe": pe, "attn_mask": attn_mask}
    if not isinstance(multiplier, torch.Tensor):
        raise TypeError("Krea NegPiP attention mask must be a tensor.")
    image_slice = options.get("img_slice")
    if (
        not isinstance(image_slice, (list, tuple))
        or len(image_slice) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in image_slice
        )
    ):
        raise ValueError("Krea NegPiP requires the model's text/image token boundary.")
    text_length = image_slice[0]
    if text_length != multiplier.shape[1] or value.shape[2] < text_length:
        raise ValueError(
            "Krea NegPiP mask does not match the joint attention sequence."
        )
    if multiplier.shape[0] not in {1, value.shape[0]} or multiplier.shape[2] != 1:
        raise ValueError("Krea NegPiP mask has an incompatible batch or channel shape.")
    text_multiplier = multiplier.to(device=value.device, dtype=value.dtype).unsqueeze(1)
    prepared_value = value.to(copy=True)
    prepared_value[:, :, :text_length, :] *= text_multiplier
    return {
        "q": query,
        "k": key,
        "v": prepared_value,
        "pe": pe,
        "attn_mask": attn_mask,
    }


def _token_weight(pair: tuple[object, ...]) -> float:
    """Return one finite scalar token weight from a tokenizer tuple."""

    if (
        len(pair) < 2
        or isinstance(pair[1], bool)
        or not isinstance(pair[1], (int, float))
    ):
        raise TypeError("Krea token weights must be numeric.")
    weight = float(pair[1])
    if not torch.isfinite(torch.tensor(weight)):
        raise ValueError("Krea token weights must be finite.")
    return weight


def _template_end(section: list[tuple[object, ...]]) -> int:
    """Resolve the exact Krea system and user-opening prefix boundary."""

    count = 0
    template_end = -1
    for index, pair in enumerate(section):
        token = pair[0]
        if (
            not isinstance(token, torch.Tensor)
            and token == IM_START_TOKEN
            and count < 2
        ):
            template_end = index
            count += 1
    if (
        len(section) > template_end + 3
        and section[template_end + 1][0] == USER_TOKEN
        and section[template_end + 2][0] == NEWLINE_TOKEN
    ):
        template_end += 3
    return template_end
