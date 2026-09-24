# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test readable concept alignment to supported denoiser token streams."""

from __future__ import annotations

from simple_syrup.domain.attention_concepts import parse_attention_concepts
from simple_syrup.domain.regional_model_capabilities import RegionalModelFamily
from simple_syrup.runtime.attention_region_tokens import ATTENTION_PROMPT_TOKENIZER


class _FakeClip:
    """Tokenize comma tags into predictable model-family streams."""

    def tokenize(self, text: str, return_word_ids: bool = False) -> dict[str, object]:
        """Return special-delimited integer character tokens for both families."""

        del return_word_ids
        content = [(ord(character), 1.0, 1) for character in text]
        stream = [[(1, 1.0, 0), *content, (2, 1.0, 0)]]
        return {"g": stream, "l": stream, "t5xxl": stream}


def test_sdxl_catalog_preserves_repeated_readable_prompt_occurrences() -> None:
    """Keep repeated comma concepts separate while advancing token positions."""

    catalog = ATTENTION_PROMPT_TOKENIZER.build_catalog(
        clip=_FakeClip(),
        prompt_text="1girl, pink hair, pink hair",
        model_family=RegionalModelFamily.STANDARD_UNET,
    )

    assert tuple(span.display_label for span in catalog.spans) == (
        "1girl",
        "pink hair",
        "pink hair #2",
    )
    assert catalog.spans[1].token_indices != catalog.spans[2].token_indices
    assert tuple(
        catalog.token_ids[index] for index in catalog.spans[1].semantic_head_indices
    ) == tuple(ord(character) for character in "hair")


def test_anima_catalog_selects_adapted_t5_token_positions() -> None:
    """Use Anima's adapted T5 stream instead of its Qwen source tokens."""

    catalog = ATTENTION_PROMPT_TOKENIZER.build_catalog(
        clip=_FakeClip(),
        prompt_text="smug, selfie",
        model_family=RegionalModelFamily.ANIMA,
    )

    assert tuple(span.label for span in catalog.spans) == ("smug", "selfie")
    assert catalog.sequence_length == len("smug, selfie") + 2


def test_pipe_parser_preserves_commas_and_explicit_concept_order() -> None:
    """Treat only vertical bars as concept boundaries."""

    assert parse_attention_concepts(
        " 1girl, pink hair black dress | cat || smug expression "
    ) == ("1girl, pink hair black dress", "cat", "smug expression")


def test_distributed_query_uses_complete_native_spans_in_any_prompt_order() -> None:
    """Cover a composite query through longest native spans across prompt entries."""

    clip = _FakeClip()
    catalog = ATTENTION_PROMPT_TOKENIZER.build_catalog(
        clip=clip,
        prompt_text="black dress, outdoors, 1girl, city, pink hair",
        model_family=RegionalModelFamily.STANDARD_UNET,
    )

    spans = ATTENTION_PROMPT_TOKENIZER.resolve_query(
        clip=clip,
        query="1girl, pink hair black dress",
        catalog=catalog,
        model_family=RegionalModelFamily.STANDARD_UNET,
    )

    assert tuple(span.label for span in spans) == ("1girl, pink hair black dress",)
    resolved_ids = tuple(catalog.token_ids[index] for index in spans[0].token_indices)
    assert all(
        ord(character) in resolved_ids for character in "1girlpinkhairblackdress"
    )
    assert tuple(
        catalog.token_ids[index] for index in spans[0].semantic_head_indices
    ) == tuple(ord(character) for character in "dress")


def test_distributed_query_requires_complete_native_coverage() -> None:
    """Return no native span when any meaningful phrase is absent."""

    clip = _FakeClip()
    catalog = ATTENTION_PROMPT_TOKENIZER.build_catalog(
        clip=clip,
        prompt_text="1girl, pink hair",
        model_family=RegionalModelFamily.STANDARD_UNET,
    )

    assert (
        ATTENTION_PROMPT_TOKENIZER.resolve_query(
            clip=clip,
            query="1girl black dress",
            catalog=catalog,
            model_family=RegionalModelFamily.STANDARD_UNET,
        )
        == ()
    )
