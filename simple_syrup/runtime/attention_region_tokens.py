# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Map readable prompt concepts to exact SDXL and Anima token positions."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..domain.attention_region_maps import (
    AttentionTokenCatalog,
    AttentionTokenSpan,
    OpenVocabularyContext,
)
from ..domain.regional_model_capabilities import RegionalModelFamily


class AttentionPromptTokenizer:
    """Build model-family token catalogs through a live Comfy CLIP value."""

    def build_catalog(
        self,
        *,
        clip: object,
        prompt_text: str,
        model_family: RegionalModelFamily,
    ) -> AttentionTokenCatalog:
        """Return comma-concept spans aligned to the denoiser context sequence."""

        tokenize = getattr(clip, "tokenize", None)
        if not callable(tokenize):
            raise TypeError(
                "Attention prompt mapping requires a tokenizable CLIP value."
            )
        tokenized = tokenize(prompt_text, return_word_ids=True)
        family_tokens = _family_tokens(tokenized, model_family)
        full = _flatten_tokens(family_tokens)
        if not full:
            raise ValueError("Attention prompt tokenization returned no tokens.")

        concepts = tuple(
            part.strip() for part in prompt_text.split(",") if part.strip()
        )
        spans: list[AttentionTokenSpan] = []
        occurrences: Counter[str] = Counter()
        search_start = 0
        for concept in concepts:
            match = _find_concept_tokens(
                tokenize=tokenize,
                concept=concept,
                model_family=model_family,
                full=full,
                search_start=search_start,
            )
            if match is None:
                continue
            indices, search_start = match
            normalized = " ".join(concept.casefold().replace("_", " ").split())
            occurrences[normalized] += 1
            spans.append(AttentionTokenSpan(concept, occurrences[normalized], indices))
        return AttentionTokenCatalog(
            len(full),
            tuple(spans),
            tuple(item[0] for item in full),
        )

    def resolve_query(
        self,
        *,
        clip: object,
        query: str,
        catalog: AttentionTokenCatalog,
        model_family: RegionalModelFamily,
    ) -> tuple[AttentionTokenSpan, ...]:
        """Resolve an exact label or complete distributed native token coverage."""

        exact = catalog.exact_matches(query)
        if exact:
            return exact
        tokenize = getattr(clip, "tokenize", None)
        if not callable(tokenize):
            raise TypeError("Attention query matching requires a tokenizable CLIP.")
        words = tuple(
            match.group(0).strip("()[]{}\"'")
            for match in re.finditer(r"[^\s,;|]+", query)
            if match.group(0).strip("()[]{}\"'")
        )
        if not words:
            return ()
        used_indices: set[int] = set()
        resolved_indices: list[int] = []
        word_index = 0
        while word_index < len(words):
            match: tuple[tuple[int, ...], int] | None = None
            consumed_words = 0
            for word_count in range(len(words) - word_index, 0, -1):
                phrase = " ".join(words[word_index : word_index + word_count])
                candidate = _find_unused_concept_tokens(
                    tokenize=tokenize,
                    concept=phrase,
                    model_family=model_family,
                    full_ids=catalog.token_ids,
                    used_indices=used_indices,
                )
                if candidate is not None:
                    match = candidate
                    consumed_words = word_count
                    break
            if match is None:
                return ()
            indices, _end = match
            used_indices.update(indices)
            resolved_indices.extend(indices)
            word_index += consumed_words
        return (AttentionTokenSpan(query, 1, tuple(sorted(set(resolved_indices)))),)

    def encode_open_vocabulary_query(
        self,
        *,
        clip: object,
        query: str,
    ) -> OpenVocabularyContext:
        """Encode one absent SDXL phrase without changing sampler conditioning."""

        tokenize = getattr(clip, "tokenize", None)
        encode = getattr(clip, "encode_from_tokens", None)
        if not callable(tokenize) or not callable(encode):
            raise TypeError("Open-vocabulary search requires an encodable CLIP value.")
        tokens = tokenize(query, return_word_ids=True)
        stream = _flatten_tokens(
            _family_tokens(tokens, RegionalModelFamily.STANDARD_UNET)
        )
        indices = tuple(index for index, item in enumerate(stream) if item[2] != 0)
        if not indices:
            raise ValueError(f"Open-vocabulary query {query!r} has no semantic tokens.")
        encoded = encode(tokens)
        if not isinstance(encoded, torch.Tensor):
            raise TypeError("Open-vocabulary CLIP encoding did not return a tensor.")
        return OpenVocabularyContext(
            query,
            encoded.detach().to(device="cpu", dtype=torch.float16),
            indices,
        )


def _family_tokens(tokenized: object, family: RegionalModelFamily) -> object:
    """Select the token stream that feeds each supported denoiser context."""

    if not isinstance(tokenized, Mapping):
        raise TypeError("Comfy CLIP tokenization must return a mapping.")
    key = "t5xxl" if family is RegionalModelFamily.ANIMA else "g"
    value = tokenized.get(key)
    if value is None and family is RegionalModelFamily.STANDARD_UNET:
        value = tokenized.get("l")
    if value is None:
        raise ValueError(f"Attention prompt tokenization has no '{key}' stream.")
    return value


def _flatten_tokens(value: object) -> tuple[tuple[object, float, int], ...]:
    """Flatten Comfy token chunks while retaining special-token positions."""

    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError("Attention token stream must contain token chunks.")
    flattened: list[tuple[object, float, int]] = []
    for chunk in value:
        if not isinstance(chunk, Sequence) or isinstance(chunk, str | bytes):
            raise TypeError("Attention token chunk must be a sequence.")
        for item in chunk:
            if not isinstance(item, Sequence) or len(item) < 3:
                raise ValueError(
                    "Attention token entries require id, weight, and word id."
                )
            word_id = item[2]
            if isinstance(word_id, bool) or not isinstance(word_id, int):
                raise TypeError("Attention token word ids must be integers.")
            flattened.append((item[0], float(item[1]), word_id))
    return tuple(flattened)


def _find_concept_tokens(
    *,
    tokenize: Any,
    concept: str,
    model_family: RegionalModelFamily,
    full: tuple[tuple[object, float, int], ...],
    search_start: int,
) -> tuple[tuple[int, ...], int] | None:
    """Find one concept occurrence using tokenizer-equivalent prefix variants."""

    full_ids = tuple(item[0] for item in full)
    for candidate in (concept, f" {concept}", f", {concept}"):
        candidate_tokens = _flatten_tokens(
            _family_tokens(tokenize(candidate, return_word_ids=True), model_family)
        )
        candidate_ids = tuple(item[0] for item in candidate_tokens if item[2] != 0)
        if not candidate_ids:
            continue
        start = _find_subsequence(full_ids, candidate_ids, search_start)
        if start is not None:
            indices = tuple(range(start, start + len(candidate_ids)))
            return indices, start + len(candidate_ids)
    return None


def _find_subsequence(
    values: tuple[object, ...],
    target: tuple[object, ...],
    start: int,
) -> int | None:
    """Return the first target occurrence at or after a stable cursor."""

    final_start = len(values) - len(target)
    for index in range(max(0, start), final_start + 1):
        if values[index : index + len(target)] == target:
            return index
    return None


def _find_unused_concept_tokens(
    *,
    tokenize: Any,
    concept: str,
    model_family: RegionalModelFamily,
    full_ids: tuple[object, ...],
    used_indices: set[int],
) -> tuple[tuple[int, ...], int] | None:
    """Find one tokenizer-equivalent phrase occurrence not already consumed."""

    for candidate in (concept, f" {concept}", f", {concept}"):
        candidate_tokens = _flatten_tokens(
            _family_tokens(tokenize(candidate, return_word_ids=True), model_family)
        )
        candidate_ids = tuple(item[0] for item in candidate_tokens if item[2] != 0)
        if not candidate_ids:
            continue
        search_start = 0
        while (
            start := _find_subsequence(full_ids, candidate_ids, search_start)
        ) is not None:
            indices = tuple(range(start, start + len(candidate_ids)))
            if used_indices.isdisjoint(indices):
                return indices, start + len(candidate_ids)
            search_start = start + 1
    return None


ATTENTION_PROMPT_TOKENIZER = AttentionPromptTokenizer()
