# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build and render prompt-token diagnostics for a compound concept."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageDraw

from .artifacts import load_font, mask_coverage
from .source_workflow import build_source_workflow
from .workflow import Graph, build_workflow, configure_mask


def build_phrase_diagnostic_workflow(
    source_json: Path,
    *,
    output_prefix: str,
    concepts: tuple[str, ...],
    strengths: tuple[float, ...] | None = None,
) -> Graph:
    """Return one shared-capture graph with unfiltered concept outputs."""

    if not concepts:
        raise ValueError("Phrase diagnostics require at least one concept.")
    if strengths is None:
        strengths = (0.0,) * len(concepts)
    if len(strengths) != len(concepts):
        raise ValueError("Phrase diagnostic strengths must align with concepts.")
    graph = build_source_workflow(
        source_json,
        output_prefix=output_prefix,
        concept=concepts[0],
    )
    template_request = deepcopy(graph["10"])
    template_conversion = deepcopy(graph["100"])
    template_save = deepcopy(graph["101"])
    for node_id in tuple(graph):
        if node_id not in {"1", "2", "3", "4", "5", "6", "900"}:
            del graph[node_id]
    for index, (concept, strength) in enumerate(
        zip(concepts, strengths, strict=True),
        start=1,
    ):
        request_id = str(1000 + index)
        conversion_id = str(2000 + index)
        save_id = str(3000 + index)
        graph[request_id] = deepcopy(template_request)
        graph[conversion_id] = deepcopy(template_conversion)
        graph[conversion_id]["inputs"]["mask"] = [request_id, 2]
        graph[save_id] = deepcopy(template_save)
        graph[save_id]["inputs"]["images"] = [conversion_id, 0]
        configure_mask(
            graph,
            request_id=request_id,
            save_id=save_id,
            concept=concept,
            strength=strength,
            consensus=0.25 if strength > 0.0 else 0.0,
            split=0.0,
            minimum_size=1,
            keep_only=0,
            solidity=0.0,
            evidence_mode="concept isolation",
            prefix=f"{output_prefix}/{_slug(concept)}_{strength:.2f}",
        )
    return graph


def build_saved_phrase_diagnostic_workflow(
    source_png: Path,
    *,
    output_prefix: str,
    concepts: tuple[str, ...],
    strengths: tuple[float, ...] | None = None,
) -> Graph:
    """Return a shared-capture diagnostic from one saved proof generation."""

    if not concepts:
        raise ValueError("Phrase diagnostics require at least one concept.")
    if strengths is None:
        strengths = (0.0,) * len(concepts)
    if len(strengths) != len(concepts):
        raise ValueError("Phrase diagnostic strengths must align with concepts.")
    graph = build_workflow(
        source_png,
        output_prefix=output_prefix,
        concept=concepts[0],
        strength=0.0,
        consensus=0.0,
        split=0.0,
    )
    template_request = deepcopy(graph["14"])
    template_conversion = deepcopy(graph["1130"])
    template_save = deepcopy(graph["1131"])
    for node_id in tuple(graph):
        if node_id not in {"1", "2", "3", "4", "5", "6", "900"}:
            del graph[node_id]
    for index, (concept, strength) in enumerate(
        zip(concepts, strengths, strict=True),
        start=1,
    ):
        request_id = str(1000 + index)
        conversion_id = str(2000 + index)
        save_id = str(3000 + index)
        graph[request_id] = deepcopy(template_request)
        graph[conversion_id] = deepcopy(template_conversion)
        graph[conversion_id]["inputs"]["mask"] = [request_id, 2]
        graph[save_id] = deepcopy(template_save)
        graph[save_id]["inputs"]["images"] = [conversion_id, 0]
        configure_mask(
            graph,
            request_id=request_id,
            save_id=save_id,
            concept=concept,
            strength=strength,
            consensus=0.0,
            split=0.0,
            minimum_size=1,
            keep_only=0,
            solidity=0.0,
            evidence_mode="concept isolation",
            prefix=f"{output_prefix}/{_slug(concept)}_{strength:.2f}",
            edge_feather=0,
        )
    return graph


def compose_phrase_diagnostic_sheet(
    *,
    title: str,
    image_path: Path,
    concept_paths: tuple[tuple[str, Path], ...],
    destination: Path,
) -> None:
    """Write one labeled sheet comparing constituent token concepts."""

    panels = (("GENERATED IMAGE", image_path, False),) + tuple(
        (label.upper(), path, True) for label, path in concept_paths
    )
    panel_width = 280
    panel_height = 392
    header_height = 112
    sheet = Image.new(
        "RGB",
        (panel_width * len(panels), header_height + panel_height),
        "#101216",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 12), title, fill="white", font=load_font(30))
    draw.text(
        (18, 54),
        (
            "Same generation and shared fast capture; concept isolation without "
            "component filtering"
        ),
        fill="#b9c1cc",
        font=load_font(15),
    )
    for index, (label, path, is_mask) in enumerate(panels):
        panel = (
            Image.open(path)
            .convert("RGB")
            .resize((panel_width, panel_height), Image.Resampling.LANCZOS)
        )
        left = index * panel_width
        sheet.paste(panel, (left, header_height))
        draw.rectangle((left, 80, left + panel_width, 112), fill="#1c2028")
        detail = f"  support {mask_coverage(path):.1f}%" if is_mask else ""
        draw.text((left + 10, 86), label + detail, fill="white", font=load_font(17))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def _slug(value: str) -> str:
    """Return a stable filename fragment for one diagnostic concept."""

    return "_".join(value.casefold().replace(",", " ").split())
