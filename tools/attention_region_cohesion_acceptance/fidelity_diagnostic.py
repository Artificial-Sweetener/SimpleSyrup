# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build and render a concept-fidelity processing-stage diagnostic."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .artifacts import load_font, mask_coverage
from .workflow import Graph, build_workflow, configure_mask

STRENGTH_STAGES = (
    ("10", "101", "STRENGTH 0.06", 0.06),
    ("11", "1101", "STRENGTH 0.09", 0.09),
    ("14", "1131", "STRENGTH 0.12", 0.12),
    ("15", "1141", "STRENGTH 0.15", 0.15),
)

COMPONENT_STAGES = (
    ("10", "101", "NO COMPONENT FILTER", 1, 0),
    ("11", "1101", "MINIMUM 128 PX", 128, 0),
    ("14", "1131", "MINIMUM 512 PX", 512, 0),
    ("15", "1141", "LARGEST COMPONENT", 512, 1),
)


def build_fidelity_diagnostic_workflow(
    source_png: Path,
    *,
    output_prefix: str,
    concept: str,
) -> Graph:
    """Return one shared capture with progressively applied mask controls."""

    graph = build_workflow(
        source_png,
        output_prefix=output_prefix,
        concept=concept,
        strength=0.0,
        consensus=0.0,
        split=0.0,
    )
    stages = (
        ("10", "101", "RAW CONCEPT EVIDENCE", 0.0, 0.0, 0),
        ("11", "1101", "STRENGTH 0.15", 0.15, 0.0, 0),
        ("14", "1131", "+ CONSENSUS 0.25", 0.15, 0.25, 0),
        ("15", "1141", "+ FEATHER 8 PX", 0.15, 0.25, 8),
    )
    for request_id, save_id, label, strength, consensus, feather in stages:
        configure_mask(
            graph,
            request_id=request_id,
            save_id=save_id,
            concept=concept,
            strength=strength,
            consensus=consensus,
            split=0.0,
            minimum_size=1,
            keep_only=0,
            solidity=0.0,
            evidence_mode="concept isolation",
            prefix=f"{output_prefix}/{request_id}_{_slug(label)}",
            edge_feather=feather,
        )
    return graph


def compose_fidelity_diagnostic_sheet(
    *,
    image_path: Path,
    stage_paths: tuple[tuple[str, Path], ...],
    destination: Path,
) -> None:
    """Write a labeled sheet showing each destructive processing boundary."""

    panels = (("GENERATED IMAGE", image_path, False),) + tuple(
        (label, path, True) for label, path in stage_paths
    )
    panel_width = 320
    panel_height = 426
    header_height = 112
    sheet = Image.new(
        "RGB",
        (panel_width * len(panels), header_height + panel_height),
        "#101216",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (18, 12),
        "SDXL Hair Fine-Structure Diagnostic",
        fill="white",
        font=load_font(30),
    )
    draw.text(
        (18, 54),
        "Same generation and shared fast capture; no component filtering or solidity",
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


def build_strength_diagnostic_workflow(
    source_png: Path,
    *,
    output_prefix: str,
    concept: str,
) -> Graph:
    """Return one shared capture rendered at four concept-strength thresholds."""

    graph = build_workflow(
        source_png,
        output_prefix=output_prefix,
        concept=concept,
        strength=0.0,
        consensus=0.0,
        split=0.0,
    )
    for request_id, save_id, label, strength in STRENGTH_STAGES:
        configure_mask(
            graph,
            request_id=request_id,
            save_id=save_id,
            concept=concept,
            strength=strength,
            consensus=0.25,
            split=0.0,
            minimum_size=1,
            keep_only=0,
            solidity=0.0,
            evidence_mode="concept isolation",
            prefix=f"{output_prefix}/{request_id}_{_slug(label)}",
            edge_feather=0,
        )
    return graph


def compose_mask_diagnostic_sheet(
    *,
    title: str,
    subtitle: str,
    image_path: Path,
    stage_paths: tuple[tuple[str, Path], ...],
    destination: Path,
) -> None:
    """Write labeled masks and dimmed overlays for one shared capture."""

    panels = (("GENERATED IMAGE", image_path, False),) + tuple(
        (label, path, True) for label, path in stage_paths
    )
    panel_width = 320
    panel_height = 426
    header_height = 112
    sheet = Image.new(
        "RGB",
        (panel_width * len(panels), header_height + panel_height * 2),
        "#101216",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (18, 12),
        title,
        fill="white",
        font=load_font(30),
    )
    draw.text(
        (18, 54),
        subtitle,
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
        overlay = (
            panel
            if not is_mask
            else _dimmed_overlay(image_path, path, panel_width, panel_height)
        )
        sheet.paste(overlay, (left, header_height + panel_height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def build_component_diagnostic_workflow(
    source_png: Path,
    *,
    output_prefix: str,
    concept: str,
) -> Graph:
    """Return one shared capture with increasingly selective component cleanup."""

    graph = build_workflow(
        source_png,
        output_prefix=output_prefix,
        concept=concept,
        strength=0.12,
        consensus=0.25,
        split=0.0,
    )
    for request_id, save_id, label, minimum_size, keep_only in COMPONENT_STAGES:
        configure_mask(
            graph,
            request_id=request_id,
            save_id=save_id,
            concept=concept,
            strength=0.12,
            consensus=0.25,
            split=0.0,
            minimum_size=minimum_size,
            keep_only=keep_only,
            solidity=0.0,
            evidence_mode="concept isolation",
            prefix=f"{output_prefix}/{request_id}_{_slug(label)}",
            edge_feather=0,
        )
    return graph


def build_split_diagnostic_workflow(
    source_png: Path,
    *,
    output_prefix: str,
    concept: str,
    sensitivities: tuple[float, float, float, float],
) -> Graph:
    """Return one shared capture rendered across four split sensitivities."""

    graph = build_workflow(
        source_png,
        output_prefix=output_prefix,
        concept=concept,
        strength=0.15,
        consensus=0.25,
        split=0.0,
    )
    stages = zip(
        ("10", "11", "14", "15"),
        ("101", "1101", "1131", "1141"),
        sensitivities,
        strict=True,
    )
    for request_id, save_id, sensitivity in stages:
        configure_mask(
            graph,
            request_id=request_id,
            save_id=save_id,
            concept=concept,
            strength=0.15,
            consensus=0.25,
            split=sensitivity,
            minimum_size=512,
            keep_only=1,
            solidity=0.0,
            evidence_mode="concept isolation",
            prefix=f"{output_prefix}/split_{sensitivity:.2f}",
            edge_feather=0,
        )
    return graph


def _dimmed_overlay(
    image_path: Path,
    mask_path: Path,
    width: int,
    height: int,
) -> Image.Image:
    """Reveal selected structure over a strongly dimmed source image."""

    source = (
        Image.open(image_path)
        .convert("RGB")
        .resize((width, height), Image.Resampling.LANCZOS)
    )
    mask = (
        Image.open(mask_path)
        .convert("L")
        .resize((width, height), Image.Resampling.LANCZOS)
    )
    dimmed = source.point(lambda value: round(value * 0.1))
    return Image.composite(source, dimmed, mask)


def _slug(value: str) -> str:
    """Return one stable lowercase stage filename."""

    return "_".join(value.casefold().replace("+", " ").split())
