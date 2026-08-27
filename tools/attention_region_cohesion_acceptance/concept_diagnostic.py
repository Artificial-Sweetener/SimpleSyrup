# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build and render focused related-token attention diagnostics."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance

from .artifacts import load_font, mask_coverage
from .workflow import Graph, build_workflow, configure_mask


def build_related_concept_workflow(
    source_png: Path,
    *,
    output_prefix: str,
) -> Graph:
    """Compare hair concepts from one shared contextual capture."""

    graph = build_workflow(
        source_png,
        output_prefix=output_prefix,
        concept="pink hair",
        strength=0.15,
        consensus=0.25,
        split=0.35,
    )
    for request_id, save_id, concept, suffix in (
        ("10", "101", "hair", "hair"),
        ("11", "1101", "pink hair", "pink_hair"),
        ("14", "1131", "twintails", "twintails"),
        ("15", "1141", "pink hair, twintails", "combined"),
    ):
        configure_mask(
            graph,
            request_id=request_id,
            save_id=save_id,
            concept=concept,
            strength=0.15,
            consensus=0.25,
            split=0.0,
            minimum_size=1,
            keep_only=0,
            solidity=0.0,
            evidence_mode="concept isolation",
            prefix=f"{output_prefix}/{suffix}",
        )
    return graph


def compose_related_concept_sheet(
    *,
    image_path: Path,
    hair_path: Path,
    pink_hair_path: Path,
    twintails_path: Path,
    combined_path: Path,
    destination: Path,
) -> None:
    """Write one labeled diagnostic comparing related conditioning spans."""

    panels = (
        ("GENERATED IMAGE", image_path, False),
        ("HAIR", hair_path, True),
        ("PINK HAIR", pink_hair_path, True),
        ("TWINTAILS", twintails_path, True),
        ("PINK HAIR + TWINTAILS", combined_path, True),
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
        "SDXL Hair Concept Localization Diagnostic",
        fill="white",
        font=load_font(30),
    )
    draw.text(
        (18, 54),
        "Same generation and one shared fast capture; no component filtering",
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
        draw.text((left + 10, 86), label + detail, fill="white", font=load_font(18))
        overlay = (
            panel
            if not is_mask
            else _dimmed_overlay(
                image_path,
                path,
                panel_width,
                panel_height,
            )
        )
        sheet.paste(overlay, (left, header_height + panel_height))
        draw.rectangle(
            (
                left,
                header_height + panel_height,
                left + panel_width,
                header_height + panel_height + 30,
            ),
            fill="#1c2028",
        )
        overlay_label = "SOURCE IMAGE" if not is_mask else "MASKED IMAGE"
        draw.text(
            (left + 10, header_height + panel_height + 5),
            overlay_label,
            fill="white",
            font=load_font(16),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def _dimmed_overlay(
    image_path: Path,
    mask_path: Path,
    width: int,
    height: int,
) -> Image.Image:
    """Reveal mask coverage by dimming non-selected source pixels."""

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
    dimmed = ImageEnhance.Brightness(source).enhance(0.12)
    return Image.composite(source, dimmed, mask)
