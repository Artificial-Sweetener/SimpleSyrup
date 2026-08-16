# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own generic prompt defaults for the SDXL visual acceptance matrix."""

BASE_POSITIVE_L = (
    "best quality, masterpiece, very aesthetic, official art, (2girls:1.4), "
    "exactly two adult women, mature females, both women visible, two-shot "
    "composition, close upper-body portrait, centered together, side by side, "
    "shoulder to shoulder, facing the viewer, interacting, shared camera, shared "
    "simple indoor background, unified perspective, unified soft lighting"
)
BASE_POSITIVE_G = (
    "cohesive high quality anime key art, exactly two adult women visible together "
    "in one close centered upper-body two-shot, side by side, shoulder to shoulder, "
    "one simple indoor scene, one camera, unified lighting and perspective"
)
REGIONAL_SCENE_G = (
    "cohesive high quality anime key art, close centered upper-body two-shot, one "
    "shared simple indoor scene, one shared camera, unified lighting and perspective"
)
BASE_NEGATIVE_L = (
    "child, teen, loli, man, male, boy, 1girl, solo, one person, single subject, "
    "portrait, wide shot, distant subject, back view, split screen, collage, panel "
    "boundary, separate scenes, 3girls, 4girls, multiple girls, crowd, group "
    "portrait, more than two people, duplicated person, fused body, merged face, "
    "extra limbs, missing limbs, cropped, blurry, low quality, large breasts, huge "
    "breasts, cleavage, low cut, lingerie, text, watermark"
)
BASE_NEGATIVE_G = (
    "bad quality, worst quality, split composition, collage, separate backgrounds, "
    "incoherent lighting, child, man, male, boy, duplicate bodies, fused people, "
    "low quality"
)
LEFT_BASE_L = (
    "1girl, one of exactly two adult women, together in the foreground, adult woman "
    "on the left, long pink twintails, pink eyes, black sleeveless dress, black "
    "ribbons, small breasts, flat chest, confident smirk, leaning toward the other "
    "woman"
)
LEFT_BASE_G = "adult pink-haired woman in an elegant black dress on the left"
RIGHT_BASE_L = (
    "1girl, one of exactly two adult women, together in the foreground, adult woman "
    "on the right, long black hair, blue ribbons, black cropped hoodie, black denim "
    "shorts, small breasts, flat chest, pink eyes, pouting, shoulder touching the "
    "other woman"
)
RIGHT_BASE_G = "adult black-haired woman in modern black clothes on the right"
