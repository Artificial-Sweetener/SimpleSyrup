"""Build installed allocation-free Anima graphs for surface tests."""

from __future__ import annotations

import comfy.ops
import torch
from comfy.ldm.anima.model import Anima

from simple_syrup.runtime.regional_lora.anima_targets import ANIMA_BLOCK_COUNT


def installed_meta_anima() -> Anima:
    """Return the real installed Anima graph with allocation-free meta weights."""

    return Anima(
        max_img_h=2,
        max_img_w=2,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=2048,
        num_blocks=ANIMA_BLOCK_COUNT,
        num_heads=16,
        mlp_ratio=4.0,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d",
        pos_emb_learnable=False,
        pos_emb_interpolation="crop",
        use_adaln_lora=True,
        adaln_lora_dim=256,
        extra_per_block_abs_pos_emb=False,
        device=torch.device("meta"),
        dtype=torch.float16,
        operations=comfy.ops.disable_weight_init,
    )
