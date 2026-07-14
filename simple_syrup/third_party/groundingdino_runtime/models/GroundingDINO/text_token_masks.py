# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Build GroundingDINO text-token masks around special-token boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def generate_text_token_masks(
    tokenized: Mapping[str, torch.Tensor],
    special_token_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Generate sub-sentence attention, position, and category token masks."""

    input_ids = tokenized["input_ids"]
    batch_size, token_count = input_ids.shape
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for special_token_id in special_token_ids:
        special_tokens_mask |= input_ids == special_token_id

    special_token_positions = torch.nonzero(special_tokens_mask)
    attention_mask = (
        torch.eye(token_count, device=input_ids.device, dtype=torch.bool)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )
    position_ids = torch.zeros(
        (batch_size, token_count), device=input_ids.device, dtype=torch.long
    )
    category_masks: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    previous_column = 0

    for row, column in special_token_positions:
        if column == 0 or column == token_count - 1:
            attention_mask[row, column, column] = True
            position_ids[row, column] = 0
        else:
            segment = slice(previous_column + 1, column + 1)
            attention_mask[row, segment, segment] = True
            position_ids[row, segment] = torch.arange(
                0,
                column - previous_column,
                device=input_ids.device,
            )
            category_mask = torch.zeros(
                token_count, device=input_ids.device, dtype=torch.bool
            )
            category_mask[previous_column + 1 : column] = True
            category_masks[row].append(category_mask)
        previous_column = int(column)

    stacked_category_masks = [
        torch.stack(batch_category_masks, dim=0)
        for batch_category_masks in category_masks
    ]
    return attention_mask, position_ids, stacked_category_masks
