"""A ViT backbone whose blocks apply a shared TokenReducer to all tokens."""

import torch.nn as nn

from src.core.backbones import ViTBackbone
from src.reduction.block import ReducerBlock


class ReducedViTBackbone(ViTBackbone):
    """ViTBackbone with every Block swapped for a ReducerBlock.

    One reducer instance is shared by the whole stack (AViT accumulates halting
    state across depth), and each block is told its own index so that
    schedule-driven reducers like DynamicViT can prune only at set depths.
    """

    def __init__(self, block_config, depth, position_encoding_size, input_size,
                 has_class_token=False, window_indices=(), windowed_class=None,
                 windowed_overrides=None, block_class="Block", reducer=None,
                 **kwargs):
        super().__init__(
            block_config=block_config,
            depth=depth,
            position_encoding_size=position_encoding_size,
            input_size=input_size,
            block_class=block_class,
            has_class_token=has_class_token,
            window_indices=window_indices,
            windowed_class=windowed_class,
            windowed_overrides=windowed_overrides,
        )

        new_blocks = nn.Sequential()
        for i in range(depth):
            block_config_i = block_config.copy()
            if i in window_indices:
                if windowed_overrides is not None:
                    block_config_i |= windowed_overrides
            else:
                block_config_i["window_size"] = None
            new_blocks.append(ReducerBlock(
                input_size=input_size,
                has_class_token=has_class_token,
                reducer=reducer,
                block_index=i,
                depth=depth,
                **block_config_i,
            ))
        self.blocks = new_blocks
