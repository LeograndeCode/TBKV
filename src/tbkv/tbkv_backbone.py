import torch
import torch.nn as nn

from src.core.backbones import ViTBackbone

from src.tbkv.tbkv_blocks import TBKVBlock


class TBKVViTBackbone(ViTBackbone):
    """
    Only changes the block class to TBKVBlock, which implements the TBKV logic.
    """

    def __init__(self, block_config, depth, position_encoding_size, input_size,
                 has_class_token=False, window_indices=(), windowed_class=None,
                 windowed_overrides=None, block_class="Block", **kwargs):
        _tbkv_keys = ("local_merge_ratio", "r_match", "caching", "raw", "use_tome", "tome_r")
        tbkv_config = {k: block_config.pop(k) for k in _tbkv_keys if k in block_config}
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

        # Replace all Block instances with TBKVBlock instances.
        # Layer-wise adaptive matching ratio (FrameFusion, ICCV 2025).
        # Early blocks get aggressive reuse (high r_match).
        # Deep blocks get conservative reuse (low r_match).
        r_match_base = tbkv_config.get("r_match", 0.75)
        r_max = min(r_match_base + 0.10, 0.95)
        r_min = max(r_match_base - 0.10, 0.30)

        new_blocks = nn.Sequential()
        for i in range(depth):
            block_config_i = block_config.copy()
            if i in window_indices:
                if windowed_overrides is not None:
                    block_config_i |= windowed_overrides
            else:
                block_config_i["window_size"] = None

            # Compute per-layer r_match using linear schedule
            if depth > 1:
                r_match_i = r_max - (i / (depth - 1)) * (r_max - r_min)
            else:
                r_match_i = r_match_base

            # Override r_match for this specific block
            tbkv_config_i = tbkv_config.copy()
            tbkv_config_i["r_match"] = r_match_i

            new_blocks.append(TBKVBlock(input_size=input_size, has_class_token=has_class_token, **block_config_i, **tbkv_config_i))
        self.blocks = new_blocks

    def forward(self, x):
        x = self.position_encoding(x)
        prev_attnmap = None
        for block in self.blocks:
            x, prev_attnmap = block(x, prev_attnmap) 
        return x