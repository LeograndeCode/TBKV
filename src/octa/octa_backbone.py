"""OCTA ViT backbone — replaces every block with an :class:`OCTABlock` and
threads token positions between blocks so the offset-window tracking stays
meaningful after the token set is reduced.
"""

import torch
import torch.nn as nn

from src.core.backbones import ViTBackbone
from src.octa.octa_blocks import OCTABlock


_OCTA_KEYS = (
    "enable_object_cache",
    "cluster_similarity_threshold",
    "offset_window_size",
    "object_match_threshold",
    "object_merge_ratio",
    "track_eviction_patience",
    "merging_iterations",
    "caching",
)


class OCTAViTBackbone(ViTBackbone):
    def __init__(self, block_config, depth, position_encoding_size, input_size,
                 has_class_token=False, window_indices=(), windowed_class=None,
                 windowed_overrides=None, block_class="Block", **kwargs):
        octa_config = {k: block_config.pop(k) for k in _OCTA_KEYS if k in block_config}
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
        self.has_class_token = has_class_token
        self.position_encoding_size = tuple(position_encoding_size)

        new_blocks = nn.Sequential()
        for i in range(depth):
            block_config_i = block_config.copy()
            if i in window_indices:
                if windowed_overrides is not None:
                    block_config_i |= windowed_overrides
            else:
                block_config_i["window_size"] = None
            new_blocks.append(
                OCTABlock(
                    input_size=input_size,
                    has_class_token=has_class_token,
                    block_idx=i,
                    **block_config_i,
                    **octa_config,
                )
            )
        self.blocks = new_blocks

    def _initial_positions(self, batch, n_patch, device, dtype):
        """Grid (row, col) coordinates for the first block's patch tokens."""
        pe = self.position_encoding_size
        if len(pe) == 2 and pe[0] * pe[1] == n_patch:
            h, w = pe
            rows = torch.arange(h, device=device, dtype=dtype).repeat_interleave(w)
            cols = torch.arange(w, device=device, dtype=dtype).repeat(h)
            grid = torch.stack([rows, cols], dim=-1)              # [N, 2]
        else:
            idx = torch.arange(n_patch, device=device, dtype=dtype)
            grid = torch.stack([idx, torch.zeros_like(idx)], dim=-1)
        return grid[None].expand(batch, -1, -1).contiguous()

    def forward(self, x):
        x = self.position_encoding(x)
        batch = x.shape[0]
        n_patch = x.shape[1] - (1 if self.has_class_token else 0)
        positions = self._initial_positions(batch, n_patch, x.device, x.dtype)
        prev_attnmap = None
        for block in self.blocks:
            x, prev_attnmap, positions = block(x, prev_attnmap, positions)
        return x
