"""ViViT with EventfulPSM spatial blocks.

The only thing this adds over the stock FactorizedViViT is caching/matching-mode
control, so the evaluation harness can run the same two-pass protocol as PSM:
feed the first frames of a clip in caching mode, then the rest in matching mode.
The spatial blocks are selected via block_class in the config, so no backbone
subclass is needed -- FactorizedViViT builds EventfulPSMBlock directly.
"""

import torch

from src.models.vivit import FactorizedViViT
from src.psm.blocks import EventfulPSMBlock


class EventfulPSMViViT(FactorizedViViT):
    def _forward_view(self, x):
        # Identical to FactorizedViViT._forward_view EXCEPT it does not reset the
        # spatial model per view. The stock method calls spatial_model.reset()
        # here, which would wipe the block cache/accumulation state between the
        # caching and matching passes (and even within the caching pass). The
        # evaluation harness owns resets, via reset()/clear_cache() per clip.
        x = self.embedding(x)
        x = torch.stack(
            [self.spatial_model(x[:, t]) for t in range(x.shape[1])], dim=1
        )
        return x

    def _blocks(self):
        for module in self.modules():
            if isinstance(module, EventfulPSMBlock):
                yield module

    def set_mode(self, mode):
        if mode not in ("caching", "matching"):
            raise ValueError(f"Unknown mode '{mode}'.")
        caching = (mode == "caching")
        for block in self._blocks():
            block.caching = caching

    def clear_cache(self):
        for block in self._blocks():
            block.reset_self()
