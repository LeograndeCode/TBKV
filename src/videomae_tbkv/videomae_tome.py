"""
ToMe ("Token Merging: Your ViT But Faster", Bolya et al., ICLR 2023) applied
to the VideoMAE ViT-B/16 Kinetics-400 fine-tune.

Follows the official ToMe repo's MAE patch (ToMe/tome/patch/mae.py +
patch/timm.py), adapted to the VideoMAETBKV model in this package:
- Each block merges r tokens after attention via bipartite soft matching on
  the attention keys (mean over heads), with weighted averaging so a merged
  token keeps the influence of everything inside it (merge_wavg).
- Optional proportional attention (prop_attn); the ToMe repo recommends
  False for MAE-family models, which VideoMAE is.
- Final mean pooling is proportional to token size, as in the MAE patch.

Applied as a class swap (apply_tome) on an already-built/loaded model, like
the original ToMe patch. FLOPs come from the same src.core.counting modules
the base model uses, so numbers are directly comparable with TBKV.
"""

from typing import Tuple

import torch

from src.tbkv.merge import bipartite_soft_matching

from .videomae import TBKVAttention, TBKVBlock, VideoMAETBKV


def parse_r(num_layers, r):
    """Schedule of per-layer r values (from ToMe/tome/utils.py)."""
    inc = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inc = r
    min_val = int(r * (1.0 - inc))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)
    return [int(min_val + step * i) for i in range(num_layers)]


def merge_wavg(merge, x, size=None):
    """Weighted-average merge tracking token size (from ToMe/tome/merge.py)."""
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size


class ToMeVideoMAEAttention(TBKVAttention):
    """
    Modifications (mirrors ToMe's ToMeAttention):
     - Optional proportional attention using token size.
     - Return the mean of k over heads as the merging metric.
    """

    def forward(  # noqa: signature intentionally differs from TBKVAttention
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        head_dim = C // self.num_heads
        q = self.q(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn = self.matmul(q * self.scale, k.transpose(-2, -1))

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)

        x = self.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x, k.mean(1)


class ToMeVideoMAEBlock(TBKVBlock):
    """
    Modifications (mirrors ToMe's ToMeBlock):
     - Apply ToMe between the attention and mlp blocks.
     - Compute and propagate token size.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        info = self.state["tome"]
        attn_size = info["size"] if info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = self.add(x, x_attn)

        r = info["r"].pop(0)
        if r > 0:
            # src/tbkv/merge.py's bipartite_soft_matching takes r as a
            # fraction of the token count; convert the absolute r.
            merge, _ = bipartite_soft_matching(
                metric, r / x.shape[1], class_token=False, distill_token=False
            )
            x, info["size"] = merge_wavg(merge, x, info["size"])

        x = self.add(x, self.mlp(self.norm2(x)))
        return x


class ToMeVideoMAE(VideoMAETBKV):
    """
    Modifications (mirrors ToMe's make_tome_class for MAE):
    - Initialize the per-layer r schedule and token size each forward.
    - Global average pool proportional to token size.
    """

    def forward(self, x):
        self.state["tome"] = {
            "r": parse_r(len(self.blocks), self.r),
            "size": None,
            "prop_attn": self.prop_attn,
        }
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x)
        n_tokens = x.shape[1]
        for block in self.blocks:
            x = block(x)

        size = self.state["tome"]["size"]
        if size is not None:
            # Weighted mean over the ORIGINAL token count (MAE-patch style).
            x = (x * size).sum(dim=1) / n_tokens
        else:
            x = x.mean(dim=1)
        x = self.fc_norm(x)
        return self.head(x)


def apply_tome(model: VideoMAETBKV, r=0, prop_attn: bool = False):
    """
    Apply ToMe to a (loaded) VideoMAETBKV model in place, ToMe-repo style.
    Set the number of merged tokens per block afterwards via model.r.

    :param r: tokens merged per block (int, list per block, or (r, inc) tuple)
    :param prop_attn: proportional attention (False for MAE-family models)
    """
    model.__class__ = ToMeVideoMAE
    model.r = r
    model.prop_attn = prop_attn
    for block in model.blocks:
        block.__class__ = ToMeVideoMAEBlock
        block.attn.__class__ = ToMeVideoMAEAttention
    return model
