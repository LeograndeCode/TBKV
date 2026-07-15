"""A ViT block that applies a TokenReducer to the full token sequence.

This is the *baseline* call site: it runs ToMe / EViT / DynamicViT / AViT as
published, on every token. The TBKV call site (TBKVBlock) reuses the very same
reducer objects, but hands them only the foreground tokens.

The reduction hook sits on the post-attention residual stream -- after the
attention output is added to the skip, before the MLP -- which is where ToMe
and EViT operate in their reference implementations.
"""

import torch

from src.core.blocks import Block
from src.reduction.base import ReductionContext


class ReducerBlock(Block):
    """
    :param reducer: a shared TokenReducer (one instance per stack, so that
        depth-accumulating reducers like AViT can carry state), or None.
    :param block_index: this block's depth index, used by schedule-driven
        reducers (DynamicViT prunes only at specific depths).
    :param depth: total blocks in the stack.
    """

    def __init__(self, reducer=None, block_index=0, depth=12,
                 has_class_token=False, **super_kwargs):
        super().__init__(**super_kwargs)
        # Assigned, not registered as a submodule: the reducer is shared across
        # every block, and registering it in each would duplicate it in the
        # state dict and break checkpoint loading.
        object.__setattr__(self, "reducer", reducer)
        self.block_index = block_index
        self.depth = depth
        self.has_class_token = has_class_token
        self._attn = None
        self._keys = None

    def reset_self(self):
        super().reset_self()
        self._attn = None
        self._keys = None
        if self.reducer is not None and self.block_index == 0:
            self.reducer.reset()

    def _forward_attention(self, x):
        # Capture the attention map and keys the reducers need. Block's own
        # _forward_attention discards both.
        x = self._partition_windows(x, in_qkv_domain=True)
        q, k, v = self._partition_heads(x)
        k = self._pool_tokens(k)
        v = self._pool_tokens(v)

        x = self.matmul(q / self.scale, k.transpose(-2, -1))
        if self.relative_position is not None:
            x = self.relative_position(x, q)
        x = x.softmax(dim=-1)

        self._attn = x
        self._keys = k

        x, ats_indices = self._adaptive_token_sampling(x, v)
        x, v, old_dtype = self._cast_matmul_2(x, v)
        x = self.matmul(x, v)
        x = self._recombine_heads(x)
        x = self._recombine_windows(x)
        x = self._uncast_matmul_2(x, old_dtype)
        return x, ats_indices

    def forward(self, x):
        skip_1 = x
        x = self.input_layer_norm(x)
        x = self.qkv(x)

        x, ats_indices = self._forward_attention(x)
        skip_1 = self._gather_ats_skip(skip_1, ats_indices)

        x = self.projection(x)
        x = self.add(self.drop_path(x), skip_1)

        x = self._reduce(x)

        skip_2 = x
        x = self.mlp_layer_norm(x)
        x = self._forward_mlp(x)
        x = self.add(self.drop_path(x), skip_2)
        return x

    def _reduce(self, x):
        if self.reducer is None:
            return x
        ctx = ReductionContext(
            attn=self._attn,
            keys=self._keys,
            has_class_token=self.has_class_token,
            block_index=self.block_index,
            depth=self.depth,
        )
        op = self.reducer.plan(x, ctx)
        return x if op is None else op(x)
