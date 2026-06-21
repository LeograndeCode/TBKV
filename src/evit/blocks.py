import torch
import torch.nn as nn
import torch.nn.functional as func
from math import sqrt, prod

from src.core.base import ExtendedModule, numeric_tuple
from src.core.counting import CountedAdd, CountedLinear, CountedMatmul
from src.core.blocks import Block

from src.core.utils import (
    DropPath,
    RelativePositionEmbedding,
    expand_row_index,
)
from src.core.constants import LN_EPS
from src.utils.image import pad_to_size

from .cache import PrototypeCache

from src.evit.modules import matching, caching 


class EVITBlock(Block):

    def __init__(self, merge_ratio: float = 0.5, caching: bool = False,reuse_threshold: float = 0.75,reduce_threshold: float = 0.5, **super_kwargs):

        super_kwargs.pop("has_class_token", None)

        super().__init__(**super_kwargs)

        # Caching mode
        self.caching = caching

        # Hyperparameters 
        self.merge_ratio = merge_ratio

        # Cache
        self.cache = PrototypeCache(reuse_threshold=reuse_threshold, reduce_threshold=reduce_threshold)

        # Split QKV so that Block has attributes q, k and v

        qkv: nn.Linear = self.qkv
        dim = qkv.in_features
        all_head_dim = qkv.out_features // 3
        device = qkv.weight.device
        dtype = qkv.weight.dtype
        

        # Split weights into q, k, v chunks
        w_q, w_k, w_v = qkv.weight.data.chunk(3, dim=0)
        
        
        def build_linear(w, b=None):
            lin = CountedLinear(dim, all_head_dim, device=device, dtype=dtype)
            lin.weight.data.copy_(w)
            if b is not None:
                lin.bias.data.copy_(b)
            return lin

        self.q = build_linear(w_q, None)
        self.k = build_linear(w_k, None)  
        self.v = build_linear(w_v, None)
        # dim saved for FLOPs accounting
        self._dim = dim
        self._head_dim = all_head_dim

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        
        """
        Override to allow loading from checkpoints that have either combined qkv or separate q/k/v.
        """
        qkv_w = prefix + "qkv.weight"
        qkv_b = prefix + "qkv.bias"
        q_w   = prefix + "q.weight"
        q_b   = prefix + "q.bias"

        # Case 1: checkpoint has combined qkv → split into q/k/v for our block
        if qkv_w in state_dict and q_w not in state_dict:
            w_q, w_k, w_v = state_dict[qkv_w].chunk(3, dim=0)
            state_dict[prefix + "q.weight"] = w_q
            state_dict[prefix + "k.weight"] = w_k
            state_dict[prefix + "v.weight"] = w_v
        if qkv_b in state_dict and q_b not in state_dict:
            b_q, b_k, b_v = state_dict[qkv_b].chunk(3, dim=0)
            state_dict[prefix + "q.bias"] = b_q
            state_dict[prefix + "k.bias"] = b_k
            state_dict[prefix + "v.bias"] = b_v

        # Case 2: checkpoint has split q/k/v but no qkv → reconstruct qkv
        # (self.qkv is still used in the standard/caching forward path)
        if q_w in state_dict and qkv_w not in state_dict:
            state_dict[qkv_w] = torch.cat(
                [state_dict[prefix + "q.weight"],
                 state_dict[prefix + "k.weight"],
                 state_dict[prefix + "v.weight"]], dim=0)
        if q_b in state_dict and qkv_b not in state_dict:
            state_dict[qkv_b] = torch.cat(
                [state_dict[prefix + "q.bias"],
                 state_dict[prefix + "k.bias"],
                 state_dict[prefix + "v.bias"]], dim=0)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):

        skip_1 = x
        x = self.input_layer_norm(x)
        B, N, C = x.shape

        # Matching incoming frames
        if not self.caching:

            x_unm, x_reuse, x_proto, k_reuse, v_reuse, k_proto, v_proto = matching(

                x,

                self.k,

                self.v,

                self.cache,

            )

            # Get new tokens from prototypes, reused tokens, and unmatched tokens

            new_tokens = torch.cat(

                [x_proto, x_reuse, x_unm],

                dim=1

            )


        
            # KV memory

            B, N_new, _ = new_tokens.shape

            head_dim = self._dim // self.heads


            # Compute query on all tokens
            

            q = self.q(new_tokens).reshape(B, N_new, self.heads, head_dim).permute(0, 2, 1, 3)


            # Compute K V only on unmatched tokens

            k_unm = self.k(x_unm)

            v_unm = self.v(x_unm)

            k = torch.cat(

                [k_proto, k_reuse, k_unm],

                dim=1

            ).reshape(B, -1, self.heads, head_dim).permute(0, 2, 1, 3)

            v = torch.cat(

                [v_proto, v_reuse, v_unm],

                dim=1

            ).reshape(B, -1, self.heads, head_dim).permute(0, 2, 1, 3)


            
            # Normal attention mechanism with new tokens as queries and all tokens in cache as keys/values.

            skip_1 = new_tokens                 # ← move skip to here, after expansion

            # Cross-attention: q over merged token set, k/v over cache+new
            attn = self.matmul(q / self.scale, k.transpose(-2, -1))   # [B,H,N_q,N_kv]
            attn = attn.softmax(dim=-1)

            x = self.matmul(attn, v)                                   # [B,H,N_q,hd]

            # Apply the post-attention linear transform and add the skip.
            x = x.permute(0, 2, 1, 3).contiguous().reshape(B, N_new, self.heads * head_dim)
            x = self.projection(x)
            x = self.add(self.drop_path(x), skip_1)

            # Apply the token-wise MLP.
            skip_2 = x
            x = self.mlp_layer_norm(x)
            x = self._forward_mlp(x)
            x = self.add(self.drop_path(x), skip_2)


            return x


        else:

            x = self.qkv(x)

            x, ats_indices = self._forward_attention(

                *self._partition_heads(

                    self._partition_windows(x, in_qkv_domain=True)

                )

            )
        
        # Populate cache with new K/V from this block's input for next block's matching
        if self.caching:
            head_dim = x.shape[-1] // self.heads
            k = self.k(skip_1).reshape(B, N, self.heads, head_dim).permute(0, 2, 1, 3)
            v = self.v(skip_1).reshape(B, N, self.heads, head_dim).permute(0, 2, 1, 3)
            self.cache = caching(
                x,
                k,
                v,
                self.cache,
                self.merge_ratio,
            )


        skip_1 = self._gather_ats_skip(skip_1, ats_indices)
        

        # Apply the post-attention linear transform and add the skip.
        x = self.projection(x)
        x = self.add(self.drop_path(x), skip_1)

        # Apply the token-wise MLP.
        skip_2 = x
        x = self.mlp_layer_norm(x)
        x = self._forward_mlp(x)
        x = self.add(self.drop_path(x), skip_2)
        return x

    def reset_self(self):
        self.last_ats_indices = None

    def _forward_attention(self, q, k, v):

        attn = self.matmul(
            q / self.scale,
            k.transpose(-2,-1)
        )

        if self.relative_position is not None:
            attn = self.relative_position(attn, q)

        attn = attn.softmax(dim=-1)

        attn, ats_indices = self._adaptive_token_sampling(attn, v)

        attn, v, old_dtype = self._cast_matmul_2(attn, v)

        x = self.matmul(attn, v)

        x = self._recombine_heads(x)
        x = self._recombine_windows(x)

        x = self._uncast_matmul_2(x, old_dtype)

        return x, ats_indices
