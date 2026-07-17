# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------
#
# TBKV patch for timm VisionTransformer models (VideoMAE-style per-frame
# ViT evaluation). Ported from the standalone TBKV prototype repo with the
# following fixes:
#  - Cache K/V rows are stored aligned with the cached (background) tokens.
#    The original code stored full-frame K/V but indexed them with positions
#    relative to the background-only token set, reusing K/V of the wrong
#    tokens on a cache hit.
#  - FLOPs are measured with src.core.counting (CountedLinear/CountedMatmul/
#    CountedConv/CountedBias/CountedAdd) instead of the prototype's ad-hoc
#    counter, so numbers are comparable with the other methods in this repo.
#    KV projections saved by cache hits are recorded under
#    "saved_kv_linear_flops" (same convention as src/tbkv/tbkv_blocks.py)
#    and are NOT part of the computed total.
#  - Per-layer debug prints are gated behind a "verbose" flag.

from typing import Optional, Tuple, Dict, Any, Callable, Type

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from src.core.base import ExtendedModule
from src.core.counting import (
    CountedAdd,
    CountedBias,
    CountedConv,
    CountedLinear,
    CountedMatmul,
)

from . import merge
from .cache import Cache
from .tbkv_utils import (
    get_objective_score,
    join_frame,
    split_frame,
    func_warper,
    join_warper,
    split_warper,
    init_generator,
    mps_gather_workaround,
)


def _counted_linear_from(lin: nn.Linear) -> CountedLinear:
    """Build a CountedLinear with the weights of an existing nn.Linear."""
    new = CountedLinear(
        lin.in_features,
        lin.out_features,
        device=lin.weight.device,
        dtype=lin.weight.dtype,
    )
    new.weight.data.copy_(lin.weight.data)
    if lin.bias is not None:
        new.bias.data.copy_(lin.bias.data)
    else:
        new.bias.data.zero_()
    return new


def _split_qkv_linear(module: Attention):
    """
    When we swap the Attention class for ToMeAttention we do not re-run __init__,
    so we need to create the split q/k/v projections manually from the existing
    fused qkv layer. The projections are CountedLinear modules so their FLOPs
    are tracked by the src.core.counting machinery.
    """
    # If they already exist (e.g., patched before), skip
    if hasattr(module, "q") and hasattr(module, "k") and hasattr(module, "v"):
        return

    qkv: nn.Linear = module.qkv
    dim = qkv.in_features
    bias = qkv.bias is not None
    device = qkv.weight.device
    dtype = qkv.weight.dtype

    # Split weights (and bias if present) into q, k, v chunks
    w_q, w_k, w_v = qkv.weight.data.chunk(3, dim=0)
    if bias:
        b_q, b_k, b_v = qkv.bias.data.chunk(3, dim=0)

    def build_linear(w, b=None):
        lin = CountedLinear(dim, dim, device=device, dtype=dtype)
        lin.weight.data.copy_(w)
        if b is not None:
            lin.bias.data.copy_(b)
        else:
            lin.bias.data.zero_()
        return lin

    module.q = build_linear(w_q, b_q if bias else None)
    module.k = build_linear(w_k, b_k if bias else None)
    module.v = build_linear(w_v, b_v if bias else None)

    # Counted output projection and attention matmuls
    if not isinstance(module.proj, CountedLinear):
        module.proj = _counted_linear_from(module.proj)
    if not hasattr(module, "matmul"):
        module.matmul = CountedMatmul()

    # Ensure normalization attrs exist (match defaults in ToMeAttention.__init__)
    if not hasattr(module, "q_norm"):
        module.q_norm = nn.Identity()
    if not hasattr(module, "k_norm"):
        module.k_norm = nn.Identity()
    if not hasattr(module, "norm"):
        module.norm = nn.Identity()


def compute_merge(
    module: torch.nn.Module, x: torch.Tensor, tome_info: Dict[str, Any]
) -> Tuple[Callable, ...]:
    args = tome_info["args"]
    generator = module.generator
    merge_mode = "mean"

    # Frames per video
    fsize = x.shape[0] // args["batch_size"]
    # Tokens per frame
    tsize = x.shape[1]

    # Init Generator
    if args["generator"] is None:
        args["generator"] = init_generator(x.device)
    elif args["generator"].device != x.device:
        args["generator"] = init_generator(x.device, fallback=args["generator"])

    # Local Token Merging!
    local_tokens = join_frame(x, fsize)
    m_ls = [join_warper(fsize)]
    u_ls = [split_warper(fsize)]
    unm = 0
    curF = fsize

    # Recursive merge multi-frame tokens into one set. Such as 4->1 for 4 frames
    # and 8->2->1 for 8 frames when target stride is 4.
    while curF > 1:
        m, u, ret_dict = merge.bipartite_soft_matching_randframe(
            local_tokens, curF, args["local_merge_ratio"], unm, generator,
            args["target_stride"])
        unm += ret_dict["unm_num"]
        m_ls.append(m)
        u_ls.append(u)
        local_tokens = m(local_tokens)

        # Total token number = current frame number * per-frame token number
        # + unmerged token number
        curF = (local_tokens.shape[1] - unm) // tsize

    merged_tokens = local_tokens

    # Global Token Merging!
    if args["merge_global"]:
        if hasattr(module, "global_tokens") and module.global_tokens is not None:
            # Merge local tokens with global tokens. Randomly determine
            # merging destination.
            if torch.rand(1, generator=generator, device=generator.device) > args["global_rand"]:
                src_len = local_tokens.shape[1]
                tokens = torch.cat(
                    [local_tokens, module.global_tokens.to(local_tokens)], dim=1)
                local_chunk = 0
            else:
                src_len = module.global_tokens.shape[1]
                tokens = torch.cat(
                    [module.global_tokens.to(local_tokens), local_tokens], dim=1)
                local_chunk = 1

            m, u, _ = merge.bipartite_soft_matching_2s(
                tokens, src_len, args["global_merge_ratio"],
                merge_mode=merge_mode, unmerge_chunk=local_chunk)
            merged_tokens = m(tokens)
            m_ls.append(m)
            u_ls.append(u)

            # Update global tokens with unmerged local tokens.
            module.global_tokens = u(merged_tokens).detach().clone().cpu()
        else:
            module.global_tokens = local_tokens.detach().clone().cpu()

        m = func_warper(m_ls)
        u = func_warper(u_ls[::-1])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)
        merged_tokens = x

    # Return merge op, unmerge op, and merged tokens.
    return m, u, merged_tokens


def extract_bg_fg_tokens(
    x: torch.Tensor, attn: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract background and foreground tokens based on the attention map.

    Returns (x_bg, x_fg, idx_bg, idx_fg) where the idx tensors give the
    positions of the bg/fg tokens in the flattened token sequence, so callers
    can keep auxiliary tensors (e.g. K/V projections) aligned with them.
    """
    B, N, C = x.shape

    scores = get_objective_score(attn)

    flat_x = x.reshape(1, B * N, C)

    flat_scores = scores.reshape(B * N)

    idx_fg = torch.nonzero(flat_scores == 0, as_tuple=True)[0]

    idx_bg = torch.nonzero(flat_scores != 0, as_tuple=True)[0]

    x_bg = flat_x[:, idx_bg, :]

    x_fg = flat_x[:, idx_fg, :]

    return x_bg, x_fg, idx_bg, idx_fg


class ToMeBlock(Block):
    """
    Modifications:
    - In caching mode, apply token merging after attention and before MLP.
    - In matching mode, use new_tokens returned from attention for residual connection.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block
        # with modifications.
        args = self._tome_info["args"]

        norm_x = self.norm1(x)

        # Get attention - pass both normalized and unnormalized x for token reduction
        x_attn, k, v, new_tokens, attn_map = self.attn(norm_x, self.cache, x_unnorm=x)

        # Handle residual connection
        # If tokens were reduced via matching, use new_tokens as base for residual
        if new_tokens is not None and new_tokens.shape[1] != x.shape[1]:
            # Token reduction happened - new_tokens contains cache + unmatched + fg
            x = self.add(new_tokens, self._drop_path1(x_attn))
        else:
            # Normal case - residual with original x
            x = self.add(x, self._drop_path1(x_attn))

        # Create cache if we are in caching mode
        if args["caching"]:
            # Extract background and foreground tokens based on attention map
            x_bg, x_fg, idx_bg, idx_fg = extract_bg_fg_tokens(x, attn_map)

            # Merge background tokens
            _, _, merged_x_bg = compute_merge(self, x_bg, self._tome_info)

            # Store K/V aligned with the cached tokens: row i of K/V must be
            # the projection of cache token i, otherwise matching reuses the
            # K/V of unrelated tokens.
            if merged_x_bg.shape[1] == idx_bg.numel():
                k_cache = k[:, :, idx_bg, :]
                v_cache = v[:, :, idx_bg, :]
            else:
                # Tokens were merged; there is no exact per-token K/V anymore.
                # Fall back to full-frame K/V (legacy behavior).
                k_cache, v_cache = k, v

            self.cache = Cache(
                k_cache, v_cache, merged_x_bg, layer=self,
                verbose=args.get("verbose", False),
            )
        else:
            # Matching mode (or baseline mode with no cache)
            if new_tokens is not None and new_tokens.shape[1] != x.shape[1]:
                # Tokens were reduced via matching - update x to reduced token set
                x = new_tokens
            # else: No matching happened - keep x as is

            # Store attention map for next layer to use during matching
            self._tome_info["prev_attn_map"] = attn_map

        x = self.add(x, self._drop_path2(self.mlp(self.norm2(x))))
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Return keys and values for caching in TBKV.
     - If there's old_attn, use it to compute saliency maps in matching algorithm.
     - split QKV projection in single linear layers for more flexibility.
    """

    def forward(
        self, x: torch.Tensor, cache: Cache = None, x_unnorm: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Note: this is copied from timm.models.vision_transformer.Attention
        # with modifications.
        # Returns: (x_attn, k, v, new_tokens, attn_map)
        # new_tokens: reduced token set (unnormalized) for residual connection
        # x_unnorm: unnormalized input tokens (for the reduced-token residual)
        B, N, C = x.shape

        args = self._tome_info["args"]
        verbose = args.get("verbose", False)

        # Get old_attn from previous layer (stored in _tome_info during matching)
        # For the first layer or during caching, this will be None
        old_attn = self._tome_info.get("prev_attn_map", None) if not args["caching"] else None

        gather = mps_gather_workaround if x.device.type == "mps" else torch.gather

        new_tokens = None
        attn_map = None

        # Check if we should do matching: cache exists, old_attn exists,
        # and there are background tokens
        if cache is not None and old_attn is not None:
            x_bg, x_fg, idx_bg, idx_fg = extract_bg_fg_tokens(x, old_attn)
            should_match = x_bg.shape[1] > 0

            # Also extract from unnormalized input (for residual connection)
            if should_match and x_unnorm is not None:
                x_bg_unnorm, x_fg_unnorm, _, _ = extract_bg_fg_tokens(x_unnorm, old_attn)
        else:
            should_match = False

        if should_match:
            # Match background tokens against the cache. The cosine-similarity
            # matmul inside is executed via self.matmul so it is counted as
            # matching overhead.
            k_matched, v_matched, matched_cache_tokens, idx_matched, idx_unmatched = \
                cache.match_tokens(x_bg, r_match=args["r_match"], matmul=self.matmul)

            if verbose:
                num_matched = (
                    idx_matched.shape[1] if idx_matched.dim() > 1 else idx_matched.shape[0]
                )
                print(
                    f"  Layer stats: Total={N}, BG={x_bg.shape[1]}, "
                    f"Matched={num_matched}, Cache={cache.tokens.shape[1]}"
                )

            # Get unmatched background tokens
            unm_x_bg = gather(
                x_bg, dim=1, index=idx_unmatched.unsqueeze(-1).expand(-1, -1, C)
            )

            # Tokens for forward pass: cache tokens + unmatched bg + fg
            tokens_forward = torch.cat([matched_cache_tokens, unm_x_bg, x_fg], dim=1)

            # For residual connection: extract unnormalized tokens and combine with cache
            if x_unnorm is not None:
                unm_x_bg_unnorm = gather(
                    x_bg_unnorm, dim=1,
                    index=idx_unmatched.unsqueeze(-1).expand(-1, -1, C),
                )
                new_tokens = torch.cat(
                    [matched_cache_tokens, unm_x_bg_unnorm, x_fg_unnorm], dim=1
                )
            else:
                # Fallback: use normalized tokens
                new_tokens = tokens_forward

            # Compute Q on ALL tokens (cache + unmatched + fg)
            q = self.q(tokens_forward).reshape(
                1, -1, self.num_heads, C // self.num_heads
            ).permute(0, 2, 1, 3)

            # Compute K and V only for unmatched + fg tokens (not for matched cache tokens)
            tokens_for_kv = torch.cat([unm_x_bg, x_fg], dim=1)
            k_unm = self.k(tokens_for_kv).reshape(
                1, -1, self.num_heads, C // self.num_heads
            ).permute(0, 2, 1, 3)
            v_unm = self.v(tokens_for_kv).reshape(
                1, -1, self.num_heads, C // self.num_heads
            ).permute(0, 2, 1, 3)

            # Concatenate matched K/V from cache with newly computed K/V
            k = torch.cat([k_matched, k_unm], dim=2)
            v = torch.cat([v_matched, v_unm], dim=2)

            # Record the K/V projection FLOPs avoided by reusing the cache:
            # one K and one V linear for every matched bg token (before
            # cache-index deduplication). Kept out of the computed total,
            # matching the src/tbkv convention.
            if self.matmul.count_mode:
                num_matched_bg = (
                    idx_matched.shape[1] if idx_matched.dim() > 1 else idx_matched.shape[0]
                )
                self.matmul.counts["saved_kv_linear_flops"] += (
                    2 * B * num_matched_bg * C * C
                )

        if not should_match:
            # Normal Attention - compute Q, K, V from input
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Compute attention (both paths need this)
        attn = self.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_map = attn.clone()  # Store for cache/saliency computation
        attn = self.attn_drop(attn)

        N_q = q.shape[2]
        x = self.matmul(attn, v).transpose(1, 2).reshape(B, N_q, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, k, v, new_tokens, attn_map


def hook_tome_model(model: torch.nn.Module):
    """Adds a forward pre hook to get the image size. Removable with remove_patch."""

    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        # Reset prev_attn_map at the start of each forward pass
        module._tome_info["prev_attn_map"] = None
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def hook_tome_module(module: torch.nn.Module):
    """Adds a forward pre hook to initialize the random number generator.
    All modules share the same generator state to keep their randomness
    consistent in one pass. Removable with remove_patch."""

    def hook(module, args):
        if not hasattr(module, "generator"):
            module.generator = init_generator(args[0].device)
        elif module.generator.device != args[0].device:
            module.generator = init_generator(args[0].device, fallback=module.generator)
        else:
            return None
        return None

    module._tome_info["hooks"].append(module.register_forward_pre_hook(hook))


def make_tome_class(transformer_class, caching: bool = False) -> Type[VisionTransformer]:
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - src.core.counting integration (counting/clear_counts/total_counts),
          mirroring the ExtendedModule API used across this repo.
        """

        def extended_modules(self):
            return [m for m in self.modules() if isinstance(m, ExtendedModule)]

        def counting(self, mode=True):
            for module in self.extended_modules():
                module.count_mode = mode

        def no_counting(self):
            self.counting(mode=False)

        def clear_counts(self):
            for module in self.extended_modules():
                module.counts.clear()

        def total_counts(self):
            return sum(m.counts for m in self.extended_modules())

    return ToMeVisionTransformer


def reset_caches(model: torch.nn.Module):
    """Clear every block's TBKV cache (call between videos)."""
    for module in model.modules():
        if isinstance(module, ToMeBlock):
            module.cache = None
    if hasattr(model, "_tome_info"):
        model._tome_info["prev_attn_map"] = None


def _instrument_counting(model: VisionTransformer):
    """Swap the remaining uncounted compute-heavy modules (patch embedding,
    MLP linears, classification head) for src.core.counting equivalents."""
    # Patch embedding conv (+ bias)
    proj = getattr(getattr(model, "patch_embed", None), "proj", None)
    if isinstance(proj, nn.Conv2d):
        conv = CountedConv(
            spatial_dims=2,
            in_channels=proj.in_channels,
            out_channels=proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            dilation=proj.dilation,
            groups=proj.groups,
            device=proj.weight.device,
            dtype=proj.weight.dtype,
        )
        conv.weight.data.copy_(proj.weight.data)
        new_proj = [conv]
        if proj.bias is not None:
            bias = CountedBias(
                proj.out_channels,
                spatial_dims=2,
                device=proj.bias.device,
                dtype=proj.bias.dtype,
            )
            bias.bias.data.copy_(proj.bias.data)
            new_proj.append(bias)
        model.patch_embed.proj = nn.Sequential(*new_proj)

    # MLP linears in each block
    for module in model.modules():
        if isinstance(module, Block):
            for name in ("fc1", "fc2"):
                lin = getattr(module.mlp, name, None)
                if isinstance(lin, nn.Linear) and not isinstance(lin, CountedLinear):
                    setattr(module.mlp, name, _counted_linear_from(lin))
            if not hasattr(module, "add"):
                module.add = CountedAdd()

    # Classification head
    if isinstance(model.head, nn.Linear) and not isinstance(model.head, CountedLinear):
        model.head = _counted_linear_from(model.head)


def apply_patch(
    model: VisionTransformer,
    local_merge_ratio: float = 0.9,
    merge_global: bool = False,
    global_merge_ratio=0.8,
    seed: int = 123,
    batch_size: int = 2,
    target_stride: int = 4,
    global_rand=0.5,
    caching: bool = False,
    target_block: int = 2,
    r_match: float = 0.3,
    reset_cache: bool = False,
    verbose: bool = False,
):
    """
    Apply the TBKV patch to a timm VisionTransformer model in place.

    Important Args:
     - model: The timm VisionTransformer to patch in place.
     - local_merge_ratio: The ratio of tokens to merge locally.
     - merge_global: Whether or not to include global token merging.
     - global_merge_ratio: The ratio of tokens to merge globally.
     - caching: Start in caching mode (build per-block caches) instead of
       matching mode. Can be toggled later via model._tome_info["args"]["caching"].
     - r_match: Fraction of background tokens matched against the cache.
     - reset_cache: Clear any existing per-block caches.
     - verbose: Print per-layer matching statistics.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__, caching=caching)

    model.__class__ = ToMeVisionTransformer

    model._tome_info = {
        "hooks": [],
        "size": None,
        "args": {
            "seed": seed,
            "merge_global": merge_global,
            "global_merge_ratio": global_merge_ratio,
            "local_merge_ratio": local_merge_ratio,
            "global_rand": global_rand,
            "target_stride": target_stride,
            "batch_size": batch_size,
            "generator": None,
            "caching": caching,
            "target_block": target_block,
            "r_match": r_match,
            "verbose": verbose,
        },
        "class_token": model.cls_token is not None,
        "prev_attn_map": None,  # For passing attention between layers during matching
    }
    hook_tome_model(model)

    for idx, module in enumerate(model.modules()):
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
            module.layer = idx
            if reset_cache or not hasattr(module, "cache"):
                module.cache = None
            hook_tome_module(module)
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
            module._tome_info = model._tome_info
            _split_qkv_linear(module)
            hook_tome_module(module)

    _instrument_counting(model)
