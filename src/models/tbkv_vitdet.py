import torch
import torch.nn as nn
from detectron2.config import LazyConfig, instantiate
from detectron2.structures import ImageList
from torchvision.transforms import Normalize

from src.tbkv.tbkv_backbone import TBKVViTBackbone
from src.tbkv.tbkv_blocks import TBKVBlock, merging as block_merging
from src.tbkv.cache import Cache
from src.tbkv.tbkv_utils import extract_bg_fg_tokens, perform_kv_reuse_no_reduction, get_objective_score
from src.core.base import ExtendedModule, numeric_tuple
from src.core.blocks import LN_EPS
from src.utils.image import as_float32, pad_to_size


# Resources consulted:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/utils.py
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py


class LinearEmbedding(nn.Module):
    """
    The initial linear patch-embedding layer for ViTDet. Linearly
    transforms each input patch into a token vector.
    """

    def __init__(self, input_channels, dim, patch_size):
        """
        :param input_channels: The number of image channels (e.g., 3 for
        RGB images)
        :param dim: The dimensionality of token vectors
        :param patch_size: The patch size for each token (a 2-element
        tuple/list)
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # (batch, dim, height, width)

        x = self.conv(x)
        # (batch, dim, height, width)

        # Flatten the spatial axes.
        x = x.flatten(start_dim=-2)
        # (batch, dim, patch)

        x = x.transpose(1, 2)
        # (batch, patch, dim)

        return x


class PointwiseLayerNorm2d(nn.LayerNorm):
    """
    A LayerNorm operation which performs x.permute(0, 2, 3, 1) before
    applying the normalization. The permutation is inverted after
    normalization.
    """

    def forward(self, x):
        # (batch, dim, height, width)

        x = x.permute(0, 2, 3, 1)
        # (batch, height, width, dim)

        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        # (batch, dim, height, width)

        return x


class SimplePyramid(nn.Module):
    """
    The ViTDet feature pyramid (precedes the object detection head).
    """

    def __init__(self, scale_factors, dim, out_channels):
        """
        :param scale_factors: A list of spatial scale factors
        :param dim: The dimensionality of token vectors in the
        Transformer backbone
        :param out_channels: The number of output channels (the number
        of channels expected by the object detection head)
        """
        super().__init__()
        self.stages = nn.ModuleList(
            self._build_scale(scale, dim, out_channels) for scale in scale_factors
        )
        self.max_pool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        x = [stage(x) for stage in self.stages]
        x.append(self.max_pool(x[-1]))
        return x

    @staticmethod
    def _build_scale(scale, dim, out_channels):
        assert scale in [4.0, 2.0, 1.0, 0.5]
        if scale == 0.5:
            mid_dim = dim
            start_layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif scale == 1.0:
            mid_dim = dim
            start_layers = []
        elif scale == 2.0:
            mid_dim = dim // 2
            start_layers = [nn.ConvTranspose2d(dim, mid_dim, kernel_size=2, stride=2)]
        else:  # scale == 4.0
            mid_dim = dim // 4
            start_layers = [
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                PointwiseLayerNorm2d(dim // 2, eps=LN_EPS),
                nn.GELU(),
                nn.ConvTranspose2d(dim // 2, mid_dim, kernel_size=2, stride=2),
            ]
        common_layers = [
            nn.Conv2d(mid_dim, out_channels, kernel_size=1, bias=False),
            PointwiseLayerNorm2d(out_channels, eps=LN_EPS),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            PointwiseLayerNorm2d(out_channels, eps=LN_EPS),
        ]
        return nn.Sequential(*start_layers, *common_layers)


class TBKVViTDet(ExtendedModule):
    """
    The ViTDet object detection Transformer model. See
    configs/models/vitdet_b_coco.yml for an example configuration.
    """

    def __init__(
        self,
        backbone_config,
        classes,
        detectron2_config,
        input_shape,
        normalize_mean,
        normalize_std,
        output_channels,
        patch_size,
        scale_factors,
    ):
        """
        :param backbone_config: A dict containing kwargs for the
        backbone constructor
        :param classes: The number of object classes
        :param detectron2_config: Path of a Python file containing a
        Detectron2 config for the detection head
        :param input_shape: The (c, h, w) shape for inputs (the
        preprocessing spatially pads inputs to this shape)
        :param normalize_mean: The mean to use with
        torchvision.transforms.Normalize
        :param normalize_std: The standard deviation to use with
        torchvision.transforms.Normalize
        :param output_channels: The number of channels expected by the
        object detection head
        :param patch_size: The patch size for each token (a 2-element
        tuple/list)
        :param scale_factors: Scale factors for the SimplePyramid module
        """
        super().__init__()
        input_c, input_h, input_w = input_shape
        patch_size = numeric_tuple(patch_size, length=2)
        self.backbone_input_size = (input_h // patch_size[0], input_w // patch_size[1])

        # Set up submodules.
        self.preprocessing = ViTDetPreprocessing(
            input_shape, normalize_mean, normalize_std
        )
        dim = backbone_config["block_config"]["dim"]
        self.embedding = LinearEmbedding(input_c, dim, patch_size)
        self.backbone = TBKVViTBackbone(
            input_size=self.backbone_input_size,
            **backbone_config,
        )
        self.pyramid = SimplePyramid(scale_factors, dim, output_channels)
        detectron2_config = LazyConfig.load(detectron2_config)["model"]
        self.proposal_generator = instantiate(detectron2_config["proposal_generator"])
        roi_heads_config = detectron2_config["roi_heads"]
        roi_heads_config["num_classes"] = classes
        self.roi_heads = instantiate(roi_heads_config)

    def forward(self, x):
        images, x = self.pre_backbone(x)
        x = self._forward_backbone_no_reset(x)
        results = self.post_backbone(images, x)
        return results

    def set_mode(self, mode):
        caching = (mode == "caching")
        for module in self.backbone.modules():
            if isinstance(module, TBKVBlock):
                # Global blocks (window_size is None) always build caches.
                # Windowed blocks build a per-position keyframe cache only in
                # the TBKV-on-all-blocks mode; otherwise they run full compute.
                if module.window_size is None or getattr(module, "tbkv_all_blocks", False):
                    module.caching = caching
                else:
                    module.caching = False

    def _qkv_from_norm(self, block, x_norm):
        b, n, c = x_norm.shape
        head_dim = c // block.heads
        q = block.q(x_norm).reshape(b, n, block.heads, head_dim).permute(0, 2, 1, 3)
        k = block.k(x_norm).reshape(b, n, block.heads, head_dim).permute(0, 2, 1, 3)
        v = block.v(x_norm).reshape(b, n, block.heads, head_dim).permute(0, 2, 1, 3)
        return q, k, v

    def _forward_tbkv_block_vitdet(self, block, x, prev_attn, block_index):
        """
        TBKV forward for global-attention blocks only (window_size is None).
        Windowed blocks are handled by block.forward(x, None) in _forward_backbone_no_reset.
        """
        skip_1 = x
        x_norm = block.input_layer_norm(x)
        b, n, c = x_norm.shape
        head_dim = c // block.heads

        cached_attn = None  # reused at the end to avoid double compute

        if block.caching or block.cache is None:
            # ── True caching pass: build merged K/V cache ──────────────────────
            q, k, v = self._qkv_from_norm(block, x_norm)
            cached_attn = block.matmul(q / block.scale, k.transpose(-2, -1))
            if block.relative_position is not None:
                cached_attn = block.relative_position(cached_attn, q)
            cached_attn = cached_attn.softmax(dim=-1)

            if block.raw:
                block.cache = Cache(k.detach(), v.detach(), x_norm.detach())
                n_cache_tok = n
            elif block.token_skip:
                # Token-skip mode needs the cache aligned 1:1 with the full N
                # grid (no merging) so a matched token can look up its cached
                # post-block output by cache index.
                block.cache = Cache(k.detach(), v.detach(), x_norm.detach())
                n_cache_tok = n
            else:
                k_new, v_new, tok_new, n_bg_c, n_fg_c = block_merging(
                    x=x_norm,
                    attn_map=cached_attn,
                    k=k,
                    v=v,
                    merging_iterations=block.merging_iterations,
                    local_merge_ratio=block.local_merge_ratio,
                    split_tokens=block.split_tokens,
                    bg_ratio=block.bg_ratio,
                )
                block.cache = Cache(k_new.detach(), v_new.detach(), tok_new.detach())
                n_cache_tok = int(tok_new.shape[1])
            saliency = get_objective_score(cached_attn).squeeze(-1).detach()  # [B, N]
            block.cache.set_old_attn(saliency)
            block._record_frame_stats({
                'phase': 'caching',
                'n_total': n,
                'cache_size': n_cache_tok,
            })

        else:
            # ── Matching pass ──────────────────────────────────────────────────
            eff_prev_attn = prev_attn
            if eff_prev_attn is None and block.cache.old_attn is not None:
                eff_prev_attn = block.cache.old_attn.to(x.device)

            # ── Decoupled mode: TBKV does KV-reuse ONLY, and the secondary
            # policy does the token skipping. matched tokens reuse cached K/V
            # but are NOT output-reused; only secondary-deferred tokens reuse
            # the previous-frame output. Removes the whole-block output-reuse
            # accuracy floor and lets the two mechanisms exploit different
            # redundancies (KV reuse = per-token temporal; token skip =
            # spatial/temporal recompute budget). ──────────────────────────
            secondary_skip = (
                block.kv_reuse_only
                and getattr(block, "secondary_policy", None) is not None
                and block.secondary_policy.active
                and block.cache is not None
                and block.cache.tokens is not None
                and eff_prev_attn is not None
                and block_index >= block.matching_start_block
            )
            if secondary_skip:
                return self._forward_kvreuse_skip(block, x, x_norm, skip_1)

            if block_index < block.matching_start_block or eff_prev_attn is None:
                # Full recompute — no KV reuse.
                q, k, v = self._qkv_from_norm(block, x_norm)
                block._record_frame_stats({
                    'phase': 'matching',
                    'n_total': n, 'n_bg': n, 'n_fg': 0,
                    'n_bg_matched': 0, 'n_bg_unmatched': n,
                })

            elif block.kv_reuse_only:
                # KV reuse only — no bg/fg split.
                q, k, v, info = perform_kv_reuse_no_reduction(
                    x=x_norm,
                    cache_tokens=block.cache.tokens,
                    cache_k=block.cache.K,
                    cache_v=block.cache.V,
                    q_proj=block.q,
                    k_proj=block.k,
                    v_proj=block.v,
                    num_heads=block.heads,
                    r_match=block.r_match,
                )
                block._record_frame_stats({
                    'phase': 'matching',
                    'n_total': n, 'n_bg': n, 'n_fg': 0,
                    'n_bg_matched': info['n_matched_tokens'],
                    'n_bg_unmatched': info['n_unmatched_tokens'],
                })

            else:
                # Full TBKV: bg/fg split, match bg tokens against cache.
                cache_tokens = block.cache.tokens
                x_bg, x_fg, idx_bg, idx_fg = extract_bg_fg_tokens(x_norm, eff_prev_attn, bg_ratio=block.bg_ratio)

                x_bg_n = x_bg / (x_bg.norm(dim=-1, keepdim=True) + 1e-6)
                c_n = cache_tokens / (cache_tokens.norm(dim=-1, keepdim=True) + 1e-6)
                sim_bg = torch.matmul(x_bg_n, c_n.transpose(-2, -1))

                best_sim_bg, best_idx_bg = sim_bg.max(dim=-1)
                n_bg = x_bg.shape[1]
                n_fg = idx_fg.shape[1]
                n_match = max(0, min(n_bg, int(n_bg * float(block.r_match))))
                sort_bg = best_sim_bg.argsort(dim=-1, descending=True)
                idx_bg_match = sort_bg[:, :n_match]
                idx_bg_unmatch = sort_bg[:, n_match:]

                # ── Full-block token skip (item A of the adjustment plan) ──────
                # A matched background token bypasses the ENTIRE block (Q, attn,
                # projection AND MLP) and reuses its cached post-block output.
                # Only unmatched-bg + foreground tokens are processed. This is
                # what moves TBKV onto the accuracy/efficiency Pareto frontier.
                if (block.token_skip and n_match > 0
                        and block.cache.outputs is not None):
                    return self._forward_tbkv_skip(
                        block, x, x_norm, skip_1, idx_bg, idx_fg,
                        idx_bg_match, idx_bg_unmatch, best_idx_bg,
                        n_match, n_bg, n_fg,
                    )

                # Compute Q for all tokens (only Q — K/V computed per-subset below).
                q = block.q(x_norm).reshape(b, n, block.heads, head_dim).permute(0, 2, 1, 3)
                k = torch.zeros((b, block.heads, n, head_dim), device=x.device, dtype=x.dtype)
                v = torch.zeros((b, block.heads, n, head_dim), device=x.device, dtype=x.dtype)

                if n_match > 0:
                    matched_cache_idx = torch.gather(best_idx_bg, 1, idx_bg_match)
                    matched_orig_idx = torch.gather(idx_bg, 1, idx_bg_match)
                    gather_cache = matched_cache_idx.unsqueeze(1).unsqueeze(-1).expand(
                        -1, block.heads, -1, head_dim
                    )
                    k_bg_match = torch.gather(block.cache.K, 2, gather_cache)
                    v_bg_match = torch.gather(block.cache.V, 2, gather_cache)
                    scatter_bg_match = matched_orig_idx.unsqueeze(1).unsqueeze(-1).expand(
                        -1, block.heads, -1, head_dim
                    )
                    k = k.scatter(2, scatter_bg_match, k_bg_match)
                    v = v.scatter(2, scatter_bg_match, v_bg_match)

                n_unmatch = idx_bg_unmatch.shape[1]
                if n_unmatch > 0:
                    unmatched_orig_idx = torch.gather(idx_bg, 1, idx_bg_unmatch)
                    gather_unmatch = unmatched_orig_idx.unsqueeze(-1).expand(-1, -1, c)
                    x_unmatch = x_norm.gather(1, gather_unmatch)
                    # Only K and V needed for unmatched tokens (Q already computed above).
                    k_unmatch = block.k(x_unmatch).reshape(b, n_unmatch, block.heads, head_dim).permute(0, 2, 1, 3)
                    v_unmatch = block.v(x_unmatch).reshape(b, n_unmatch, block.heads, head_dim).permute(0, 2, 1, 3)
                    scatter_unmatch = unmatched_orig_idx.unsqueeze(1).unsqueeze(-1).expand(
                        -1, block.heads, -1, head_dim
                    )
                    k = k.scatter(2, scatter_unmatch, k_unmatch)
                    v = v.scatter(2, scatter_unmatch, v_unmatch)

                if n_fg > 0:
                    gather_fg = idx_fg.unsqueeze(-1).expand(-1, -1, c)
                    x_fg_full = x_norm.gather(1, gather_fg)
                    # Only K and V needed for fg tokens (Q already computed above).
                    k_fg = block.k(x_fg_full).reshape(b, n_fg, block.heads, head_dim).permute(0, 2, 1, 3)
                    v_fg = block.v(x_fg_full).reshape(b, n_fg, block.heads, head_dim).permute(0, 2, 1, 3)
                    scatter_fg = idx_fg.unsqueeze(1).unsqueeze(-1).expand(
                        -1, block.heads, -1, head_dim
                    )
                    k = k.scatter(2, scatter_fg, k_fg)
                    v = v.scatter(2, scatter_fg, v_fg)

                block._record_frame_stats({
                    'phase': 'matching',
                    'n_total': n, 'n_bg': n_bg, 'n_fg': n_fg,
                    'n_bg_matched': n_match,
                    'n_bg_unmatched': n_bg - n_match,
                })

        # ── Compute attention output ────────────────────────────────────────────
        # Reuse cached_attn from the caching pass to avoid redundant computation.
        if cached_attn is not None:
            attn = cached_attn
        else:
            attn = block.matmul(q / block.scale, k.transpose(-2, -1))
            if block.relative_position is not None:
                attn = block.relative_position(attn, q)
            attn = attn.softmax(dim=-1)

        x = block.matmul(attn, v)
        x = block._recombine_heads(x)

        x = block.projection(x)
        x = block.add(block.drop_path(x), skip_1)

        skip_2 = x
        x = block.mlp_layer_norm(x)
        x = block._forward_mlp(x)
        x = block.add(block.drop_path(x), skip_2)

        # In token-skip mode, cache the post-block output of the keyframe so
        # that matched tokens in subsequent frames can reuse the whole block
        # output (not just the cached K/V).  cached_attn is only set on the
        # caching pass, so this stores exactly the keyframe output.
        if block.token_skip and cached_attn is not None and block.cache is not None:
            block.cache.outputs = x.detach()

        # Return compact saliency [B, N] so global_prev_attn in
        # _forward_backbone_no_reset stays tiny (~7 KB vs ~150 MB for full attn).
        saliency_out = get_objective_score(attn).squeeze(-1).detach()  # [B, N]
        return x, saliency_out

    def _forward_tbkv_skip(self, block, x, x_norm, skip_1, idx_bg, idx_fg,
                           idx_bg_match, idx_bg_unmatch, best_idx_bg,
                           n_match, n_bg, n_fg):
        """
        Full-block token-skip forward for a global-attention block, with an
        optional *secondary* policy (Eventful / MaskVD / STGT) stacked on top.

        TBKV decides which background tokens are stable enough to reuse their
        cached post-block output (``matched``).  The remaining *active* tokens
        (unmatched background + foreground) are then handed to the secondary
        policy, which splits them into:

          * ``keep``  — fully recomputed this frame (Q/K/V, attention, MLP);
          * ``defer`` — reuse the previous matching frame's block output
                        (``cache.prev_output``) and are dropped from the key set.

        So two reuse mechanisms compose: TBKV's cross-frame K/V matching plus the
        secondary method's intra-frame token gating.  All projections/MLP run on
        the ``keep`` subset only, so the CountedLinear counters record the extra
        savings automatically.
        """
        b, n, c = x_norm.shape
        head_dim = c // block.heads

        matched_cache_idx = torch.gather(best_idx_bg, 1, idx_bg_match)  # cache positions
        matched_orig_idx = torch.gather(idx_bg, 1, idx_bg_match)        # current positions
        unmatched_orig_idx = torch.gather(idx_bg, 1, idx_bg_unmatch)

        # Active token positions = unmatched background + foreground.
        active_idx = torch.cat([unmatched_orig_idx, idx_fg], dim=1)     # [b, n_active]

        # ── Secondary policy: split active tokens into keep / defer ────────────
        policy = getattr(block, "secondary_policy", None)
        can_defer = (
            policy is not None and policy.active
            and block.cache.prev_output is not None
            and active_idx.shape[1] > 1
        )
        if can_defer:
            gather_active = active_idx.unsqueeze(-1).expand(-1, -1, c)
            x_active_all = x_norm.gather(1, gather_active)
            if block.cache.old_attn is not None:
                sal_active = block.cache.old_attn.to(x.device).gather(1, active_idx)
            else:
                sal_active = torch.zeros(active_idx.shape, device=x.device)
            keep_local, defer_local = policy.select(
                id(block), x_active_all, sal_active,
                active_idx=active_idx, n_grid=n,
            )
            keep_idx = torch.gather(active_idx, 1, keep_local)
            defer_idx = torch.gather(active_idx, 1, defer_local)
        else:
            keep_idx = active_idx
            defer_idx = active_idx[:, :0]

        n_keep = keep_idx.shape[1]
        n_defer = defer_idx.shape[1]

        gather_keep = keep_idx.unsqueeze(-1).expand(-1, -1, c)
        x_keep = x_norm.gather(1, gather_keep)                          # [b, n_keep, c]
        skip_keep = skip_1.gather(1, gather_keep)                       # residual for keep

        # ── Compact key/value set = matched cached K/V ++ keep fresh K/V ───────
        gather_cache = matched_cache_idx.unsqueeze(1).unsqueeze(-1).expand(
            -1, block.heads, -1, head_dim
        )
        k_match = torch.gather(block.cache.K, 2, gather_cache)          # [b, h, n_match, hd]
        v_match = torch.gather(block.cache.V, 2, gather_cache)

        k_keep = block.k(x_keep).reshape(b, n_keep, block.heads, head_dim).permute(0, 2, 1, 3)
        v_keep = block.v(x_keep).reshape(b, n_keep, block.heads, head_dim).permute(0, 2, 1, 3)
        q_keep = block.q(x_keep).reshape(b, n_keep, block.heads, head_dim).permute(0, 2, 1, 3)

        k_cat = torch.cat([k_match, k_keep], dim=2)                     # [b, h, n_match+n_keep, hd]
        v_cat = torch.cat([v_match, v_keep], dim=2)

        attn = block.matmul(q_keep / block.scale, k_cat.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out_keep = block.matmul(attn, v_cat)                           # [b, h, n_keep, hd]
        out_keep = block._recombine_heads(out_keep)                    # [b, n_keep, c]
        out_keep = block.projection(out_keep)
        out_keep = block.add(block.drop_path(out_keep), skip_keep)

        skip2_keep = out_keep
        out_keep = block.mlp_layer_norm(out_keep)
        out_keep = block._forward_mlp(out_keep)
        out_keep = block.add(block.drop_path(out_keep), skip2_keep)    # [b, n_keep, c]

        # Matched tokens: reuse cached keyframe post-block output.
        gather_out = matched_cache_idx.unsqueeze(-1).expand(-1, -1, c)
        out_match = block.cache.outputs.gather(1, gather_out)          # [b, n_match, c]

        # Scatter keep + matched onto the full token grid.
        output = torch.zeros((b, n, c), device=x.device, dtype=x.dtype)
        output = output.scatter(1, keep_idx.unsqueeze(-1).expand(-1, -1, c), out_keep)
        output = output.scatter(1, matched_orig_idx.unsqueeze(-1).expand(-1, -1, c), out_match)

        # Deferred active tokens: reuse the previous matching-frame output.
        if n_defer > 0:
            gather_defer = defer_idx.unsqueeze(-1).expand(-1, -1, c)
            out_defer = block.cache.prev_output.gather(1, gather_defer)
            output = output.scatter(1, gather_defer, out_defer)

        # Update temporal state for the next frame.
        block.cache.prev_output = output.detach()
        if policy is not None and getattr(policy, "name", "none") == "eventful":
            policy.update(id(block), x_norm)

        # Saliency for the next block: reuse the cached keyframe saliency
        # (background/foreground assignment is stable — that is the premise of
        # token skipping) so downstream bg/fg splits stay well-defined.
        if block.cache.old_attn is not None:
            saliency_out = block.cache.old_attn.to(x.device).detach()
        else:
            saliency_out = get_objective_score(attn).squeeze(-1).detach()

        block._record_frame_stats({
            'phase': 'matching',
            'n_total': n, 'n_bg': n_bg, 'n_fg': n_fg,
            'n_bg_matched': n_match,
            'n_bg_unmatched': n_bg - n_match,
            'token_skip': True,
            'n_active': int(active_idx.shape[1]),
            'n_keep': int(n_keep),
            'n_defer': int(n_defer),
        })
        return output, saliency_out

    def _forward_kvreuse_skip(self, block, x, x_norm, skip_1):
        """
        Decoupled combo: TBKV does *KV-reuse only* and the SECONDARY policy does
        the token skipping.

        Difference from :meth:`_forward_tbkv_skip`:
          * matched tokens reuse cached **K/V** (skipping their K/V projection),
            but they are **NOT** output-reused — every token still runs Q +
            attention (over the FULL key set) + projection + MLP unless the
            *secondary* policy defers it.  This removes the whole-block
            output-reuse approximation that pins TBKV's accuracy floor.
          * only secondary-*deferred* tokens reuse ``cache.prev_output``.

        The two mechanisms therefore target different redundancies: TBKV reuses
        per-token K/V across frames (cheap attention keys), while the secondary
        policy decides which tokens are recomputed this frame.  All heavy
        projections/MLP run on the ``keep`` subset only, so the CountedLinear
        counters record the savings automatically.
        """
        b, n, c = x_norm.shape
        head_dim = c // block.heads
        policy = block.secondary_policy

        # ── Secondary policy: pick keep (recompute) vs defer (reuse prev) ──────
        active_idx = torch.arange(n, device=x.device).unsqueeze(0).expand(b, -1)
        can_defer = policy.active and block.cache.prev_output is not None
        if can_defer:
            if block.cache.old_attn is not None:
                saliency = block.cache.old_attn.to(x.device)
            else:
                saliency = torch.zeros((b, n), device=x.device)
            keep_idx, defer_idx = policy.select(
                id(block), x_norm, saliency, active_idx=active_idx, n_grid=n,
            )
        else:
            keep_idx = active_idx
            defer_idx = active_idx[:, :0]
        n_keep = keep_idx.shape[1]
        n_defer = defer_idx.shape[1]

        # ── TBKV KV-reuse: full-grid K/V, matched reuse cache, unmatched fresh ─
        cache_tokens = block.cache.tokens
        x_nrm = x_norm / (x_norm.norm(dim=-1, keepdim=True) + 1e-6)
        c_nrm = cache_tokens / (cache_tokens.norm(dim=-1, keepdim=True) + 1e-6)
        sim = torch.matmul(x_nrm, c_nrm.transpose(-2, -1))
        best_sim, best_cache_idx = sim.max(dim=-1)
        n_match = max(0, min(n, int(n * float(block.r_match))))
        sorted_idx = best_sim.argsort(dim=-1, descending=True)
        idx_match = sorted_idx[:, :n_match]
        idx_unmatch = sorted_idx[:, n_match:]

        k = torch.zeros((b, block.heads, n, head_dim), device=x.device, dtype=x.dtype)
        v = torch.zeros((b, block.heads, n, head_dim), device=x.device, dtype=x.dtype)
        if n_match > 0:
            matched_cache_idx = torch.gather(best_cache_idx, 1, idx_match)
            gc = matched_cache_idx.unsqueeze(1).unsqueeze(-1).expand(-1, block.heads, -1, head_dim)
            sm = idx_match.unsqueeze(1).unsqueeze(-1).expand(-1, block.heads, -1, head_dim)
            k = k.scatter(2, sm, torch.gather(block.cache.K, 2, gc))
            v = v.scatter(2, sm, torch.gather(block.cache.V, 2, gc))
        n_unmatch = idx_unmatch.shape[1]
        if n_unmatch > 0:
            x_um = x_norm.gather(1, idx_unmatch.unsqueeze(-1).expand(-1, -1, c))
            k_um = block.k(x_um).reshape(b, n_unmatch, block.heads, head_dim).permute(0, 2, 1, 3)
            v_um = block.v(x_um).reshape(b, n_unmatch, block.heads, head_dim).permute(0, 2, 1, 3)
            su = idx_unmatch.unsqueeze(1).unsqueeze(-1).expand(-1, block.heads, -1, head_dim)
            k = k.scatter(2, su, k_um)
            v = v.scatter(2, su, v_um)

        # ── Attention + projection + MLP for KEEP tokens only (full key set) ───
        gather_keep = keep_idx.unsqueeze(-1).expand(-1, -1, c)
        x_keep = x_norm.gather(1, gather_keep)
        skip_keep = skip_1.gather(1, gather_keep)
        q_keep = block.q(x_keep).reshape(b, n_keep, block.heads, head_dim).permute(0, 2, 1, 3)
        attn = block.matmul(q_keep / block.scale, k.transpose(-2, -1)).softmax(dim=-1)
        out_keep = block.matmul(attn, v)
        out_keep = block._recombine_heads(out_keep)
        out_keep = block.projection(out_keep)
        out_keep = block.add(block.drop_path(out_keep), skip_keep)

        skip2_keep = out_keep
        out_keep = block.mlp_layer_norm(out_keep)
        out_keep = block._forward_mlp(out_keep)
        out_keep = block.add(block.drop_path(out_keep), skip2_keep)

        # ── Assemble full output grid ──────────────────────────────────────────
        output = torch.zeros((b, n, c), device=x.device, dtype=x.dtype)
        output = output.scatter(1, gather_keep, out_keep)
        if n_defer > 0:
            gd = defer_idx.unsqueeze(-1).expand(-1, -1, c)
            output = output.scatter(1, gd, block.cache.prev_output.gather(1, gd))
        block.cache.prev_output = output.detach()
        if getattr(policy, "name", "none") == "eventful":
            policy.update(id(block), x_norm)

        if block.cache.old_attn is not None:
            saliency_out = block.cache.old_attn.to(x.device).detach()
        else:
            saliency_out = get_objective_score(attn).squeeze(-1).detach()

        block._record_frame_stats({
            'phase': 'matching',
            'n_total': n, 'n_bg': n, 'n_fg': 0,
            'n_bg_matched': n_match, 'n_bg_unmatched': n_unmatch,
            'kvreuse_skip': True,
            'n_keep': int(n_keep), 'n_defer': int(n_defer),
        })
        return output, saliency_out

    def _forward_backbone_no_reset(self, x):
        # ViTDet detection requires a fixed token grid across all blocks.
        #
        # Windowed blocks (8 of 12 in ViTDet-B) use TBKVBlock.forward(x, None)
        # which correctly handles window partitioning, relative position
        # embeddings, and caching.  Passing None prevents token reduction.
        #
        # Global attention blocks (4 of 12) use _forward_tbkv_block_vitdet
        # which maintains the full token grid while applying TBKV reuse.
        # Only global-block attention is tracked for bg/fg split since windowed
        # attention maps have incompatible shapes.
        x = self.backbone.position_encoding(x)
        global_prev_attn = None
        for block_index, block in enumerate(self.backbone.blocks):
            if isinstance(block, TBKVBlock) and block.window_size is not None:
                # Windowed block: per-position KV-reuse + MLP-skip when the
                # TBKV-on-all-blocks mode is enabled, else full windowed compute.
                if getattr(block, "tbkv_all_blocks", False):
                    x, _ = block.forward_tbkv_all(x)
                else:
                    x, _ = block.forward(x, None)
            elif isinstance(block, TBKVBlock):
                # Global block: TBKV with fixed token grid.
                x, global_prev_attn = self._forward_tbkv_block_vitdet(
                    block, x, global_prev_attn, block_index
                )
            else:
                x = block(x)
        return x

    def post_backbone(self, images, x):
        """
        Computes the portion of the model after the Transformer
        backbone.
        """
        x = x.transpose(-1, -2)
        x = x.view(x.shape[:-1] + self.backbone_input_size)
        x = self.pyramid(x)

        # Compute region proposals and bounding boxes.
        x = dict(zip(self.proposal_generator.in_features, x))
        proposals = self.proposal_generator(images, x, None)[0]
        result = self.roi_heads(images, x, proposals, None)[0]
        result = [
            {"boxes": y.pred_boxes.tensor, "scores": y.scores, "labels": y.pred_classes}
            for y in result
        ]
        return result

    def pre_backbone(self, x):
        """
        Computes the portion of the model before the Transformer
        backbone.
        """
        x = as_float32(x)  # Range [0, 1]
        x = self.preprocessing(x)
        images = ImageList.from_tensors([x])
        x = self.embedding(x)
        return images, x


class ViTDetPreprocessing(nn.Module):
    """
    Preprocessing for ViTDet. Applies value normalization and square
    padding. Expects inputs scaled to the range [0, 1].
    """

    def __init__(self, input_shape, normalize_mean, normalize_std):
        """
        :param input_shape: The (c, h, w) shape to which inputs should
        be padded
        :param normalize_mean: The mean to use with
        torchvision.transforms.Normalize
        :param normalize_std: The standard deviation to use with
        torchvision.transforms.Normalize
        """
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.normalization = Normalize(normalize_mean, normalize_std)

    def forward(self, x):
        # This normalization assumes x in the range [0, 255], but the
        # parent model (ViTDet) scales the input image to [0, 1].
        x = self.normalization(x * 255.0)

        # This is bottom-right padding, so it won't affect the bounding
        # box coordinates.
        x = pad_to_size(x, self.input_shape[-2:])

        return x
