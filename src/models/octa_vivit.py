"""OCTA factorized ViViT model.

Mirrors :class:`~src.models.tbkv_vivit.TBKVFactorizedViViT` but swaps the
spatial/temporal backbones for :class:`OCTAViTBackbone`.  The tubelet embedding
and preprocessing are reused (imported, not modified) from the ViViT model
module.  Object-centric caching is enabled per sub-model via the block config
(``enable_object_cache``); the temporal sub-model is typically left dense.
"""

import torch
import torch.nn as nn

from src.core.base import ExtendedModule
from src.core.blocks import LN_EPS
from src.core.counting import CountedLinear
from src.core.utils import expand_row_index  # noqa: F401  (kept for parity)
from src.octa.octa_backbone import OCTAViTBackbone
from src.models.tbkv_vivit import TubeletEmbedding, ViViTPreprocessing


class OCTAViViTSubModel(ExtendedModule):
    """A factorized ViViT sub-model (spatial or temporal) backed by OCTA."""

    def __init__(self, input_size, backbone_config):
        super().__init__()
        dim = backbone_config["block_config"]["dim"]
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.backbone = OCTAViTBackbone(
            input_size=input_size, has_class_token=True, **backbone_config
        )
        self.layer_norm = nn.LayerNorm(dim, eps=LN_EPS)

    def forward(self, x):
        expand_shape = (x.shape[0],) + self.class_token.shape[1:]
        x = torch.concat([self.class_token.expand(expand_shape), x], dim=1)
        x = self.backbone(x)
        x = self.layer_norm(x)
        return x[:, 0]


class OCTAFactorizedViViT(ExtendedModule):
    """Spatio-temporal factorized ViViT with object-centric temporal caching."""

    def __init__(
        self,
        classes,
        input_shape,
        normalize_mean,
        normalize_std,
        spatial_config,
        spatial_views,
        temporal_config,
        temporal_stride,
        temporal_views,
        tubelet_shape,
        batch_views=True,
        dropout_rate=0.0,
        spatial_only=False,
        temporal_only=False,
    ):
        super().__init__()
        assert not (spatial_only and temporal_only)
        assert not (dropout_rate < 0.0 or dropout_rate > 1.0)
        input_shape = tuple(input_shape)
        tubelet_shape = tuple(tubelet_shape)
        input_t, input_c, input_h, input_w = input_shape
        backbone_input_size = (input_h // tubelet_shape[1], input_w // tubelet_shape[2])
        self.batch_views = batch_views
        self.spatial_only = spatial_only
        self.temporal_only = temporal_only

        self.preprocessing = ViViTPreprocessing(
            input_shape, normalize_mean, normalize_std,
            spatial_views, temporal_stride, temporal_views,
        )
        dim = spatial_config["block_config"]["dim"]
        self.embedding = TubeletEmbedding(input_c, dim, tubelet_shape)
        self.spatial_model = OCTAViViTSubModel(backbone_input_size, spatial_config)
        temporal_input_size = (input_t // tubelet_shape[0],)
        self.temporal_model = OCTAViViTSubModel(temporal_input_size, temporal_config)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.classifier = CountedLinear(in_features=dim, out_features=classes)

    # -- OCTA cache control (parallels the TBKV interface) ------------------
    def set_mode(self, mode):
        if mode not in {"caching", "matching"}:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'caching' or 'matching'.")
        caching = (mode == "caching")
        for module in self.modules():
            if hasattr(module, "caching") and hasattr(module, "cache"):
                module.caching = caching

    def clear_cache(self):
        for module in self.modules():
            if hasattr(module, "cache"):
                module.cache = None
            if hasattr(module, "prev_attn_map"):
                module.prev_attn_map = None

    # -- forward ------------------------------------------------------------
    def forward(self, x):
        batch_size = x.shape[0]
        if not self.temporal_only:
            x = self._forward_spatial(x)
        if not self.spatial_only:
            x = self._forward_temporal(x, batch_size)
        return x

    def _forward_spatial(self, x):
        x = self.preprocessing(x)
        if self.batch_views:
            x = torch.stack(x, dim=1).flatten(end_dim=1)
            x = self._forward_view(x)
        else:
            x = [self._forward_view(view) for view in x]
            x = torch.stack(x, dim=1).flatten(end_dim=1)
        return x

    def _forward_temporal(self, x, batch_size):
        x = x.view((-1,) + x.shape[-2:])
        x = self.temporal_model(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = x.view(batch_size, -1, x.shape[-1])
        x = x.mean(dim=-2)
        x = x.softmax(dim=-1)
        return x

    def _forward_view(self, x):
        x = self.embedding(x)
        # Preserve OCTA track state across frames (do NOT reset here); the outer
        # model.reset() in the eval loop resets between videos.
        x = torch.stack([self.spatial_model(x[:, t]) for t in range(x.shape[1])], dim=1)
        return x
