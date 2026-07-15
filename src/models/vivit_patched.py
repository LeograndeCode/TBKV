"""ViViT patched with real token-reduction algorithms.

These are the *baselines*: each algorithm runs on the full token sequence of the
spatial stack, as published. The exact same reducer objects are reused by TBKV
(via ``fg_reducer``) to reduce only foreground tokens -- see src/reduction/.

Reduction is applied to the spatial stack only. The temporal stack sees a
handful of tokens (one per frame), so reducing it saves nothing and only
compounds error.

Fidelity note: ToMe and EViT are faithful and training-free. DynamicViT's and
AViT's *schedules* are faithful, but both papers train a scoring mechanism, and
we run inference-only on frozen ViViT weights -- see the module docstrings in
src/reduction/dynamicvit.py and src/reduction/avit.py for exactly what that
does and does not preserve. Label them accordingly in any comparison table.
"""

from copy import deepcopy

from src.models.vivit import FactorizedViViT
from src.reduction import build_reducer
from src.reduction.backbone import ReducedViTBackbone


class _ReducedViViT(FactorizedViViT):
    """ViViT whose spatial backbone applies a token reducer to all tokens."""

    reducer_name = None

    def __init__(self, *args, spatial_config, temporal_config,
                 reducer_args=None, **kwargs):
        spatial_config = deepcopy(spatial_config)
        temporal_config = deepcopy(temporal_config)
        super().__init__(
            *args, spatial_config=spatial_config,
            temporal_config=temporal_config, **kwargs
        )

        reducer = build_reducer(self.reducer_name, **(reducer_args or {}))

        # Swap the spatial backbone for a reducer-aware one. Safe to do after
        # super().__init__: checkpoint weights are loaded by the evaluation
        # harness *after* the model is constructed, and ReducerBlock keeps
        # Block's parameter layout, so the state dict still matches.
        input_size = self.spatial_model.backbone.blocks[0].input_size
        backbone_config = {
            k: v for k, v in spatial_config.items() if k != "block_config"
        }
        self.spatial_model.backbone = ReducedViTBackbone(
            block_config=spatial_config["block_config"],
            input_size=input_size,
            has_class_token=True,
            reducer=reducer,
            **backbone_config,
        )


class ToMePatchedViViT(_ReducedViViT):
    """ToMe (Bolya et al., ICLR 2023): bipartite soft matching, r merged/block."""

    reducer_name = "tome"

    def __init__(self, *args, tome_r=8, **kwargs):
        super().__init__(*args, reducer_args={"r": tome_r}, **kwargs)


class EViTPatchedViViT(_ReducedViViT):
    """EViT (Liang et al., ICLR 2022): keep attentive tokens, fuse the rest."""

    reducer_name = "evit"

    def __init__(self, *args, evit_keep_rate=0.7, evit_fuse=True,
                 evit_prune_layers=(3, 6, 9), **kwargs):
        super().__init__(
            *args,
            reducer_args={
                "keep_rate": evit_keep_rate,
                "fuse": evit_fuse,
                "prune_layers": tuple(evit_prune_layers),
            },
            **kwargs,
        )


class DynamicViTPatchedViViT(_ReducedViViT):
    """DynamicViT (Rao et al., NeurIPS 2021): staged pruning at fixed depths."""

    reducer_name = "dynamicvit"

    def __init__(self, *args, dynamicvit_keep_ratio=0.7,
                 dynamicvit_prune_layers=(3, 6, 9), **kwargs):
        super().__init__(
            *args,
            reducer_args={
                "keep_ratio": dynamicvit_keep_ratio,
                "prune_layers": tuple(dynamicvit_prune_layers),
            },
            **kwargs,
        )


class AViTPatchedViViT(_ReducedViViT):
    """A-ViT (Yin et al., CVPR 2022): parameter-free per-token halting."""

    reducer_name = "avit"

    def __init__(self, *args, avit_gamma=5.0, avit_beta=10.0,
                 avit_max_drop_rate=0.2, **kwargs):
        super().__init__(
            *args,
            reducer_args={
                "gamma": avit_gamma,
                "beta": avit_beta,
                "max_drop_rate": avit_max_drop_rate,
            },
            **kwargs,
        )
