from copy import deepcopy

from src.models.vivit import FactorizedViViT


class _PatchedBaseViViT(FactorizedViViT):
    def _apply_ats(self, spatial_config, temporal_config, fraction):
        s_cfg = deepcopy(spatial_config)
        t_cfg = deepcopy(temporal_config)
        # Apply token reduction only on the spatial stack. Keep the temporal
        # stack unchanged to avoid compounding errors and unfairly harsh drops.
        s_cfg.setdefault("block_config", {})["ats_fraction"] = float(fraction)
        t_cfg.setdefault("block_config", {}).pop("ats_fraction", None)
        return s_cfg, t_cfg


class AViTPatchedViViT(_PatchedBaseViViT):
    """Practical AViT-style proxy via adaptive token sampling (ATS)."""

    def __init__(self, *args, spatial_config, temporal_config,
                 avit_fraction=0.85, **kwargs):
        s_cfg, t_cfg = self._apply_ats(spatial_config, temporal_config, avit_fraction)
        super().__init__(*args, spatial_config=s_cfg, temporal_config=t_cfg, **kwargs)


class DynamicViTPatchedViViT(_PatchedBaseViViT):
    """Practical DynamicViT-style proxy via stronger ATS pruning."""

    def __init__(self, *args, spatial_config, temporal_config,
                 dynamicvit_fraction=0.8, **kwargs):
        s_cfg, t_cfg = self._apply_ats(
            spatial_config, temporal_config, dynamicvit_fraction
        )
        super().__init__(*args, spatial_config=s_cfg, temporal_config=t_cfg, **kwargs)


class ToMePatchedViViT(_PatchedBaseViViT):
    """Practical ToMe-style proxy via moderate ATS token reduction."""

    def __init__(self, *args, spatial_config, temporal_config,
                 tome_fraction=0.88, **kwargs):
        s_cfg, t_cfg = self._apply_ats(spatial_config, temporal_config, tome_fraction)
        super().__init__(*args, spatial_config=s_cfg, temporal_config=t_cfg, **kwargs)


class EViTPatchedViViT(_PatchedBaseViViT):
    """Practical EViT-style proxy via ATS with a conservative keep rate."""

    def __init__(self, *args, spatial_config, temporal_config,
                 evit_fraction=0.82, **kwargs):
        s_cfg, t_cfg = self._apply_ats(spatial_config, temporal_config, evit_fraction)
        super().__init__(*args, spatial_config=s_cfg, temporal_config=t_cfg, **kwargs)
