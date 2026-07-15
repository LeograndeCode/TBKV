"""Modular token-reduction algorithms, usable standalone or inside TBKV.

Build one reducer and share it across a whole block stack:

    reducer = build_reducer("tome", r=8)

Then either:
  * give it to a ReducerBlock stack, and it reduces the full token sequence
    (the published algorithm, used as a baseline); or
  * give it to a TBKVBlock stack via ``fg_reducer``, and it reduces only the
    foreground tokens, while TBKV handles the background via its K/V cache.

Both paths call the same reducer object through the same plan() interface, so
the algorithms are written exactly once.
"""

from src.reduction.avit import AViTReducer
from src.reduction.base import ReduceOp, ReductionContext, TokenReducer
from src.reduction.dynamicvit import DynamicViTReducer
from src.reduction.evit import EViTReducer
from src.reduction.tome import ToMeReducer

REDUCERS = {
    "tome": ToMeReducer,
    "evit": EViTReducer,
    "dynamicvit": DynamicViTReducer,
    "avit": AViTReducer,
}


def build_reducer(name, **kwargs):
    """Instantiate a reducer by name. ``None``/"none" means no reduction."""
    if name is None or name == "none":
        return None
    if name not in REDUCERS:
        raise ValueError(
            f"Unknown reducer '{name}'. Available: {sorted(REDUCERS)}"
        )
    return REDUCERS[name](**kwargs)


__all__ = [
    "AViTReducer",
    "DynamicViTReducer",
    "EViTReducer",
    "ToMeReducer",
    "ReduceOp",
    "ReductionContext",
    "TokenReducer",
    "REDUCERS",
    "build_reducer",
]
