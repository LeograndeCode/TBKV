#!/usr/bin/env python3
"""Small self-checks for the PSM archive.

Fast and self-contained: no GPU, no dataset, no model weights. They check that
the package is wired together correctly and that the mechanisms described in
the paper behave as described, so a reviewer can tell in seconds whether the
archive is intact before committing to a multi-hour evaluation.

    python tests/test_psm.py          # from the archive root

Exit code 0 if everything passes, 1 otherwise. Runs under pytest too.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.psm import (  # noqa: E402
    EventfulPSMBlock,
    EventfulPSMTokenwiseBlock,
    compute_merge,
    psm_keep_mask,
)


# --------------------------------------------------------------- retrieval --
def test_reuse_fraction_is_exactly_gamma():
    """gamma of the candidates are dropped (reused), the rest recomputed.

    Paper: Algorithm 1 keeps the floor(gamma * k) best-matching candidates as
    the reuse set D; the kept set R is the remainder.
    """
    k = 100
    cand, cache = torch.randn(1, k, 32), torch.randn(1, 8, 32)
    for gamma in (0.0, 0.25, 0.5, 0.75, 0.95):
        keep = psm_keep_mask(cand, cache, gamma)
        expected = max(1, k - round(gamma * k))
        assert int(keep.sum()) == expected, (
            f"gamma={gamma}: recomputed {int(keep.sum())}, expected {expected}"
        )


def test_gamma_zero_is_a_no_op():
    """gamma = 0 must reduce to the unmodified host: every token recomputed."""
    cand, cache = torch.randn(2, 16, 32), torch.randn(2, 4, 32)
    assert psm_keep_mask(cand, cache, 0.0).all()


def test_empty_cache_recomputes_everything():
    """Before the memory exists there is nothing to reuse."""
    cand = torch.randn(1, 16, 32)
    assert psm_keep_mask(cand, None, 0.9).all()
    assert psm_keep_mask(cand, torch.zeros(1, 0, 32), 0.9).all()


def test_matched_tokens_are_the_ones_dropped():
    """A candidate identical to a prototype must be reused, not recomputed."""
    cache = torch.randn(1, 4, 32)
    cand = torch.randn(1, 8, 32)
    cand[0, 3] = cache[0, 2]                     # exact match -> should drop
    keep = psm_keep_mask(cand, cache, 0.25)      # drops 2 of 8
    assert not bool(keep[0, 3]), "an exact cache match was recomputed"


def test_protected_tokens_are_never_dropped():
    """The class token is protected and must survive any gamma."""
    cache = torch.randn(1, 4, 32)
    cand = cache[:, :1].repeat(1, 8, 1)          # all perfect matches
    protect = torch.zeros(1, 8, dtype=torch.bool)
    protect[0, 0] = True
    keep = psm_keep_mask(cand, cache, 0.95, protect=protect)
    assert bool(keep[0, 0]), "protected token was dropped"


# ------------------------------------------------------------ construction --
def test_merge_passes_halve_the_prototype_count():
    """T merge passes at ratio 0.5 shrink the memory geometrically.

    Paper: prototypes are built by T passes of bipartite soft matching, so
    larger T yields fewer, coarser prototypes.
    """
    tok = torch.randn(1, 64, 32)
    sizes = []
    for T in (0, 1, 2, 4):
        _m, _u, merged = compute_merge(tok, T, 0.5)
        sizes.append(merged.shape[1])
    assert sizes == sorted(sizes, reverse=True), f"not monotonic: {sizes}"
    assert sizes[0] == 64, "T=0 must leave the tokens untouched"
    assert sizes[-1] < sizes[0], "merging did not reduce the prototype count"


def test_merge_preserves_width_and_is_finite():
    tok = torch.randn(2, 32, 48)
    _m, _u, merged = compute_merge(tok, 2, 0.5)
    assert merged.shape[0] == 2 and merged.shape[-1] == 48
    assert torch.isfinite(merged).all()


# ------------------------------------------------------------------ wiring --
def test_block_classes_resolve_from_config_names():
    """Configs name block classes as strings; the backbone must resolve them."""
    from src.core.backbones import _resolve_block_class
    for name, expected in [
        ("EventfulPSMBlock", EventfulPSMBlock),
        ("EventfulPSMTokenwiseBlock", EventfulPSMTokenwiseBlock),
    ]:
        assert _resolve_block_class(name) is expected, name
    for name in ("EventfulBlock", "EventfulTokenwiseBlock"):
        assert _resolve_block_class(name) is not None, name


def test_psm_blocks_subclass_the_unmodified_host():
    """PSM is layered on the host, not a reimplementation of it."""
    from src.core.blocks import EventfulBlock, EventfulTokenwiseBlock
    assert issubclass(EventfulPSMBlock, EventfulBlock)
    assert issubclass(EventfulPSMTokenwiseBlock, EventfulTokenwiseBlock)


def test_reported_configs_resolve():
    """Every config behind a reported number loads with its _defaults chain."""
    from omegaconf import OmegaConf

    from src.utils.config import load_config
    reported = {
        "configs/evaluate/vitdet_vid": [
            "base_672", "base_1024", "temporal_672", "temporal_1024",
            "spatiotemporal_672", "spatiotemporal_1024", "stgt_672", "stgt_1024",
            "psm_eventful_filter_672", "psm_eventful_filter_1024"],
        "configs/evaluate/vivit_kinetics400": [
            "base", "eventful_psm", "eventful_psm_24", "eventful_psm_48",
            "eventful_psm_96"],
    }
    for d, names in reported.items():
        for n in names:
            cfg = load_config(ROOT / d / f"{n}.yml", to_container=False)
            cfg["_name"] = n                    # normally supplied by the CLI
            OmegaConf.to_container(cfg, resolve=True)


def test_paper_hyperparameters_are_pinned_in_the_configs():
    """The configs must carry the paper's PSM block, not a stale one."""
    from omegaconf import OmegaConf

    from src.utils.config import load_config
    det = load_config(ROOT / "configs/evaluate/vitdet_vid/psm_eventful_filter_672.yml",
                      to_container=False)
    det["_name"] = "x"
    det = OmegaConf.to_container(det, resolve=True)
    bc = det["model"]["backbone_config"]
    assert bc["block_class"] == "EventfulPSMBlock"
    assert bc["windowed_class"] == "EventfulPSMTokenwiseBlock"
    assert bc["block_config"]["merge_ratio"] == 0.5

    rec = load_config(ROOT / "configs/evaluate/vivit_kinetics400/eventful_psm_24.yml",
                      to_container=False)
    rec["_name"] = "x"
    rec = OmegaConf.to_container(rec, resolve=True)
    sc = rec["model"]["spatial_config"]
    assert sc["block_class"] == "EventfulPSMBlock"
    assert rec["token_top_k"] == [24], "k=24 budget not pinned"
    # The baseline configs deliberately default the filter OFF; the PSM rows
    # switch it on from the command line (see README section 5).
    assert sc["block_config"]["cache_reuse"] == 0.0


def test_evaluation_scripts_import():
    """Every script behind a reported number must import cleanly."""
    import importlib.util
    for name in ["vitdet_vid", "psm_eventful_vitdet_vid",
                 "vivit_kinetics400", "eventful_psm_vivit_kinetics400"]:
        path = ROOT / "scripts/evaluate" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"_probe_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)


def test_no_orphan_python_files():
    """Everything shipped under src/ must be reachable from the entry points."""
    import importlib
    for m in ["src.psm", "src.models.vitdet", "src.models.eventful_psm_vivit",
              "src.datasets.vid", "src.datasets.kinetics400",
              "src.utils.evaluate", "utils.evaluate"]:
        importlib.import_module(m)
    loaded = {
        pathlib.Path(m.__file__).resolve()
        for m in sys.modules.values()
        if getattr(m, "__file__", None)
        and ROOT in pathlib.Path(m.__file__).resolve().parents
    }
    on_disk = {p.resolve() for p in (ROOT / "src").rglob("*.py")}
    orphans = sorted(str(p.relative_to(ROOT)) for p in on_disk - loaded)
    assert not orphans, f"unreachable files shipped: {orphans}"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:                            # noqa: BLE001
            failed.append(t.__name__)
            print(f"  FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
