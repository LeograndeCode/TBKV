# Layer-wise Adaptive r_match for TBKV

## Summary

This branch adds a layer-wise adaptive `r_match` schedule to TBKV.

Instead of using the same temporal matching/reuse ratio for every transformer block, the model now assigns a different `r_match` value to each block. Early blocks use a higher matching ratio, while deeper blocks use a lower matching ratio.

The motivation is that early ViT layers usually capture lower level visual features that are more stable across nearby video frames, while deeper layers capture more semantic information and should reuse cached tokens more conservatively.

## Main implementation file

```text
src/tbkv/tbkv_backbone.py
The schedule is applied when TBKVViTBackbone constructs the transformer blocks.

For a base value such as:

r_match = 0.75

the schedule becomes approximately:

Block 0  -> 0.85
Block 1  -> 0.8318
Block 2  -> 0.8136
Block 3  -> 0.7955
Block 4  -> 0.7773
Block 5  -> 0.7591
Block 6  -> 0.7409
Block 7  -> 0.7227
Block 8  -> 0.7045
Block 9  -> 0.6864
Block 10 -> 0.6682
Block 11 -> 0.65

This means early blocks reuse more aggressively and deeper blocks reuse more conservatively.

Verification scripts

Two verification scripts are included:

verify_layerwise.py
verify_model_layerwise.py
1. Formula-level verification

Run:

python verify_layerwise.py

This prints the expected decreasing r_match schedule from early to deep layers.

2. Model-level verification

Run:

python verify_model_layerwise.py

This instantiates a TBKVViTBackbone and confirms that each actual TBKVBlock receives a unique r_match value.

Evaluation notes

Use the vanilla ViViT script for the true baseline:

python scripts/evaluate/vivit_kinetics400.py base n_items=25

Use the TBKV script for layer-wise TBKV:

python scripts/evaluate/tbkv_vivit_kinetics400.py tbkv n_items=25 model.spatial_config.block_config.r_match=0.75 model.temporal_config.block_config.r_match=0.75

Important: running the TBKV script with base is not a true vanilla baseline because the TBKV evaluation path still uses caching and matching mode. The true vanilla baseline should be run with scripts/evaluate/vivit_kinetics400.py.

Preliminary CPU results

These results were used only for smoke testing and early validation before GPU evaluation.

Method	Videos	Top-1	Top-5	Linear FLOPs	Matmul FLOPs
Vanilla ViViT	25	84.00%	96.00%	3.218e+12	1.374e+11
Layer-wise TBKV, base r_match=0.75	25	84.00%	100.00%	2.905e+12	1.237e+11

The layer-wise setting preserved Top-1 accuracy on this 25-video subset and reduced compute by approximately 9.7% in linear FLOPs and 10.0% in matmul FLOPs.

These CPU results are preliminary. Full evaluation should be run on GPU.
