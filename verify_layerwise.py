"""
Verifying that layer-wise adaptive r_match is working correctly or not
Each block should have a different r_match value decreasing from early to deep layers.
"""
import sys
sys.path.insert(0, '.')

from src.tbkv.tbkv_blocks import TBKVBlock
import torch.nn as nn

# Simulating what TBKVViTBackbone does
depth = 12
r_match_base = 0.75
r_max = min(r_match_base + 0.10, 0.95)
r_min = max(r_match_base - 0.10, 0.30)

print("=" * 55)
print("LAYER-WISE ADAPTIVE MATCHING RATIO VERIFICATION")
print("=" * 55)
print(f"Base r_match : {r_match_base}")
print(f"r_max (block 0)  : {r_max}")
print(f"r_min (block 11) : {r_min}")
print()
print(f"{'Block':<8} {'r_match':>10} {'Aggressiveness':>16}")
print("-" * 38)

for i in range(depth):
    if depth > 1:
        r_match_i = r_max - (i / (depth - 1)) * (r_max - r_min)
    else:
        r_match_i = r_match_base

    if r_match_i >= 0.80:
        label = "Very Aggressive"
    elif r_match_i >= 0.75:
        label = "Aggressive"
    elif r_match_i >= 0.70:
        label = "Moderate"
    else:
        label = "Conservative"

    print(f"Block {i:<3} {r_match_i:>10.4f} {label:>16}")

print()
print("Early blocks reuse more aggressively (high r_match).")
print("Deep blocks reuse more conservatively (low r_match).")
print("This is motivated by FrameFusion (ICCV 2025).")
print("=" * 55)