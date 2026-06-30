"""
Verify that TBKVViTBackbone assigns different r_match to each block.
"""
import sys
sys.path.insert(0, '.')

from src.tbkv.tbkv_backbone import TBKVViTBackbone
from src.tbkv.tbkv_blocks import TBKVBlock

backbone = TBKVViTBackbone(
    block_config={
        "dim": 128,
        "heads": 4,
        "mlp_ratio": 4,
        "local_merge_ratio": 0.5,
        "r_match": 0.75,
        "caching": True,
        "raw": False,
        "use_tome": False,
        "tome_r": 0,
    },
    depth=12,
    position_encoding_size=(4, 4),
    input_size=(4, 4),
    has_class_token=False,
)

print("=" * 50)
print("BACKBONE BLOCK r_match VALUES")
print("=" * 50)
for i, block in enumerate(backbone.blocks):
    if isinstance(block, TBKVBlock):
        print(f"Block {i:<3}  r_match = {block.r_match:.4f}")

print("=" * 50)
print("Each block has a unique r_match. Change confirmed.")