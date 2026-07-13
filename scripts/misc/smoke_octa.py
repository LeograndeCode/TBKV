import copy
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.octa_vivit import OCTAFactorizedViViT
from src.models.vivit import FactorizedViViT


def build_octa(enable):
    base = yaml.safe_load(open("configs/models/vivit_b_kinetics400.yml"))
    cfg = copy.deepcopy(base["model"])
    cfg["spatial_config"]["block_config"].update(dict(
        enable_object_cache=enable, cluster_similarity_threshold=0.6,
        offset_window_size=2.0, object_match_threshold=0.55,
        object_merge_ratio=0.5, track_eviction_patience=4,
        merging_iterations=2, caching=False))
    cfg["temporal_config"]["block_config"]["enable_object_cache"] = False
    m = OCTAFactorizedViViT(**cfg)
    sd = torch.load("weights/vivit_b_kinetics400.pth", map_location="cpu")
    miss, unexp = m.load_state_dict(sd, strict=False)
    m.eval()
    return m, miss, unexp


torch.manual_seed(0)
x = torch.randn(1, 6, 3, 224, 224)

# 1) Object-cache ON: caching + matching passes run without error.
m, miss, unexp = build_octa(True)
print("weights: missing", len(miss), "unexpected", len(unexp))
m.reset(); m.clear_cache(); m.set_mode("caching")
with torch.inference_mode():
    _ = m(x[:, :3])
m.set_mode("matching")
with torch.inference_mode():
    out = m(x[:, 3:])
print("OCTA ON  output:", tuple(out.shape), "argmax", int(out.argmax()))

# 2) enable_object_cache=False must equal the plain dense ViViT forward.
m_off, _, _ = build_octa(False)
base = yaml.safe_load(open("configs/models/vivit_b_kinetics400.yml"))
plain = FactorizedViViT(**base["model"])
plain.load_state_dict(torch.load("weights/vivit_b_kinetics400.pth", map_location="cpu"))
plain.eval()
with torch.inference_mode():
    o_off = m_off(x)
    o_plain = plain(x)
max_diff = (o_off - o_plain).abs().max().item()
print("dense-bypass max|OCTA_off - plain| =", max_diff)
assert max_diff < 1e-4, f"dense bypass mismatch: {max_diff}"
print("OK: enable_object_cache=False matches plain dense forward")
