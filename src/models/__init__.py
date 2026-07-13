from src.models.tbkv_vivit import TBKVFactorizedViViT
from src.models.vivit import FactorizedViViT
from src.models.vivit_patched import (
	AViTPatchedViViT,
	DynamicViTPatchedViViT,
	ToMePatchedViViT,
	EViTPatchedViViT,
)

try:
	from src.models.vitdet import ViTDet
	# ViTDet depends on detectron2/fvcore and may fail to import in
	# lightweight or constrained environments (missing optional deps,
	# unusable temp dir, etc.). Keep ViViT imports usable regardless.
except Exception:
	ViTDet = None

__all__ = [
	"FactorizedViViT",
	"TBKVFactorizedViViT",
	"AViTPatchedViViT",
	"DynamicViTPatchedViViT",
	"ToMePatchedViViT",
	"EViTPatchedViViT",
	"ViTDet",
]
