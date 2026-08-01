from src.models.vivit import FactorizedViViT
try:
	from src.models.vitdet import ViTDet
	# ViTDet depends on detectron2/fvcore and may fail to import in
	# lightweight or constrained environments (missing optional deps,
	# unusable temp dir, etc.). Keep ViViT imports usable regardless.
except Exception:
	ViTDet = None

__all__ = [
	"FactorizedViViT",
	"ViTDet",
]
