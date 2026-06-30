import torch.nn as nn
from detectron2.config import LazyConfig, instantiate
from detectron2.structures import ImageList
from torchvision.transforms import Normalize

from src.tbkv.tbkv_backbone import TBKVViTBackbone
from src.tbkv.tbkv_blocks import TBKVBlock
from src.tbkv.cache import Cache
from src.core.base import ExtendedModule, numeric_tuple
from src.core.blocks import LN_EPS
from src.utils.image import as_float32, pad_to_size


# Resources consulted:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/utils.py
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py


class LinearEmbedding(nn.Module):
    """
    The initial linear patch-embedding layer for ViTDet. Linearly
    transforms each input patch into a token vector.
    """

    def __init__(self, input_channels, dim, patch_size):
        """
        :param input_channels: The number of image channels (e.g., 3 for
        RGB images)
        :param dim: The dimensionality of token vectors
        :param patch_size: The patch size for each token (a 2-element
        tuple/list)
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # (batch, dim, height, width)

        x = self.conv(x)
        # (batch, dim, height, width)

        # Flatten the spatial axes.
        x = x.flatten(start_dim=-2)
        # (batch, dim, patch)

        x = x.transpose(1, 2)
        # (batch, patch, dim)

        return x


class PointwiseLayerNorm2d(nn.LayerNorm):
    """
    A LayerNorm operation which performs x.permute(0, 2, 3, 1) before
    applying the normalization. The permutation is inverted after
    normalization.
    """

    def forward(self, x):
        # (batch, dim, height, width)

        x = x.permute(0, 2, 3, 1)
        # (batch, height, width, dim)

        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        # (batch, dim, height, width)

        return x


class SimplePyramid(nn.Module):
    """
    The ViTDet feature pyramid (precedes the object detection head).
    """

    def __init__(self, scale_factors, dim, out_channels):
        """
        :param scale_factors: A list of spatial scale factors
        :param dim: The dimensionality of token vectors in the
        Transformer backbone
        :param out_channels: The number of output channels (the number
        of channels expected by the object detection head)
        """
        super().__init__()
        self.stages = nn.ModuleList(
            self._build_scale(scale, dim, out_channels) for scale in scale_factors
        )
        self.max_pool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        x = [stage(x) for stage in self.stages]
        x.append(self.max_pool(x[-1]))
        return x

    @staticmethod
    def _build_scale(scale, dim, out_channels):
        assert scale in [4.0, 2.0, 1.0, 0.5]
        if scale == 0.5:
            mid_dim = dim
            start_layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif scale == 1.0:
            mid_dim = dim
            start_layers = []
        elif scale == 2.0:
            mid_dim = dim // 2
            start_layers = [nn.ConvTranspose2d(dim, mid_dim, kernel_size=2, stride=2)]
        else:  # scale == 4.0
            mid_dim = dim // 4
            start_layers = [
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                PointwiseLayerNorm2d(dim // 2, eps=LN_EPS),
                nn.GELU(),
                nn.ConvTranspose2d(dim // 2, mid_dim, kernel_size=2, stride=2),
            ]
        common_layers = [
            nn.Conv2d(mid_dim, out_channels, kernel_size=1, bias=False),
            PointwiseLayerNorm2d(out_channels, eps=LN_EPS),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            PointwiseLayerNorm2d(out_channels, eps=LN_EPS),
        ]
        return nn.Sequential(*start_layers, *common_layers)


class TBKVViTDet(ExtendedModule):
    """
    The ViTDet object detection Transformer model. See
    configs/models/vitdet_b_coco.yml for an example configuration.
    """

    def __init__(
        self,
        backbone_config,
        classes,
        detectron2_config,
        input_shape,
        normalize_mean,
        normalize_std,
        output_channels,
        patch_size,
        scale_factors,
    ):
        """
        :param backbone_config: A dict containing kwargs for the
        backbone constructor
        :param classes: The number of object classes
        :param detectron2_config: Path of a Python file containing a
        Detectron2 config for the detection head
        :param input_shape: The (c, h, w) shape for inputs (the
        preprocessing spatially pads inputs to this shape)
        :param normalize_mean: The mean to use with
        torchvision.transforms.Normalize
        :param normalize_std: The standard deviation to use with
        torchvision.transforms.Normalize
        :param output_channels: The number of channels expected by the
        object detection head
        :param patch_size: The patch size for each token (a 2-element
        tuple/list)
        :param scale_factors: Scale factors for the SimplePyramid module
        """
        super().__init__()
        input_c, input_h, input_w = input_shape
        patch_size = numeric_tuple(patch_size, length=2)
        self.backbone_input_size = (input_h // patch_size[0], input_w // patch_size[1])

        # Set up submodules.
        self.preprocessing = ViTDetPreprocessing(
            input_shape, normalize_mean, normalize_std
        )
        dim = backbone_config["block_config"]["dim"]
        self.embedding = LinearEmbedding(input_c, dim, patch_size)
        self.backbone = TBKVViTBackbone(
            input_size=self.backbone_input_size,
            **backbone_config,
        )
        self.pyramid = SimplePyramid(scale_factors, dim, output_channels)
        detectron2_config = LazyConfig.load(detectron2_config)["model"]
        self.proposal_generator = instantiate(detectron2_config["proposal_generator"])
        roi_heads_config = detectron2_config["roi_heads"]
        roi_heads_config["num_classes"] = classes
        self.roi_heads = instantiate(roi_heads_config)

    def forward(self, x):
        images, x = self.pre_backbone(x)
        x = self._forward_backbone_no_reset(x)
        results = self.post_backbone(images, x)
        return results

    def set_mode(self, mode):
        caching = (mode == "caching")
        for module in self.backbone.modules():
            if isinstance(module, TBKVBlock):
                module.caching = caching

    def _qkv_from_norm(self, block, x_norm):
        b, n, c = x_norm.shape
        head_dim = c // block.heads
        q = block.q(x_norm).reshape(b, n, block.heads, head_dim).permute(0, 2, 1, 3)
        k = block.k(x_norm).reshape(b, n, block.heads, head_dim).permute(0, 2, 1, 3)
        v = block.v(x_norm).reshape(b, n, block.heads, head_dim).permute(0, 2, 1, 3)
        return q, k, v

    def _forward_tbkv_block_vitdet(self, block, x, prev_scores):
        skip_1 = x
        x_norm = block.input_layer_norm(x)
        b, n, c = x_norm.shape

        # Caching (or cold start): compute full q/k/v and refresh cache.
        if block.caching or block.cache is None or prev_scores is None:
            q, k, v = self._qkv_from_norm(block, x_norm)
            block.cache = Cache(k.detach(), v.detach(), x_norm.detach())
        else:
            # Matching mode with fixed token count: reuse cached K/V for matched positions,
            # recompute K/V only for unmatched positions.
            cache_tokens = block.cache.tokens
            x_n = x_norm / (x_norm.norm(dim=-1, keepdim=True) + 1e-6)
            c_n = cache_tokens / (cache_tokens.norm(dim=-1, keepdim=True) + 1e-6)
            sim = (x_n * c_n).sum(dim=-1)

            n_match = max(0, min(n, int(n * block.r_match)))
            sort_idx = sim.argsort(dim=-1, descending=True)
            idx_match = sort_idx[:, :n_match]
            idx_unmatch = sort_idx[:, n_match:]

            q, _, _ = self._qkv_from_norm(block, x_norm)
            k = block.cache.K.clone()
            v = block.cache.V.clone()

            n_unmatch = idx_unmatch.shape[1]
            if n_unmatch > 0:
                gather_idx = idx_unmatch.unsqueeze(-1).expand(-1, -1, c)
                x_unmatch = x_norm.gather(1, gather_idx)
                _, k_unmatch, v_unmatch = self._qkv_from_norm(block, x_unmatch)

                head_dim = c // block.heads
                scatter_idx = idx_unmatch.unsqueeze(1).unsqueeze(-1).expand(-1, block.heads, -1, head_dim)
                k = k.scatter(2, scatter_idx, k_unmatch)
                v = v.scatter(2, scatter_idx, v_unmatch)

            # Keep cache refreshed with most recent tokens/kv.
            block.cache = Cache(k.detach(), v.detach(), x_norm.detach())

        attn = block.matmul(q / block.scale, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        x = block.matmul(attn, v)
        x = block._recombine_heads(x)

        x = block.projection(x)
        x = block.add(block.drop_path(x), skip_1)

        skip_2 = x
        x = block.mlp_layer_norm(x)
        x = block._forward_mlp(x)
        x = block.add(block.drop_path(x), skip_2)

        # Use mean attention as next-block saliency signal.
        return x, attn.mean(dim=1)

    def _forward_backbone_no_reset(self, x):
        # ViTDet evaluates frame-by-frame; keep TBKV cache/mode state across frames.
        x = self.backbone.position_encoding(x)
        prev_scores = None
        for block in self.backbone.blocks:
            x, prev_scores = self._forward_tbkv_block_vitdet(block, x, prev_scores)
        return x

    def post_backbone(self, images, x):
        """
        Computes the portion of the model after the Transformer
        backbone.
        """
        x = x.transpose(-1, -2)
        x = x.view(x.shape[:-1] + self.backbone_input_size)
        x = self.pyramid(x)

        # Compute region proposals and bounding boxes.
        x = dict(zip(self.proposal_generator.in_features, x))
        proposals = self.proposal_generator(images, x, None)[0]
        result = self.roi_heads(images, x, proposals, None)[0]
        result = [
            {"boxes": y.pred_boxes.tensor, "scores": y.scores, "labels": y.pred_classes}
            for y in result
        ]
        return result

    def pre_backbone(self, x):
        """
        Computes the portion of the model before the Transformer
        backbone.
        """
        x = as_float32(x)  # Range [0, 1]
        x = self.preprocessing(x)
        images = ImageList.from_tensors([x])
        x = self.embedding(x)
        return images, x


class ViTDetPreprocessing(nn.Module):
    """
    Preprocessing for ViTDet. Applies value normalization and square
    padding. Expects inputs scaled to the range [0, 1].
    """

    def __init__(self, input_shape, normalize_mean, normalize_std):
        """
        :param input_shape: The (c, h, w) shape to which inputs should
        be padded
        :param normalize_mean: The mean to use with
        torchvision.transforms.Normalize
        :param normalize_std: The standard deviation to use with
        torchvision.transforms.Normalize
        """
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.normalization = Normalize(normalize_mean, normalize_std)

    def forward(self, x):
        # This normalization assumes x in the range [0, 255], but the
        # parent model (ViTDet) scales the input image to [0, 1].
        x = self.normalization(x * 255.0)

        # This is bottom-right padding, so it won't affect the bounding
        # box coordinates.
        x = pad_to_size(x, self.input_shape[-2:])

        return x
