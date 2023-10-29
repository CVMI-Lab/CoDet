
import torch.nn as nn
from functools import partial

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import ViT, SimpleFeaturePyramid
from centernet.modeling.backbone.fpn_p5 import LastLevelP6P7_P5

@BACKBONE_REGISTRY.register()
def build_vitdet(cfg, input_shape):
    input_size = cfg.INPUT.TRAIN_SIZE
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    vit = ViT(
        img_size=input_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        drop_path_rate=0.4,
        window_size=32,
        mlp_ratio=4*2/3,
        qkv_bias=True,
        use_act_checkpoint=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=(
            list(range(0, 2)) + \
            list(range(3, 5)) + \
            list(range(6, 8)) + \
            list(range(9, 11)) + \
            list(range(12, 14)) + \
            list(range(15, 17)) + \
            list(range(18, 20)) + \
            list(range(21, 23))
        ),
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    )
    backbone = SimpleFeaturePyramid(
        net=vit,
        in_feature="last_feat",
        out_channels=out_channels,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        norm="LN",
        square_pad=input_size,
    )
    return backbone