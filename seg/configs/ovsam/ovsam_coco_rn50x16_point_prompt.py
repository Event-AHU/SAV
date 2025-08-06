from mmcv.ops import RoIAlign
from mmdet.models import FPN, SingleRoIExtractor
from mmengine.config import read_base


from seg.models.data_preprocessor import OVSAMDataPreprocessor
from seg.models.backbones import OpenCLIPBackbone
from seg.models.detectors.ovsam import OVSAM
# from seg.models.heads.ovsam_head_prompt_content import OVSAMHead
from seg.models.heads.ovsam_head_prompt_all_2 import OVSAMHead
from seg.models.necks import SAMPromptEncoder, MultiLayerTransformerNeck
from seg.models.label_anything.models.prompt_encoder import RandomMatrixEncoder
from seg.models.label_anything.models.prompt_encoder import PromptImageEncoder
from seg.models.label_anything.models.transformer import TwoWayTransformer
with read_base():
    from .._base_.default_runtime import *
    from .._base_.datasets.carseg import *
    from .._base_.schedules.schedule_12e import *

# 定义基础类和新类的ID
CAR_BASE_IDS = list(range(13))  #
CAR_NOVEL_IDS = []

image_size = (1024, 1024)
_data_preprocessor = dict(
    type=OVSAMDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=image_size[0],
    pad_mask=False,
    mask_pad_value=0,
    pad_seg=False,
    seg_pad_value=255,
    batch_augments=None,
    use_center_point=True,
)
model = dict(
    type=OVSAM,
    data_preprocessor=_data_preprocessor,
    use_gt_prompt=True,
    use_clip_feat=True,
    use_head_feat=True,
    use_point=True,
    num_classes=13,  # 修改为CarPartsDataset的类别数
    base_classes=CAR_BASE_IDS,
    novel_classes=CAR_NOVEL_IDS,
    backbone=dict(
        type=OpenCLIPBackbone,
        model_name='RN50x16',
        fix=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=r"/data1/Code/wangziwen/Project/123/ovsam-main/work_dirs/clip2sam_coco_rn50x16_prompt/all/epoch_50.pth",
            prefix='backbone',
        )
        # init_cfg=dict(
        #     type='clip_pretrain',
        #     checkpoint='openai'
        # )
    ),
    neck=dict(
        type=MultiLayerTransformerNeck,
        input_size=(1024, 1024),
        in_channels=[384, 768, 1536, 3072],
        strides=[4, 8, 16, 32],
        layer_ids=(0, 1, 2, 3),
        embed_channels=1280,
        out_channels=256,
        fix=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=r"/data1/Code/wangziwen/Project/123/ovsam-main/work_dirs/clip2sam_coco_rn50x16_prompt/epoch_50.pth",
            prefix='neck',
        )
    ),
    fpn_neck=dict(
        type=FPN,
        in_channels=[384, 768, 1536, 3072],
        out_channels=256,
        num_outs=4,
        init_cfg=dict(
            type='Pretrained',
            # checkpoint='./models/clip2sam_coco_rn50x16.pth',
            checkpoint=r"/data1/Code/wangziwen/Project/123/ovsam-main/work_dirs/clip2sam_coco_rn50x16_prompt/epoch_50.pth",
            prefix='fpn_neck',
        ),
    ),
    prompt_encoder=dict(
        type=SAMPromptEncoder,
        model_name='vit_h',
        fix=True,
        init_cfg=dict(
            type='sam_pretrain',
            checkpoint='vit_h'
        )
    ),
    mask_decoder=dict(
        type=OVSAMHead,
        gen_box=True,
        model_name='vit_h',
        with_label_token=True,
        fix=False,
        ov_classifier_name='RN50x16_CarPartsDataset_new',  # 修改为适配CarPartsDataset的分类器名称
        roi_extractor=dict(
            type=SingleRoIExtractor,
            roi_layer=dict(type=RoIAlign, output_size=12, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        prompt_encoder_cfg=dict(
            type=PromptImageEncoder,
            embed_dim=256,
            image_embedding_size=(64, 64), 
            input_image_size=(640, 480),
            mask_in_chans=16,
            class_attention=False,
            example_attention=False,
            example_class_attention=False,
            class_embedding_dim=None,
            dropout=0.0,
            use_support_features=True,
            transformer=dict(
                type=TwoWayTransformer,
                depth=1,
                embedding_dim=256, 
                mlp_dim=1024, 
                attention_downsample_rate=4,
                num_heads=8,
                dropout=0.0,
            ),
            class_encoder=dict(
                type=RandomMatrixEncoder,
                bank_size=100,
                embed_dim=256,
            ),
        ),
        init_cfg=dict(
            type='Pretrained',
            # checkpoint='./models/clip2sam_coco_rn50x16.pth',
            checkpoint=r"/data1/Code/wangziwen/Project/123/ovsam-main/work_dirs/clip2sam_coco_rn50x16_prompt/epoch_50.pth",
            prefix='mask_decoder',
        )
    )
)