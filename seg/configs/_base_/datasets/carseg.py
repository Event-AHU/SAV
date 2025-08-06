from mmcv import LoadImageFromFile
from mmdet.datasets import AspectRatioBatchSampler
from mmdet.datasets.transforms import (
    LoadAnnotations, Resize, RandomFlip, PackDetInputs
)
from mmengine.dataset import DefaultSampler
from seg.datasets.pipeliens.loading import FilterAnnotationsHB
from seg.datasets.mask_dataset import CarPartsMaskDataset
from seg.evaluation.ins_cls_iou_metric import InsClsIoUMetric

data_root = '/data1/Datasets/Seg/3DRealCar_Segment_Dataset/'
backend_args = None
dataset_type = CarPartsMaskDataset
image_size = (640, 480)

train_pipeline = [
    dict(
        type=LoadImageFromFile,
        to_float32=True,
        backend_args=backend_args
    ),
    dict(
        type=LoadAnnotations,
        with_bbox=True,
        with_mask=True,
        with_edge=False,
        backend_args=backend_args
    ),
    dict(
        type=Resize,
        scale=image_size,
        keep_ratio=True
    ),
    dict(
        type=RandomFlip,
        # prob=0.5
        prob = 0.0
    ),
    dict(
        type=FilterAnnotationsHB,
        by_box=False,
        by_mask=True,
        min_gt_mask_area=32
    ),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=CarPartsMaskDataset,
        data_root=data_root,
        data_prefix=dict(
            img='images/train'
        ),
        mask_prefix='annotations/train',
        filter_cfg=dict(
            filter_empty_gt=True,
            min_size=32
        ),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

test_pipeline = [
    dict(
        type=LoadImageFromFile,
        backend_args=backend_args
    ),
    dict(
        type=LoadAnnotations,
        with_bbox=True,
        with_mask=True,
        with_edge=False,
        backend_args=backend_args
    ),
    dict(
        type=Resize,
        scale=image_size,
        keep_ratio=False
    ),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=CarPartsMaskDataset,
        data_root=data_root,
        data_prefix=dict(
            img='images/valid'
        ),
        mask_prefix='annotations/valid', 
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=CarPartsMaskDataset, 
        data_root=data_root,
        data_prefix=dict(
            img='images/valid'
        ),
        mask_prefix='annotations/valid',
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)


val_evaluator = [
    dict(
        type=InsClsIoUMetric,
        prefix='car_ins',
    ),
]

test_evaluator = val_evaluator