
from typing import List
import traceback
from mmengine import get_local_path, list_from_file, join_path, scandir, print_log

from mmdet.datasets import BaseDetDataset
from seg.datasets.coco_ins_ov import CocoOVDataset
import mmcv
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmengine import get_local_path, print_log
import copy
from typing import List
import os
@DATASETS.register_module()
class CarPartsDataset(CocoDataset):
    METAINFO = {
        'classes': (
            'Foregroud',
            'Plate',
            'Wheel',
            'Front window',
            'Back window',
            'Left front window',
            'Left front door',
            'Left back window',
            'Left back door',
            'Right front window',
            'Right front door',
            'Right back window',
            'Right back door'
        ),
        'palette': [
            (220, 20, 60),   # foreground - red
            (119, 11, 32),   # back window - dark red
            (0, 0, 142),     # right back door - dark blue
            (0, 0, 230),     # wheel - blue
            (106, 0, 228),   # right back window - purple
            (0, 60, 100),    # right front door - dark cyan
            (0, 80, 100),    # right front window - cyan
            (0, 0, 70),      # left back door - very dark blue
            (250, 170, 30),  # left front door - orange
            (100, 170, 30),  # left back window - olive
            (220, 220, 0),   # left front window - yellow
            (175, 116, 175), # plate - pink
            (250, 0, 30)     # front window - bright red
        ]
    }

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = COCO(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.METAINFO['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        valid_data_infos = []
        for data_info in self.data_list:
            img_h, img_w = data_info['height'], data_info['width']
            if min_size is not None:
                if img_h < min_size or img_w < min_size:
                    continue
            if filter_empty_gt:
                if len(data_info['instances']) == 0:
                    continue
            valid_data_infos.append(data_info)

        return valid_data_infos