import os.path as osp
import os
import numpy as np
from PIL import Image
import mmengine.fileio as fileio
from mmengine.dataset import BaseDataset
from mmdet.registry import DATASETS
import logging
from tqdm import tqdm
import time
import gc
from typing import List, Dict, Any, Optional, Union
from pycocotools import mask as maskUtils
from mmdet.structures.mask import BitmapMasks, PolygonMasks
import torch
from mmengine.structures import InstanceData
import cv2

@DATASETS.register_module()
class CarPartsMaskDataset_fake(BaseDataset):
    """Dataset for car parts segmentation with mask images."""
    
    METAINFO = {
        'classes': (
            'back_bumper',
            'back_left_door',
            'back_left_wheel',
            'back_left_window',
            'back_license_plate',
            'back_right_door',
            'back_right_wheel',
            'back_right_window',
            'back_windshield',
            'front_bumper',
            'front_left_door',
            'front_left_wheel',
            'front_left_window',
            'front_license_plate',
            'front_right_door',
            'front_right_wheel',
            'front_right_window',
            'front_windshield',
            'hood',
            'left_frame',
            'left_head_light',
            'left_mirror',
            'left_quarter_window',
            'left_tail_light',
            'right_frame',
            'right_head_light',
            'right_mirror',
            'right_quarter_window',
            'right_tail_light',
            'roof',
            'trunk'
        ),
        'palette': [
            (220, 20, 60),    # back_bumper
            (255, 0, 0),      # back_left_door
            (0, 0, 142),      # back_left_wheel
            (119, 11, 32),    # back_left_window
            (0, 60, 100),     # back_license_plate
            (0, 80, 100),     # back_right_door
            (0, 0, 230),      # back_right_wheel
            (106, 0, 228),    # back_right_window
            (0, 80, 250),     # back_windshield
            (0, 60, 250),     # front_bumper
            (255, 40, 0),     # front_left_door
            (0, 0, 192),      # front_left_wheel
            (250, 170, 30),   # front_left_window
            (100, 170, 30),   # front_license_plate
            (220, 220, 0),    # front_right_door
            (0, 0, 170),      # front_right_wheel
            (190, 153, 153),  # front_right_window
            (70, 130, 180),   # front_windshield
            (255, 0, 255),    # hood
            (154, 77, 66),    # left_frame
            (194, 158, 0),    # left_head_light
            (63, 133, 205),   # left_mirror
            (153, 153, 153),  # left_quarter_window
            (180, 165, 180),  # left_tail_light
            (100, 100, 100),  # right_frame
            (150, 100, 100),  # right_head_light
            (0, 191, 255),    # right_mirror
            (160, 160, 164),  # right_quarter_window
            (170, 170, 170),  # right_tail_light
            (32, 178, 170),   # roof
            (0, 255, 0)       # trunk
        ]
    }

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[Dict] = None,
                 data_root: str = '',
                 data_prefix: Dict[str, str] = dict(img=''),
                 mask_prefix: str = 'masks',
                 filter_cfg: Optional[Dict] = None,
                 indices: Optional[List[int]] = None,
                 serialize_data: bool = True,
                 pipeline: Optional[List[Dict]] = None,
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 backend_args: Optional[Dict] = None,
                 min_mask_area: int = 200):  # Added a min_mask_area parameter
        
        self.setup_logging()
        self.mask_prefix = self.clean_filename(mask_prefix)
        self.backend_args = backend_args
        self.min_mask_area = min_mask_area  # Store the minimum mask area threshold
        
        if data_root:
            data_root = self.clean_filename(data_root)
        
        if data_prefix and 'img' in data_prefix:
            data_prefix['img'] = self.clean_filename(data_prefix['img'])
        
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=False,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def setup_logging(self):
        """设置日志记录"""
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f'dataset_processing_{time.strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        
    def clean_filename(self, filename: str) -> str:
        """清理和规范化文件路径"""
        if not filename:
            return filename
        cleaned = os.path.normpath(filename)
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32)
        return cleaned

    def create_default_instances(self, height: int, width: int) -> Dict[str, Any]:
        """创建默认实例数据"""
        default_mask = np.zeros((height, width), dtype=np.uint8)
        rle = maskUtils.encode(
            np.array(default_mask[:, :, None], order='F')
        )[0]
        
        default_instance = {
            'bbox': [0.0, 0.0, float(width - 1), float(height - 1)],
            'bbox_label': 0,
            'mask': {
                'counts': rle['counts'],
                'size': rle['size']
            },
            'ignore_flag': 0
        }
        return {'instances': [default_instance]}

    def validate_instances(self, instances: List[Dict[str, Any]], img_id: str = '') -> bool:
        """验证实例数据的有效性"""
        try:
            if not isinstance(instances, list):
                logging.error(f"Invalid instances type for {img_id}: {type(instances)}")
                return False
                
            for instance in instances:
                required_keys = ['bbox', 'bbox_label', 'mask', 'ignore_flag']
                if not all(key in instance for key in required_keys):
                    logging.error(f"Missing required keys in instance for {img_id}")
                    return False
                
                bbox = instance['bbox']
                if not isinstance(bbox, list) or len(bbox) != 4:
                    logging.error(f"Invalid bbox format for {img_id}")
                    return False
                
                if not isinstance(instance['bbox_label'], int):
                    logging.error(f"Invalid bbox_label type for {img_id}")
                    return False
                
                if not isinstance(instance['ignore_flag'], int):
                    logging.error(f"Invalid ignore_flag type for {img_id}")
                    return False
                
                mask = instance['mask']
                if not isinstance(mask, dict) or 'counts' not in mask or 'size' not in mask:
                    logging.error(f"Invalid mask format for {img_id}")
                    return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error validating instances for {img_id}: {str(e)}")
            return False

    def process_mask(self, mask: np.ndarray, height: int, width: int, 
                    img_id: str = '') -> Dict[str, Any]:
        """处理单个mask，提取实例信息，并过滤掉小区域"""
        try:
            if mask.ndim != 2:
                logging.warning(f"跳过非2D mask，shape: {mask.shape}, img_id: {img_id}")
                return self.create_default_instances(height, width)

            unique_ids = np.unique(mask)
            unique_ids = unique_ids[unique_ids != 0]  # 排除背景

            if len(unique_ids) == 0:
                logging.debug(f"Mask中没有有效标签, img_id: {img_id}")
                return self.create_default_instances(height, width)

            instances = []
            filtered_count = 0  # 记录过滤掉的实例数量
            
            for instance_id in unique_ids:
                instance_mask = (mask == instance_id).astype(np.uint8)
                
                if instance_mask.shape != (height, width):
                    continue
                
                # 计算掩码区域面积
                area = np.sum(instance_mask)
                
                # 过滤掉小于阈值的区域
                if area < self.min_mask_area:
                    filtered_count += 1
                    continue
                    
                # 将二值mask转换为RLE格式
                rle = maskUtils.encode(
                    np.array(instance_mask[:, :, None], order='F')
                )[0]
                
                # 获取边界框
                bbox = maskUtils.toBbox(rle)  # [x,y,width,height]
                # 转换为[x1,y1,x2,y2]格式
                bbox[2:] += bbox[:2]
                
                instance = {
                    'bbox': bbox.tolist(),
                    'bbox_label': int(instance_id - 1),
                    'mask': {
                        'counts': rle['counts'],
                        'size': rle['size']
                    },
                    'ignore_flag': 0
                }
                instances.append(instance)

            if filtered_count > 0:
                logging.info(f"图像 {img_id} 中过滤掉 {filtered_count} 个小区域 (面积 < {self.min_mask_area})")
                
            if not instances:
                return self.create_default_instances(height, width)

            result = {'instances': instances}
            
            if not self.validate_instances(result['instances'], img_id):
                return self.create_default_instances(height, width)
                
            return result

        except Exception as e:
            logging.error(f"处理mask时出错: {str(e)}, img_id: {img_id}")
            return self.create_default_instances(height, width)

    def load_data_list(self) -> List[Dict[str, Any]]:
        """加载数据列表"""
        start_time = time.time()
        data_list = []
        skipped_count = 0
        filtered_masks_count = 0  # 添加计数器记录过滤掉的掩码数量
        
        img_dir = self.clean_filename(osp.join(self.data_root, self.data_prefix['img']))
        mask_dir = self.clean_filename(osp.join(self.data_root, self.mask_prefix))
        
        logging.info(f"开始扫描数据...")
        logging.info(f"图像目录: {img_dir}")
        logging.info(f"掩码目录: {mask_dir}")
        logging.info(f"最小掩码面积阈值: {self.min_mask_area}")

        # 获取所有PNG掩码文件
        mask_files = []
        with tqdm(desc="扫描目录结构", unit="files") as pbar:
            for root, _, files in os.walk(mask_dir):
                for file in files:
                    if file.endswith('.png') and '_color' not in file:
                        rel_path = os.path.relpath(os.path.join(root, file), mask_dir)
                        mask_files.append(rel_path)
                    pbar.update(1)

        if not mask_files:
            raise ValueError(f"未找到mask文件在: {mask_dir}")

        logging.info(f"找到 {len(mask_files)} 个候选文件，开始处理...")
        
        # 获取所有图像文件
        image_files = {}
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png']:
            for root, _, files in os.walk(img_dir):
                for file in files:
                    if file.endswith(ext):
                        # 保存不带扩展名的文件名作为键
                        base_name = osp.splitext(file)[0]
                        image_files[base_name] = osp.join(root, file)
        
        logging.info(f"找到 {len(image_files)} 个图像文件")
        
        batch_size = 20  # 批处理大小
        
        with tqdm(total=len(mask_files), desc="处理数据", ncols=100,
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
            for i in range(0, len(mask_files), batch_size):
                batch_files = mask_files[i:i+batch_size]
                
                for mask_rel_path in batch_files:
                    try:
                        mask_file = osp.join(mask_dir, mask_rel_path)
                        base_name = osp.splitext(osp.basename(mask_file))[0]
                        
                        # 查找对应的图像文件
                        img_path = image_files.get(base_name)
                        
                        if img_path is None:
                            logging.warning(f"未找到对应的图像文件: {base_name}")
                            skipped_count += 1
                            pbar.update(1)
                            continue

                        # 读取图像尺寸
                        try:
                            with Image.open(img_path) as img:
                                width, height = img.size
                                if width <= 0 or height <= 0:
                                    logging.warning(f"无效的图像尺寸: {img_path}")
                                    skipped_count += 1
                                    pbar.update(1)
                                    continue
                        except Exception as e:
                            logging.error(f"读取图像失败 {img_path}: {str(e)}")
                            skipped_count += 1
                            pbar.update(1)
                            continue

                        # 读取和处理mask
                        try:
                            with Image.open(mask_file) as mask_img:
                                mask = np.array(mask_img)
                                processed_data = self.process_mask(mask, height, width, 
                                                                img_id=base_name)
                                if processed_data is not None:
                                    # 统计过滤前后的掩码数量
                                    original_masks = len(np.unique(mask)) - 1  # 减去背景
                                    remaining_masks = len(processed_data.get('instances', []))
                                    filtered_masks_count += (original_masks - remaining_masks)
                                    
                                    data_info = {
                                        'img_path': img_path,
                                        'img_id': base_name,
                                        'width': width,
                                        'height': height,
                                        **processed_data
                                    }
                                    data_list.append(data_info)
                                else:
                                    logging.warning(f"处理mask返回无效数据: {mask_file}")
                                    skipped_count += 1
                        except Exception as e:
                            logging.error(f"处理mask文件失败 {mask_file}: {str(e)}")
                            skipped_count += 1
                            pbar.update(1)
                            continue
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logging.error(f"处理文件出错 {mask_rel_path}: {str(e)}")
                        skipped_count += 1
                        pbar.update(1)
                        continue
                
                # 定期进行垃圾回收
                if i % (batch_size * 5) == 0:
                    gc.collect()

        if not data_list:
            raise ValueError("没有找到任何有效数据")

        elapsed_time = time.time() - start_time
        self.log_data_statistics(data_list)
        logging.info(f"数据处理完成，用时 {elapsed_time:.2f} 秒")
        logging.info(f"跳过的文件数: {skipped_count}")
        logging.info(f"因面积过小而过滤的掩码数量: {filtered_masks_count}")

        return data_list

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """获取指定索引的数据信息"""
        data_info = self.data_list[idx].copy()
        
        # 获取掩码文件路径，需要将图像路径中的扩展名替换为.png
        img_basename = os.path.basename(data_info['img_path'])
        img_name_without_ext = os.path.splitext(img_basename)[0]
        mask_file = os.path.join(self.data_root, self.mask_prefix, f"{img_name_without_ext}.png")
        
        try:
            mask = np.array(Image.open(mask_file))
            height, width = mask.shape[:2]
            
            # 处理掩码数据
            gt_instances = InstanceData()
            masks = []
            bboxes = []
            labels = []
            ignore_flags = []
            
            # 处理每个类别的掩码
            unique_labels = np.unique(mask)
            unique_labels = unique_labels[unique_labels != 0]  # 排除背景
            
            for label in unique_labels:
                binary_mask = (mask == label).astype(np.uint8)
                
                # 计算区域面积并过滤小区域
                area = np.sum(binary_mask)
                if area < self.min_mask_area:
                    continue
                
                masks.append(binary_mask)
                
                # 计算边界框
                contours, _ = cv2.findContours(binary_mask, 
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    bboxes.append([x, y, x + w, y + h])
                    labels.append(int(label - 1))  # 调整类别索引
                    ignore_flags.append(False)
            
            if masks:
                gt_instances.bboxes = torch.tensor(bboxes, dtype=torch.float32)
                gt_instances.labels = torch.tensor(labels, dtype=torch.int64)
                gt_instances.ignore_flags = torch.tensor(ignore_flags, dtype=torch.bool)
                data_info['gt_instances'] = gt_instances
                data_info['gt_masks'] = BitmapMasks(masks, height, width)
            else:
                # 创建默认实例
                default_data = self.create_default_instances(height, width)
                instance = default_data['instances'][0]
                gt_instances.bboxes = torch.tensor([instance['bbox']], dtype=torch.float32)
                gt_instances.labels = torch.tensor([instance['bbox_label']], dtype=torch.int64)
                gt_instances.ignore_flags = torch.tensor([instance['ignore_flag']], dtype=torch.bool)
                data_info['gt_instances'] = gt_instances
                
                # 创建默认掩码
                default_mask = np.zeros((height, width), dtype=np.uint8)
                data_info['gt_masks'] = BitmapMasks([default_mask], height, width)
                
            data_info['ori_shape'] = (height, width)
            return data_info
            
        except Exception as e:
            logging.error(f"处理掩码文件失败 {mask_file}: {str(e)}")
            # 返回默认实例
            height, width = data_info['height'], data_info['width']
            default_data = self.create_default_instances(height, width)
            instance = default_data['instances'][0]
            
            gt_instances = InstanceData()
            gt_instances.bboxes = torch.tensor([instance['bbox']], dtype=torch.float32)
            gt_instances.labels = torch.tensor([instance['bbox_label']], dtype=torch.int64)
            gt_instances.ignore_flags = torch.tensor([instance['ignore_flag']], dtype=torch.bool)
            
            data_info['gt_instances'] = gt_instances
            data_info['gt_masks'] = BitmapMasks([np.zeros((height, width), dtype=np.uint8)], height, width)
            data_info['ori_shape'] = (height, width)
            
            return data_info

    def filter_data(self) -> List[dict]:
        """根据filter_cfg过滤数据"""
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filtered_data_list = []
        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        for data_info in self.data_list:
            img_h, img_w = data_info['height'], data_info['width']
            
            # 检查图像尺寸
            if min_size is not None:
                if img_h < min_size or img_w < min_size:
                    continue
                    
            # 检查GT实例
            if filter_empty_gt:
                instances = data_info.get('instances', [])
                if not instances:
                    continue
                # 检查是否有有效的mask
                valid_instances = False
                for instance in instances:
                    if isinstance(instance.get('mask'), dict) and 'counts' in instance['mask']:
                        valid_instances = True
                        break
                if not valid_instances:
                    continue
                    
            filtered_data_list.append(data_info)

        return filtered_data_list

    def log_data_statistics(self, data_list: List[Dict[str, Any]]):
        """记录数据集统计信息"""
        total_samples = len(data_list)
        if total_samples == 0:
            logging.warning("数据集为空！")
            return
            
        valid_samples = 0
        total_instances = 0
        max_instances = 0
        total_mask_pixels = 0
        instance_areas = []  # 记录所有实例的面积
        
        for data in data_list:
            instances = data.get('instances', [])
            num_instances = len(instances)
            valid_instances = 0
            
            for instance in instances:
                if isinstance(instance.get('mask'), dict) and instance['mask'].get('counts') is not None:
                    valid_instances += 1
                    # 解码RLE并计算像素数
                    mask_array = maskUtils.decode(instance['mask'])
                    area = np.sum(mask_array)
                    total_mask_pixels += area
                    instance_areas.append(area)
            
            if valid_instances > 0:
                valid_samples += 1
                total_instances += valid_instances
                max_instances = max(max_instances, valid_instances)
        
        empty_samples = total_samples - valid_samples
        avg_instances = total_instances / total_samples if total_samples > 0 else 0
        avg_mask_pixels = total_mask_pixels / total_instances if total_instances > 0 else 0
        
        # 计算面积分布统计
        if instance_areas:
            min_area = min(instance_areas)
            max_area = max(instance_areas)
            median_area = np.median(instance_areas)
            q1_area = np.percentile(instance_areas, 25)
            q3_area = np.percentile(instance_areas, 75)
        else:
            min_area = max_area = median_area = q1_area = q3_area = 0
        
        logging.info(f"""
        数据集统计信息:
        - 总样本数: {total_samples}
        - 有效样本数: {valid_samples}
        - 空样本数: {empty_samples}
        - 有效样本比例: {(valid_samples/total_samples)*100:.2f}%
        - 平均实例数: {avg_instances:.2f}
        - 最大实例数: {max_instances}
        - 平均mask像素数: {avg_mask_pixels:.2f}
        - 最小面积: {min_area}
        - 最大面积: {max_area}
        - 中位数面积: {median_area}
        - 25%分位面积: {q1_area}
        - 75%分位面积: {q3_area}
        - 面积过滤阈值: {self.min_mask_area}
        """)
        
        if empty_samples > 0:
            logging.warning(f"发现 {empty_samples} 个空样本，请检查数据集")