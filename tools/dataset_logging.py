import logging
import sys
import os
from datetime import datetime

def setup_logging(log_dir='logs'):
    """设置日志配置"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'dataset_processing_{timestamp}.log')
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

def log_mask_info(mask, img_id='', prefix=''):
    """记录mask的详细信息"""
    if mask is None:
        logging.debug(f"{prefix}Mask is None for {img_id}")
        return
        
    try:
        unique_values = np.unique(mask)
        mask_shape = mask.shape
        non_zero = np.count_nonzero(mask)
        min_val = mask.min()
        max_val = mask.max()
        
        logging.debug(
            f"{prefix}Mask info for {img_id}:\n"
            f"  Shape: {mask_shape}\n"
            f"  Unique values: {unique_values}\n"
            f"  Non-zero pixels: {non_zero}\n"
            f"  Value range: [{min_val}, {max_val}]"
        )
    except Exception as e:
        logging.error(f"{prefix}Failed to log mask info: {str(e)}")

def log_instance_info(instances, img_id=''):
    """记录实例数据的详细信息"""
    try:
        logging.debug(
            f"Instance info for {img_id}:\n"
            f"  Number of masks: {len(instances['masks'])}\n"
            f"  Number of bboxes: {len(instances['bboxes'])}\n"
            f"  Number of labels: {len(instances['labels'])}\n"
            f"  Mask shape: {instances['mask_shape']}"
        )
        
        # 记录每个实例的详细信息
        for i, (mask, bbox, label) in enumerate(zip(
            instances['masks'], 
            instances['bboxes'], 
            instances['labels']
        )):
            logging.debug(
                f"  Instance {i}:\n"
                f"    Label: {label}\n"
                f"    Bbox: {bbox}\n"
                f"    Mask shape: {mask.shape if mask is not None else 'None'}\n"
                f"    Mask non-zero pixels: {np.count_nonzero(mask) if mask is not None else 0}"
            )
    except Exception as e:
        logging.error(f"Failed to log instance info: {str(e)}")

# 修改数据集类中的方法
def get_data_info(self, idx: int) -> Dict[str, Any]:
    """获取单个数据样本的信息"""
    try:
        logging.debug(f"Processing sample index: {idx}")
        data_info = self.data_list[idx].copy()
        img_id = data_info.get('img_id', 'unknown')
        
        logging.debug(f"Processing image {img_id} (idx: {idx})")
        instances = data_info.pop('instances')
        height, width = instances['mask_shape']
        
        logging.debug(f"Image dimensions - height: {height}, width: {width}")
        
        # 记录原始实例信息
        logging.debug("Original instance data:")
        log_instance_info(instances, img_id)
        
        # 验证所有mask的有效性
        valid_masks = []
        valid_bboxes = []
        valid_labels = []
        
        # 首先尝试验证现有的masks
        if isinstance(instances['masks'], list):
            for i, mask in enumerate(instances['masks']):
                logging.debug(f"Validating mask {i} for {img_id}")
                if mask is None:
                    logging.warning(f"Mask {i} is None for {img_id}")
                    continue
                    
                log_mask_info(mask, img_id, f"Mask {i}: ")
                
                if mask.shape != (height, width):
                    logging.warning(
                        f"Invalid mask shape for {img_id} - "
                        f"Expected: ({height}, {width}), Got: {mask.shape}"
                    )
                    continue
                
                if not np.any(mask):
                    logging.warning(f"Empty mask found for {img_id} at index {i}")
                    continue
                    
                valid_masks.append(mask)
                valid_bboxes.append(instances['bboxes'][i])
                valid_labels.append(instances['labels'][i])
                logging.debug(f"Mask {i} is valid for {img_id}")
        
        # 记录有效mask的数量
        logging.info(
            f"Valid masks for {img_id}: {len(valid_masks)} out of "
            f"{len(instances['masks'])} original masks"
        )
        
        # 如果没有有效的mask，创建默认mask
        if not valid_masks:
            logging.warning(
                f"No valid masks for {img_id} (idx: {idx}), creating default mask"
            )
            default_instance = self.create_default_mask(height, width)
            valid_masks = default_instance['masks']
            valid_bboxes = default_instance['bboxes']
            valid_labels = default_instance['labels']
        
        # 转换为numpy数组
        try:
            valid_bboxes = np.array(valid_bboxes, dtype=np.float32)
            valid_labels = np.array(valid_labels, dtype=np.int64)
            logging.debug(
                f"Arrays converted successfully for {img_id}:\n"
                f"  Bboxes shape: {valid_bboxes.shape}\n"
                f"  Labels shape: {valid_labels.shape}"
            )
        except Exception as e:
            logging.error(f"Array conversion failed for {img_id}: {str(e)}")
            default_instance = self.create_default_mask(height, width)
            valid_masks = default_instance['masks']
            valid_bboxes = default_instance['bboxes']
            valid_labels = default_instance['labels']
        
        # 创建BitmapMasks对象
        try:
            bitmap_masks = BitmapMasks(valid_masks, height=height, width=width)
            logging.debug(
                f"BitmapMasks created successfully for {img_id} with "
                f"{len(bitmap_masks.masks)} masks"
            )
        except Exception as e:
            logging.error(f"BitmapMasks creation failed for {img_id}: {str(e)}")
            default_instance = self.create_default_mask(height, width)
            bitmap_masks = BitmapMasks(
                default_instance['masks'],
                height=height,
                width=width
            )
        
        # 确保创建的BitmapMasks不为空
        if len(bitmap_masks.masks) == 0:
            logging.error(f"Empty BitmapMasks for {img_id}, creating default mask")
            default_instance = self.create_default_mask(height, width)
            bitmap_masks = BitmapMasks(
                default_instance['masks'],
                height=height,
                width=width
            )
            valid_bboxes = default_instance['bboxes']
            valid_labels = default_instance['labels']
        
        # 转换为tensor
        try:
            bboxes = torch.from_numpy(valid_bboxes)
            labels = torch.from_numpy(valid_labels)
            logging.debug(
                f"Tensors created successfully for {img_id}:\n"
                f"  Bboxes tensor shape: {bboxes.shape}\n"
                f"  Labels tensor shape: {labels.shape}"
            )
        except Exception as e:
            logging.error(f"Tensor conversion failed for {img_id}: {str(e)}")
            default_instance = self.create_default_mask(height, width)
            bboxes = torch.from_numpy(default_instance['bboxes'])
            labels = torch.from_numpy(default_instance['labels'])
        
        # 创建InstanceData对象
        data_info['gt_instances'] = InstanceData(
            bboxes=bboxes,
            labels=labels,
            masks=bitmap_masks
        )
        
        logging.debug(
            f"Successfully created InstanceData for {img_id} with "
            f"{len(bitmap_masks.masks)} masks"
        )
        
        return data_info
            
    except Exception as e:
        logging.error(f"Error processing index {idx}: {str(e)}", exc_info=True)
        # 创建一个有效的默认实例
        height = data_info.get('height', 100)
        width = data_info.get('width', 100)
        default_instance = self.create_default_mask(height, width)
        
        data_info['gt_instances'] = InstanceData(
            bboxes=torch.from_numpy(default_instance['bboxes']),
            labels=torch.from_numpy(default_instance['labels']),
            masks=BitmapMasks(
                default_instance['masks'],
                height=height,
                width=width
            )
        )
        return data_info