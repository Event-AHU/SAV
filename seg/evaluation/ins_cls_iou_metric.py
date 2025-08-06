import torch
from mmengine.dist import broadcast_object_list, collect_results, is_main_process
from typing import Dict, Optional, Sequence, List
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS
from mmengine.evaluator.metric import _to_cpu

@METRICS.register_module()
class InsClsIoUMetric(BaseMetric):
    def __init__(
            self,
            collect_device: str = 'cpu',
            prefix: Optional[str] = None,
            base_classes=None,
            novel_classes=None,
            with_score=True,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.with_score = with_score
        
        # Store classes info
        self.base_classes = base_classes
        self.novel_classes = novel_classes
        
        if base_classes is not None:
            assert novel_classes is not None
            max_class_id = max(max(base_classes) + 1, max(novel_classes) + 1)
            self.base_novel_indicator = torch.zeros((max_class_id,), dtype=torch.long)
            for clss in base_classes:
                self.base_novel_indicator[clss] = 1
            for clss in novel_classes:
                self.base_novel_indicator[clss] = 2
        else:
            self.base_novel_indicator = None

    def get_iou(self, gt_mask, pred_mask):
        """Calculate IoU between ground truth and predicted masks"""
        if gt_mask.dim() == 2:
            gt_mask = gt_mask.unsqueeze(0)
        if pred_mask.dim() == 2:
            pred_mask = pred_mask.unsqueeze(0)
        
        # Convert masks to boolean type for binary operations
        gt_mask = gt_mask.bool()
        pred_mask = pred_mask.bool()
        
        n, h, w = gt_mask.shape
        intersection = (gt_mask & pred_mask).reshape(n, h * w).sum(dim=-1)
        union = (gt_mask | pred_mask).reshape(n, h * w).sum(dim=-1)
        iou = (intersection.float() / (union.float() + 1.e-8))
        return iou

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        The processed results should be stored in self.results, which will be
        used to compute the metrics when all batches have been processed.
        
        Args:
            data_batch: A batch of data from the dataloader.
            data_samples: A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # Skip if no ground truth instances
            if 'gt_instances' not in data_sample or len(data_sample['gt_instances']['labels']) == 0:
                continue
                
            result_dict = {}
            
            # Get ground truth data
            gt_labels = data_sample['gt_instances']['labels']
            gt_masks = data_sample['gt_instances']['masks'].to_tensor(
                dtype=torch.bool, device=gt_labels.device)
            
            # Get predicted data
            pred_labels = data_sample['pred_instances']['labels']
            pred_masks = data_sample['pred_instances']['masks']
            
            # --- INSTANCE-LEVEL IoU CALCULATION ---
            ious_list = []
            for i in range(len(gt_masks)):
                if i < len(pred_masks):  # Make sure we have a prediction for this instance
                    gt_mask = gt_masks[i]
                    pred_mask = pred_masks[i]
                    iou = self.get_iou(gt_mask, pred_mask)
                    ious_list.append(iou)
            
            if ious_list:
                ious = torch.cat(ious_list)
                result_dict['iou'] = ious.cpu()
            
                # Calculate base/novel IoU
                if self.base_novel_indicator is not None:
                    base_mask = torch.tensor([self.base_novel_indicator[label.item()] == 1 
                                            for label in gt_labels[:len(ious)]])
                    novel_mask = torch.tensor([self.base_novel_indicator[label.item()] == 2 
                                            for label in gt_labels[:len(ious)]])
                    
                    if base_mask.any():
                        result_dict['base_iou'] = ious[base_mask].cpu()
                    
                    if novel_mask.any():
                        result_dict['novel_iou'] = ious[novel_mask].cpu()
            
            # --- CLASSIFICATION SCORE (instance-level) ---
            if self.with_score:
                if len(gt_labels) > 0 and len(pred_labels) > 0:
                    min_len = min(len(gt_labels), len(pred_labels))
                    score = (pred_labels[:min_len] == gt_labels[:min_len]).float() * 100
                    result_dict['score'] = score.cpu()
                    
                    # Base/novel classification score
                    if self.base_novel_indicator is not None:
                        base_mask = torch.tensor([self.base_novel_indicator[label.item()] == 1 
                                                for label in gt_labels[:min_len]])
                        novel_mask = torch.tensor([self.base_novel_indicator[label.item()] == 2 
                                                for label in gt_labels[:min_len]])
                        
                        if base_mask.any():
                            result_dict['base_score'] = score[base_mask].cpu()
                        
                        if novel_mask.any():
                            result_dict['novel_score'] = score[novel_mask].cpu()
            
            # --- PIXEL-WISE ACCURACY CALCULATION (semantic-level) ---
            h, w = gt_masks.shape[-2:]
            
            # Initialize with background (class 0)
            background_class = 0
            gt_semantic = torch.full((h, w), background_class, 
                                    dtype=torch.long, device=gt_masks.device)
            pred_semantic = torch.full((h, w), background_class, 
                                    dtype=torch.long, device=pred_masks.device)
            
            # Fill in class labels (later instances overwrite earlier ones)
            for i, label in enumerate(gt_labels):
                if i < len(gt_masks):
                    gt_semantic[gt_masks[i]] = label
            
            for i, label in enumerate(pred_labels):
                if i < len(pred_masks):
                    pred_semantic[pred_masks[i]] = label
            
            # Calculate per-class pixel metrics
            unique_classes = torch.unique(gt_semantic).cpu().numpy()
            class_metrics = {}
            
            for c in unique_classes:
                class_id = int(c)
                mask = (gt_semantic == class_id)
                total = mask.sum().item()
                
                if total > 0:
                    correct = ((gt_semantic == class_id) & (pred_semantic == class_id)).sum().item()
                    class_metrics[class_id] = {
                        'correct': correct,
                        'total': total,
                        'is_base': self.base_novel_indicator is not None and 
                                  class_id < len(self.base_novel_indicator) and 
                                  self.base_novel_indicator[class_id] == 1,
                        'is_novel': self.base_novel_indicator is not None and 
                                   class_id < len(self.base_novel_indicator) and 
                                   self.base_novel_indicator[class_id] == 2
                    }
            
            result_dict['pixel_metrics'] = class_metrics
            
            # Store results for this sample
            self.results.append(result_dict)

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """Compute the metrics from processed results.
        
        Args:
            results: The processed results of each batch.
            
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of the
            metrics, and the values are corresponding results.
        """
        # Initialize metrics
        metrics = {}
        
        # Collect all instance-level IoUs
        all_ious = []
        all_base_ious = []
        all_novel_ious = []
        all_scores = []
        all_base_scores = []
        all_novel_scores = []
        
        # Collect all pixel-level metrics
        all_pixel_metrics = {}
        
        # Process results
        for result in results:
            # Collect IoUs
            if 'iou' in result:
                all_ious.append(result['iou'])
            
            if 'base_iou' in result:
                all_base_ious.append(result['base_iou'])
                
            if 'novel_iou' in result:
                all_novel_ious.append(result['novel_iou'])
            
            # Collect scores
            if 'score' in result:
                all_scores.append(result['score'])
                
            if 'base_score' in result:
                all_base_scores.append(result['base_score'])
                
            if 'novel_score' in result:
                all_novel_scores.append(result['novel_score'])
            
            # Collect pixel metrics
            if 'pixel_metrics' in result:
                pixel_metrics = result['pixel_metrics']
                for class_id, class_metric in pixel_metrics.items():
                    if class_id not in all_pixel_metrics:
                        all_pixel_metrics[class_id] = {
                            'correct': 0,
                            'total': 0,
                            'is_base': class_metric['is_base'],
                            'is_novel': class_metric['is_novel']
                        }
                    
                    all_pixel_metrics[class_id]['correct'] += class_metric['correct']
                    all_pixel_metrics[class_id]['total'] += class_metric['total']
        
        # Calculate IoU metrics
        if all_ious:
            all_ious_tensor = torch.cat(all_ious)
            metrics['miou'] = all_ious_tensor.mean().item()
        else:
            metrics['miou'] = 0.0
        
        # Base/novel IoU
        if all_base_ious:
            base_ious_tensor = torch.cat(all_base_ious)
            metrics['base_iou'] = base_ious_tensor.mean().item()
        else:
            metrics['base_iou'] = 0.0
            
        if all_novel_ious:
            novel_ious_tensor = torch.cat(all_novel_ious)
            metrics['novel_iou'] = novel_ious_tensor.mean().item()
        else:
            metrics['novel_iou'] = 0.0
        
        # Classification metrics
        if all_scores:
            all_scores_tensor = torch.cat(all_scores)
            metrics['score'] = all_scores_tensor.mean().item()
        else:
            metrics['score'] = 0.0
            
        if all_base_scores:
            base_scores_tensor = torch.cat(all_base_scores)
            metrics['base_score'] = base_scores_tensor.mean().item()
        else:
            metrics['base_score'] = 0.0
            
        if all_novel_scores:
            novel_scores_tensor = torch.cat(all_novel_scores)
            metrics['novel_score'] = novel_scores_tensor.mean().item()
        else:
            metrics['novel_score'] = 0.0
        
        # Calculate pixel-level metrics
        if all_pixel_metrics:
            # Overall metrics
            total_correct = sum(m['correct'] for m in all_pixel_metrics.values())
            total_pixels = sum(m['total'] for m in all_pixel_metrics.values())
            metrics['pixel_acc'] = total_correct / total_pixels if total_pixels > 0 else 0.0
            
            # Class-wise metrics
            class_accs = []
            base_class_accs = []
            novel_class_accs = []
            
            for class_id, class_metric in all_pixel_metrics.items():
                if class_metric['total'] > 0:
                    acc = class_metric['correct'] / class_metric['total']
                    class_accs.append(acc)
                    
                    if class_metric['is_base']:
                        base_class_accs.append(acc)
                    elif class_metric['is_novel']:
                        novel_class_accs.append(acc)
            
            metrics['macc'] = sum(class_accs) / len(class_accs) if class_accs else 0.0
            
            # Base/novel pixel metrics
            if base_class_accs:
                metrics['base_macc'] = sum(base_class_accs) / len(base_class_accs)
            else:
                metrics['base_macc'] = 0.0
                
            if novel_class_accs:
                metrics['novel_macc'] = sum(novel_class_accs) / len(novel_class_accs)
            else:
                metrics['novel_macc'] = 0.0
            
            # Base/novel pixel accuracy
            base_correct = sum(m['correct'] for m in all_pixel_metrics.values() if m['is_base'])
            base_total = sum(m['total'] for m in all_pixel_metrics.values() if m['is_base'])
            novel_correct = sum(m['correct'] for m in all_pixel_metrics.values() if m['is_novel'])
            novel_total = sum(m['total'] for m in all_pixel_metrics.values() if m['is_novel'])
            
            metrics['base_pixel_acc'] = base_correct / base_total if base_total > 0 else 0.0
            metrics['novel_pixel_acc'] = novel_correct / novel_total if novel_total > 0 else 0.0
        else:
            metrics['pixel_acc'] = 0.0
            metrics['macc'] = 0.0
            metrics['base_macc'] = 0.0
            metrics['novel_macc'] = 0.0
            metrics['base_pixel_acc'] = 0.0
            metrics['novel_pixel_acc'] = 0.0
        
        return metrics