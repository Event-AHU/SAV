import copy
import os
from typing import Literal, Tuple, List, Optional

import torch
from mmcv.cnn import ConvModule
from mmdet.structures.bbox import bbox2roi
from mmdet.structures.mask import mask2bbox
from torch import nn
import torch.nn.functional as F
from mmcv.ops import point_sample
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.structures import SampleList
from mmdet.utils import reduce_mean
from mmengine import MMLogger
from mmengine.model import BaseModule
from mmdet.registry import MODELS, TASK_UTILS
from mmengine.structures import InstanceData

from ext.sam import MaskDecoder
from ext.sam.mask_decoder import MLP as SAMMLP
from ext.meta.sam_meta import meta_dict, checkpoint_dict
from seg.models.utils.load_checkpoint import load_checkpoint_with_prefix


@MODELS.register_module()
class OVSAMHead(BaseModule):
    """Modified SAM head for open-vocabulary segmentation without prompts."""
    
    def __init__(
            self,
            model_name: Literal['vit_h', 'vit_l', 'vit_b'] = 'vit_h',
            with_label_token: bool = False,
            ov_classifier_name: Optional[str] = None,
            logit: Optional[float] = None,
            roi_extractor=None,
            fix: bool = True,
            init_cfg=None,
            loss_cls=None,
            loss_mask=None,
            loss_dice=None,
            cur_mask=14,
            load_roi_conv=None,
            gen_box=False,
    ):
        assert init_cfg is not None and \
            init_cfg['type'] in ['sam_pretrain', 'Pretrained'], f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__(init_cfg=None)
        self.init_cfg = init_cfg
        self.logger = MMLogger.get_current_instance()
        
        # Store actual number of classes
        self.num_classes = cur_mask - 1

        # Initialize mask decoder with proper number of outputs
        mask_decoder = MaskDecoder(
            num_multimask_outputs=self.num_classes,  # 使用 self.num_classes
            transformer_dim=meta_dict[model_name]['prompt_embed_dim'],
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            with_iou=False
        )

        # Load SAM pretrained weights if specified
        if self.init_cfg['type'] == 'sam_pretrain':
            checkpoint_path = checkpoint_dict[pretrained]
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix='mask_decoder')
            original_mask_tokens = state_dict['mask_tokens.weight']
            original_dim = original_mask_tokens.size(1)
            num_original_masks = original_mask_tokens.size(0)
            
            # Create new state dict for modified architecture
            new_state_dict = {}
            
            # Copy transformer and output_upscaling weights
            for name, param in state_dict.items():
                if name.startswith('transformer.') or name.startswith('output_upscaling.'):
                    new_state_dict[name] = param
            
            # Initialize mask tokens
            new_mask_tokens = torch.empty(cur_mask, original_dim)
            if num_original_masks <= cur_mask:
                # Copy existing tokens and initialize remaining ones
                new_mask_tokens[:num_original_masks] = original_mask_tokens
                if num_original_masks < cur_mask:
                    nn.init.normal_(new_mask_tokens[num_original_masks:], mean=0, std=0.02)
            else:
                # If we need fewer tokens, just take the first ones we need
                new_mask_tokens = original_mask_tokens[:cur_mask]
            new_state_dict['mask_tokens.weight'] = new_mask_tokens
            
            # Handle hypernetworks MLPs
            for i in range(cur_mask):
                if i < num_original_masks:
                    # Copy original weights for existing classes
                    for name, param in state_dict.items():
                        if name.startswith(f'output_hypernetworks_mlps.{i}.'):
                            new_state_dict[name] = param
                else:
                    # Initialize new weights for additional classes
                    for j in range(3):
                        if j == 0 or j == 1:
                            in_dim = original_dim
                            out_dim = original_dim
                        else:
                            in_dim = original_dim
                            mlp_out_features = mask_decoder.output_hypernetworks_mlps[0].layers[2].out_features
                            out_dim = mlp_out_features
                        
                        weight = torch.empty(out_dim, in_dim)
                        bias = torch.empty(out_dim)
                        nn.init.xavier_uniform_(weight)
                        nn.init.zeros_(bias)
                        
                        new_state_dict[f'output_hypernetworks_mlps.{i}.layers.{j}.weight'] = weight
                        new_state_dict[f'output_hypernetworks_mlps.{i}.layers.{j}.bias'] = bias
            
            # Load modified state dict
            mask_decoder.load_state_dict(new_state_dict, strict=True)

        self.mask_decoder = mask_decoder
        self.with_label_token = with_label_token

        if self.with_label_token:
            ov_path = os.path.join(os.path.expanduser('~/.cache/embd'), f"{ov_classifier_name}.pth")
            cls_embed = torch.load(ov_path)
            cls_embed_norm = cls_embed.norm(p=2, dim=-1)
            assert torch.allclose(cls_embed_norm, torch.ones_like(cls_embed_norm))

            _dim = cls_embed.size(2)
            _prototypes = cls_embed.size(1)
            back_token = torch.zeros(1, _dim, dtype=torch.float32, device='cpu')
            cls_embed = torch.cat([
                cls_embed, back_token.repeat(_prototypes, 1)[None]
            ], dim=0)
            self.register_buffer('cls_embed', cls_embed.permute(2, 0, 1).contiguous(), persistent=False)

            if logit is None:
                logit_scale = torch.tensor(4.6052, dtype=torch.float32)
            else:
                logit_scale = torch.tensor(logit, dtype=torch.float32)
            self.register_buffer('logit_scale', logit_scale, persistent=False)

            transformer_dim = self.mask_decoder.mask_tokens.weight.shape[1]
            self.label_token = nn.Embedding(1, transformer_dim)
            self.label_mlp = SAMMLP(transformer_dim, transformer_dim, _dim, 3)

            if loss_cls is not None:
                _loss_cls = copy.deepcopy(loss_cls)
                _loss_cls.update(class_weight=[1.] * (self.cls_embed.shape[1] - 1) + [.1])
                self.loss_cls = MODELS.build(_loss_cls)
                self.register_buffer('class_weight', torch.tensor(self.loss_cls.class_weight), persistent=False)
            else:
                self.loss_cls = None

            if loss_mask is not None:
                self.loss_mask = MODELS.build(loss_mask)
            else:
                self.loss_mask = None

            if loss_dice is not None:
                self.loss_dice = MODELS.build(loss_dice)
            else:
                self.loss_dice = None

        # Meta parameters
        self.num_points = 12544
        self.oversample_ratio = 3.
        self.importance_sample_ratio = .75
        self.gen_box = gen_box

        if roi_extractor is not None:
            self.roi = MODELS.build(roi_extractor)
            self.roi_conv = nn.Sequential(*[
                ConvModule(in_channels=self.roi.out_channels, out_channels=_dim, kernel_size=1, bias=False)
            ])
        else:
            self.roi = None

        # Load pretrained weights if specified
        if self.init_cfg['type'] == 'Pretrained':
            checkpoint_path = pretrained
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=self.init_cfg['prefix'])
            self.load_state_dict(state_dict, strict=False)

        # Load ROI conv weights if specified
        if roi_extractor is not None and load_roi_conv is not None:
            checkpoint_path = load_roi_conv['checkpoint']
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=load_roi_conv['prefix'])
            self.roi_conv.load_state_dict(state_dict, strict=True)

        self.fix = fix
        if self.fix:
            self.train(mode=False)
            for name, param in self.named_parameters():
                param.requires_grad = False

    def forward_logit(self, cls_embd):
        """Calculate classification logits."""
        cls_pred = torch.einsum('bnc,ckp->bnkp', F.normalize(cls_embd, dim=-1), self.cls_embed)
        cls_pred = cls_pred.max(-1).values
        cls_pred = self.logit_scale.exp() * cls_pred
        return cls_pred

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            fpn_feats: List[torch.Tensor],
            roi_list: Optional[List[torch.Tensor]],
            backbone_feature: torch.Tensor,
            backbone=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict masks for all classes."""
        num_instances = int(sparse_prompt_embeddings.size(0))
        
        # Concatenate output tokens for all classes
        output_tokens = torch.cat([
            self.label_token.weight.repeat(self.num_classes, 1),
            self.mask_decoder.mask_tokens.weight
        ], dim=0)
        
        output_tokens = output_tokens.unsqueeze(0).expand(num_instances, -1, -1)
        queries = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Process image embeddings
        image_embeddings = image_embeddings + dense_prompt_embeddings
        pos_img = torch.repeat_interleave(image_pe, num_instances, dim=0)
        b, c, h, w = image_embeddings.shape

        # Run transformer
        queries, mask_feats = self.mask_decoder.transformer(image_embeddings, pos_img, queries)
        
        # Separate queries for each class
        label_queries = queries[:, :self.num_classes, :]
        mask_embeds = queries[:, self.num_classes:(self.num_classes + self.num_classes), :]

        # Upscale mask embeddings and predict masks
        mask_feats = mask_feats.transpose(1, 2).view(b, c, h, w)
        mask_feats = self.mask_decoder.output_upscaling(mask_feats)
        
        # Generate mask for each class
        mask_queries_list: List[torch.Tensor] = []
        for i in range(self.num_classes):
            mask_queries_list.append(self.mask_decoder.output_hypernetworks_mlps[i](mask_embeds[:, i, :]))
        mask_queries = torch.stack(mask_queries_list, dim=1)
        
        b, c, h, w = mask_feats.shape
        masks = (mask_queries @ mask_feats.view(b, c, h * w)).view(b, -1, h, w)

        # Generate class labels
        if self.with_label_token:
            cls_embed_list = []
            for i in range(self.num_classes):
                cls_embed_list.append(self.label_mlp(label_queries[:, i, :]))
            cls_embed = torch.stack(cls_embed_list, dim=1)
            
            if self.gen_box:
                import numpy as np
                from skimage import measure
                from torchvision.ops import nms
                
                # Convert masks to probabilities
                mask_probs = masks.sigmoid()
                
                # Use only the first image copy since all are the same
                first_image_masks = mask_probs[0]  # Shape: [13, 256, 256]
                
                # Define threshold for considering a mask valid
                threshold = 0.5
                
                # Lists to store all bounding boxes, scores, and class indices
                all_boxes = []
                all_scores = []
                all_class_indices = []
                
                # For each class mask, find connected components and generate boxes
                for class_idx in range(first_image_masks.shape[0]):
                    class_mask = first_image_masks[class_idx]
                    mask_binary = (class_mask > threshold).cpu().numpy().astype(np.uint8)
                    
                    # Skip if no pixels are above threshold
                    if not np.any(mask_binary):
                        continue
                    
                    # Find connected components
                    labeled_mask, num_components = measure.label(mask_binary, return_num=True, connectivity=2)
                    
                    # For each connected component, generate a bounding box
                    for component_idx in range(1, num_components + 1):
                        # Extract this component
                        component_mask = (labeled_mask == component_idx)
                        
                        # Skip very small components (optional, adjust as needed)
                        if np.sum(component_mask) < 20:
                            continue
                        
                        # Create a mask tensor for this component only
                        component_mask_tensor = torch.from_numpy(component_mask).to(masks.device).unsqueeze(0)
                        
                        # Generate bounding box
                        try:
                            bbox = mask2bbox(component_mask_tensor)[0] * 4  # Scale by 4
                            
                            # Calculate confidence score for this region
                            region_mask = torch.from_numpy(component_mask).to(class_mask.device)
                            conf = (class_mask * region_mask).sum() / region_mask.sum()
                            
                            all_boxes.append(bbox)
                            all_scores.append(conf.item())
                            all_class_indices.append(class_idx)
                        except Exception as e:
                            # Skip this component if bbox generation fails
                            continue
                
                # Convert lists to tensors for NMS
                if all_boxes:
                    boxes_tensor = torch.stack(all_boxes)
                    scores_tensor = torch.tensor(all_scores, device=boxes_tensor.device)
                    
                    # Apply NMS to remove duplicate boxes
                    # You may need to adjust the IoU threshold
                    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
                    
                    # Filter boxes, scores and class indices
                    filtered_boxes = boxes_tensor[keep_indices]
                    filtered_class_indices = [all_class_indices[i] for i in keep_indices.cpu().numpy()]
                    
                    # Limit to num_instances if needed
                    if len(filtered_boxes) > num_instances:
                        # Sort by score and take top num_instances
                        sorted_indices = torch.argsort(scores_tensor[keep_indices], descending=True)[:num_instances]
                        filtered_boxes = filtered_boxes[sorted_indices]
                        filtered_class_indices = [filtered_class_indices[i] for i in sorted_indices.cpu().numpy()]
                    
                    # Create ROI list
                    roi_list = bbox2roi([filtered_boxes])
                    
                    # Store class indices for later use
                    self.filtered_class_indices = filtered_class_indices
                else:
                    # No valid boxes found
                    filtered_boxes = torch.zeros((0, 4), device=masks.device)
                    self.filtered_class_indices = []
                    roi_list = bbox2roi([filtered_boxes])
            
            # Extract ROI features
            roi_feats = self.roi(fpn_feats, roi_list)
            roi_feats = self.roi_conv(roi_feats)
            roi_feats = roi_feats.mean(dim=-1).mean(dim=-1)
            
            # Combine with class embeddings
            if self.gen_box and hasattr(self, 'filtered_class_indices') and self.filtered_class_indices:
                # Use class embeddings that correspond to our detected objects
                selected_cls_embed = cls_embed[0, self.filtered_class_indices]
                roi_feats = roi_feats[:, None] + selected_cls_embed
            else:
                # Original approach for non-gen_box case
                roi_feats = roi_feats[:, None] + cls_embed
                
            cls_pred = self.forward_logit(roi_feats)
        else:
            cls_pred = None
            
        return masks, None, cls_pred

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multi_mask_output: bool,
            data_samples=None,
            fpn_feats=None,
            backbone_feats=None,
            backbone=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Forward pass."""
        # For inference, we assume one image with multiple prompts
        batch_size = image_embeddings.size(0)
        num_prompts = len(sparse_prompt_embeddings)
        
        # Prepare image embeddings for each prompt
        image_embed_list = []
        for idx in range(batch_size):
            image_embed_list.append(
                torch.repeat_interleave(image_embeddings[idx:idx + 1], num_prompts, dim=0)
            )
        image_embed = torch.cat(image_embed_list, dim=0)

        # Get predictions using the same embedding as both image and sparse prompt embeddings
        masks, iou_pred, cls_pred = self.predict_masks(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,  # Using image embedding as sparse prompt
            dense_prompt_embeddings=dense_prompt_embeddings,
            fpn_feats=fpn_feats,
            roi_list=bbox2roi([itm.gt_instances.bboxes for itm in [data_samples]]) if data_samples is not None and hasattr(data_samples, 'gt_instances') and hasattr(data_samples.gt_instances, 'bboxes') else None,
            backbone_feature=backbone_feats,
            backbone=backbone,
        )

        return masks, iou_pred, cls_pred

    def forward_train(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        batch_ind_list: List[int],
        data_samples: SampleList,
        fpn_feats=None,
        backbone_feats=None,
        backbone=None,
    ):
        """Modified forward_train to ensure all parameters contribute to loss calculation."""
        # Prepare image embeddings
        image_embed_list = []
        for idx, num_ins in enumerate(batch_ind_list):
            image_embed_list.append(
                torch.repeat_interleave(image_embeddings[idx:idx + 1], num_ins, dim=0)
            )
        image_embed = torch.cat(image_embed_list, dim=0)

        # Get predictions
        masks, iou_pred, cls_preds = self.predict_masks(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            fpn_feats=fpn_feats,
            roi_list=bbox2roi([itm.gt_instances.bboxes for itm in data_samples]),
            backbone_feature=backbone_feats,
            backbone=backbone,
        )

        # Prepare ground truth instances
        instances = []
        for data_sample in data_samples:
            if 'masks' in data_sample.gt_instances:
                instances.append(
                    InstanceData(
                        labels=data_sample.gt_instances.labels,
                        masks=data_sample.gt_instances.masks
                    )
                )
            else:
                instances.append(
                    InstanceData(
                        labels=data_sample.gt_instances.labels,
                    )
                )
        gt_instances = InstanceData.cat(instances)
        
        device = image_embed.device
        batch_size = len(batch_ind_list)
        num_proposals = masks.size(0) // batch_size
        
        # Create multi-label ground truth
        gt_labels_multi = torch.zeros(
            (masks.size(0), self.num_classes),
            dtype=torch.float32,
            device=device
        )
        
        # Fill multi-label ground truth
        start_idx = 0
        for idx, num_ins in enumerate(batch_ind_list):
            sample_labels = gt_instances.labels[start_idx:start_idx + num_ins]
            for proposal_idx in range(num_proposals):
                if proposal_idx < len(sample_labels):
                    label = sample_labels[proposal_idx]
                    if label != -1:
                        gt_labels_multi[idx * num_proposals + proposal_idx, label] = 1.0
            start_idx += num_ins

        # Classification loss with regularization to ensure all parameters are used
        cls_scores = cls_preds.mean(dim=1)  # [B*N, C+1]
        gt_labels_with_bg = torch.zeros(
            (gt_labels_multi.size(0), self.num_classes + 1),
            dtype=torch.float32,
            device=device
        )
        gt_labels_with_bg[:, :-1] = gt_labels_multi
        gt_labels_with_bg[:, -1] = 1 - gt_labels_multi.any(dim=1).float()
        
        valid_mask = torch.ones_like(gt_labels_with_bg[:, 0])
        
        # Add regularization term to ensure all parameters are used
        reg_term = 0.0
        for param in self.parameters():
            reg_term = reg_term + param.mean() * 0.0
        
        loss_cls = self.loss_cls(
            cls_scores,
            gt_labels_with_bg,
            valid_mask,
            avg_factor=valid_mask.sum().clamp(min=1.0)
        ) + reg_term

        # Mask loss computation
        loss_dice = 0.0
        loss_mask = 0.0
        
        if 'masks' in gt_instances:
            gt_masks = gt_instances.masks.to_tensor(dtype=torch.float, device=device)
            
            for class_idx in range(self.num_classes):
                # Get predictions and ground truth for current class
                pred_masks_class = masks[:, class_idx:class_idx + 1]  # [B*N, 1, H, W]
                class_mask = gt_labels_multi[:, class_idx] > 0
                
                if class_mask.sum() > 0:
                    # Prepare ground truth masks
                    gt_masks_class = gt_masks[class_mask].unsqueeze(1)  # [M, 1, H, W]
                    pred_masks_class = pred_masks_class[class_mask]  # [M, 1, H, W]
                    
                    # Resize ground truth if needed
                    if gt_masks_class.shape[-2:] != pred_masks_class.shape[-2:]:
                        gt_masks_class = F.interpolate(
                            gt_masks_class,
                            size=pred_masks_class.shape[-2:],
                            mode='nearest'
                        )
                    
                    # Point sampling
                    with torch.no_grad():
                        point_coords = get_uncertain_point_coords_with_randomness(
                            pred_masks_class,
                            None,
                            self.num_points,
                            self.oversample_ratio,
                            self.importance_sample_ratio
                        )
                    
                    sampled_pred = point_sample(pred_masks_class, point_coords)
                    sampled_gt = point_sample(gt_masks_class, point_coords)
                    
                    # Compute losses
                    num_masks = class_mask.sum()
                    loss_dice += self.loss_dice(
                        sampled_pred,
                        sampled_gt,
                        avg_factor=num_masks
                    )
                    
                    loss_mask += self.loss_mask(
                        sampled_pred.reshape(-1),
                        sampled_gt.reshape(-1),
                        avg_factor=num_masks * self.num_points
                    )
            
            # Average losses
            num_active_classes = (gt_labels_multi.sum(0) > 0).sum()
            if num_active_classes > 0:
                loss_dice = loss_dice / num_active_classes
                loss_mask = loss_mask / num_active_classes
        
        losses = {
            'loss_cls': loss_cls,
            'loss_dice': loss_dice,
            'loss_mask': loss_mask
        }
        
        return losses