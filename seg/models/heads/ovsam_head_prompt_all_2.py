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

from seg.models.label_anything.models.prompt_encoder import PromptImageEncoder
from einops import rearrange
from ext.sam.transformer import TwoWayTransformer
from seg.models.label_anything.utils.utils import ResultDict
from torch import Tensor
import math


import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from seg.models.TransReid.config import cfg
from seg.models.TransReid.model import make_model
from seg.models.TransReid.utils.logger import setup_logger
from seg.models.TransReid.utils.metrics import euclidean_distance

import os
import torch
import numpy as np
from PIL import Image

def save_masks_to_folder(masks, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    masks_single = masks[0]
    
    for i, mask in enumerate(masks_single):
        binary_mask = (mask.sigmoid() > 0.5).float().cpu().numpy()
        
        mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8))
        
        mask_image.save(os.path.join(folder_path, f"mask_{i}.png"))
    
    print(f"所有掩码已成功保存到文件夹：{folder_path}")

def stack_batch(batch_inputs, pad_size_divisor, pad_value):
    if len(batch_inputs) == 1:
        tensor = batch_inputs[0]
        
        _, h, w = tensor.shape
        
        pad_h = int(math.ceil(h / pad_size_divisor)) * pad_size_divisor - h
        pad_w = int(math.ceil(w / pad_size_divisor)) * pad_size_divisor - w
        
        if pad_h > 0 or pad_w > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), value=pad_value)
            
        return tensor.unsqueeze(0)
    else:
        max_shape = [0, 0]
        for tensor in batch_inputs:
            _, h, w = tensor.shape
            max_shape[0] = max(max_shape[0], h)
            max_shape[1] = max(max_shape[1], w)
        
        max_shape[0] = int(math.ceil(max_shape[0] / pad_size_divisor)) * pad_size_divisor
        max_shape[1] = int(math.ceil(max_shape[1] / pad_size_divisor)) * pad_size_divisor
        
        padded_tensors = []
        for tensor in batch_inputs:
            _, h, w = tensor.shape
            pad_h = max_shape[0] - h
            pad_w = max_shape[1] - w
            
            if pad_h > 0 or pad_w > 0:
                tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), value=pad_value)
            
            padded_tensors.append(tensor)
        
        return torch.stack(padded_tensors, dim=0)

def save_reference_masks(reference_masks, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dimensions
    batch_size, num_examples, num_classes, height, width = reference_masks.shape
    
    # Convert to numpy array for easier manipulation
    reference_masks_np = reference_masks.cpu().numpy()
    
    # Iterate over each mask and save as an image
    for batch_idx in range(batch_size):
        for example_idx in range(num_examples):
            for class_idx in range(num_classes):
                mask = reference_masks_np[batch_idx, example_idx, class_idx]
                
                # Normalize mask to [0, 255] and convert to uint8
                mask = (mask * 255).astype(np.uint8)
                
                # Create image from mask
                mask_image = Image.fromarray(mask)
                
                # Save image
                save_path = os.path.join(save_dir, f"batch_{batch_idx}_example_{example_idx}_class_{class_idx}.png")
                mask_image.save(save_path)
                print(f"Saved mask to {save_path}")



# Helper function to load reference data from mask path
def load_reference_data(image_path, mask_path):
    """
    Load mask and label data for a reference image
    
    Args:
        image_path: Path to the reference image
        mask_path: Path to the corresponding mask annotation
        
    Returns:
        Dictionary with 'masks' (list of numpy arrays) and 'labels' (list of class indices)
    """
    try:
        # Check if mask file exists
        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}")
            return None
            
        # Load mask data - implementation depends on your annotation format
        # For example, if using PNG masks where pixel values represent class IDs:
        mask_data = np.array(Image.open(mask_path))
        
        # Extract unique class labels (excluding background class 0)
        unique_labels = np.unique(mask_data)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background
        
        if len(unique_labels) == 0:
            print(f"No valid class labels found in mask: {mask_path}")
            return None
            
        # Extract binary mask for each class
        masks = []
        labels = []
        
        for label in unique_labels:
            binary_mask = (mask_data == label).astype(np.float32)
            masks.append(binary_mask)
            labels.append(int(label))
            
        return {
            'masks': masks,
            'labels': labels
        }
        
    except Exception as e:
        print(f"Error loading reference data for {image_path}: {str(e)}")
        return None
def load_model(config_file, weights_path):
    """Load the pre-trained model"""
    # Load configuration
    cfg.merge_from_file(config_file)
    cfg.freeze()
    
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = make_model(cfg, num_class=0, camera_num=0, view_num=0)  # Dummy values for testing
    model.load_param(weights_path)
    model.to(device)
    model.eval()
    
    return model, device, cfg

def load_gallery_features(gallery_dir, model, device, cfg):
    """Load or compute features for gallery images"""
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    
    # Check if pre-computed features exist
    if os.path.exists('gallery_features.pt'):
        print("Loading pre-computed gallery features...")
        gallery_data = torch.load('gallery_features.pt')
        return gallery_data['features'], gallery_data['paths']
    
    print("Computing gallery features...")
    gallery_features = []
    gallery_paths = []
    
    # Recursively find all images in gallery directory
    for root, dirs, files in os.walk(gallery_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                gallery_paths.append(img_path)
                
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img = transform(img).unsqueeze(0).to(device)
                
                # Extract features
                with torch.no_grad():
                    feat = model(img)
                
                if cfg.TEST.FEAT_NORM == 'yes':
                    feat = torch.nn.functional.normalize(feat, dim=1, p=2)
                
                gallery_features.append(feat.cpu())
    
    # Convert to tensors
    gallery_features = torch.cat(gallery_features, dim=0)
    
    # Save computed features
    torch.save({
        'features': gallery_features,
        'paths': gallery_paths
    }, 'gallery_features.pt')
    
    return gallery_features, gallery_paths

import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image

def query_image(query_img_path, model, gallery_features, gallery_paths, device, cfg, top_k=5):
    """Query with a target image and retrieve similar images"""
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    
    # Load and preprocess query image
    query_img = Image.open(query_img_path).convert('RGB')
    query_img_tensor = transform(query_img).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        query_feat = model(query_img_tensor)
    
    if cfg.TEST.FEAT_NORM == 'yes':
        query_feat = torch.nn.functional.normalize(query_feat, dim=1, p=2)
    
    # Compute distances
    dist = euclidean_distance(query_feat.cpu(), gallery_features)
    
    # Ensure dist is a numpy array
    if isinstance(dist, torch.Tensor):
        dist = dist.numpy()
    
    dist = dist.flatten()
    
    # Get top-k matches
    indices = np.argsort(dist)[:top_k]
    match_paths = [gallery_paths[idx] for idx in indices]
    match_distances = [dist[idx] for idx in indices]
    
    return match_paths, match_distances

def display_results(query_img_path, match_paths, match_distances, top_k=5):
    """Display the query image and top matches"""
    plt.figure(figsize=(15, 8))
    
    # Display query image
    query_img = Image.open(query_img_path).convert('RGB')
    plt.subplot(1, top_k+1, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')
    
    # Display matches
    for i, (path, dist) in enumerate(zip(match_paths, match_distances)):
        img = Image.open(path).convert('RGB')
        plt.subplot(1, top_k+1, i+2)
        plt.imshow(img)
        plt.title(f"Match {i+1}\nDist: {dist:.2f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('query_results.jpg')
    plt.show()

class Prototype_Prompt_Encoder_MultiClass(nn.Module):
    def __init__(self, feat_dim=256,
                      hidden_dim_dense=128,
                      hidden_dim_sparse=128,
                      size=64,
                      num_tokens=8,
                      num_classes=7):
                      
        super(Prototype_Prompt_Encoder_MultiClass, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.size = size
        
        self.dense_fc_1 = nn.Conv2d(feat_dim, hidden_dim_dense, 1)
        self.dense_fc_2 = nn.Conv2d(hidden_dim_dense, feat_dim, 1)
        
        self.relu = nn.ReLU() 
        
        self.sparse_fc_1 = nn.Conv1d(size*size, hidden_dim_sparse, 1)
        self.sparse_fc_2 = nn.Conv1d(hidden_dim_sparse, num_tokens, 1)
        
        pn_cls_embeddings = [nn.Embedding(num_tokens, feat_dim) for _ in range(2)] # one for positive and one for negative 
        
        self.pn_cls_embeddings = nn.ModuleList(pn_cls_embeddings)
        
        self.class_aggregation = nn.Conv2d(feat_dim * num_classes, feat_dim, 1)
            
    def forward(self, feat, prototypes, mask_labels=None):
        batch_size = feat.size(0)
        
        if mask_labels is None:
            mask_labels = torch.ones(batch_size, self.num_classes, device=feat.device)
        
        if prototypes.dim() == 2:  # [num_classes, C]
            prototypes = prototypes.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_classes, C]
        
        cls_prompts = prototypes.unsqueeze(-1)  # [B, num_classes, C, 1]
        
        feat_expanded = torch.stack([feat for _ in range(self.num_classes)], dim=1)  # [B, num_classes, H*W, C]
        
        sim = torch.matmul(feat_expanded, cls_prompts)  # [B, num_classes, H*W, 1]
        
        activated_feat = feat_expanded + feat_expanded * sim  # [B, num_classes, H*W, C]
        
        feat_sparse = activated_feat.clone()
        
        all_dense_embeddings = []
        
        for cls_idx in range(self.num_classes):
            curr_feat = activated_feat[:, cls_idx]  # [B, H*W, C]
            curr_feat = rearrange(curr_feat, 'b (h w) c -> b c h w', h=self.size, w=self.size)
            curr_dense = self.dense_fc_2(self.relu(self.dense_fc_1(curr_feat)))  # [B, C, H, W]
            all_dense_embeddings.append(curr_dense)
        
        all_dense_concat = torch.cat(all_dense_embeddings, dim=1)  # [B, num_classes*C, H, W]
        dense_embeddings = self.class_aggregation(all_dense_concat)  # [B, C, H, W]
        
        feat_sparse = rearrange(feat_sparse, 'b num_cls hw c -> (b num_cls) hw c')
        sparse_embeddings = self.sparse_fc_2(self.relu(self.sparse_fc_1(feat_sparse)))
        sparse_embeddings = rearrange(sparse_embeddings, '(b num_cls) n c -> b num_cls n c', num_cls=self.num_classes)
        
        pos_embed = self.pn_cls_embeddings[1].weight.unsqueeze(0).unsqueeze(0) * mask_labels.unsqueeze(-1).unsqueeze(-1)
        neg_embed = self.pn_cls_embeddings[0].weight.unsqueeze(0).unsqueeze(0) * (1-mask_labels).unsqueeze(-1).unsqueeze(-1)
        
        sparse_embeddings = sparse_embeddings + pos_embed.detach() + neg_embed.detach()
        sparse_embeddings = rearrange(sparse_embeddings, 'b num_cls n c -> b (num_cls n) c')
        
        return dense_embeddings, sparse_embeddings


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class ShortCut_CrossAttention(nn.Module):

    def __init__(self, d_model, nhead):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu

        self._reset_parameters()

        self.MLP = nn.Linear(d_model, d_model)
        # nn.init.constant(self.MLP.weight, 0.0)
        # nn.init.constant(self.MLP.bias, 0.0)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.MLP(tgt2)
        tgt = self.norm(tgt)
        
        return tgt


class ContentDependentTransfer(nn.Module):

    def __init__(self, d_model, nhead):
        super().__init__()
        self.pe_layer = PositionEmbeddingSine(d_model//2, normalize=True)
        self.cross_atten = ShortCut_CrossAttention(d_model = d_model, nhead = nhead)

    def visual_prediction_forward_convnext(self, x):
        batch, channel, h, w = x.shape
        x = x.reshape(batch*h*w, channel).unsqueeze(-1).unsqueeze(-1) # fake 2D input
        x = self.truck_head(x)
        x = self.head(x)
        return x.reshape(batch, h, w, x.shape[-1]).permute(0,3,1,2) # B x num_queries x 640
    
    def forward(self, img_feat, text_classifier):
        # Remove the unsqueeze and repeat as text_classifier already has a batch dimension [batch, num_queries, dim]
        text_classifier = text_classifier.to('cuda')

        pos = self.pe_layer(img_feat, None).flatten(2).permute(2, 0, 1).to('cuda')  # hw * b * c
        img_feat = img_feat.flatten(2).permute(2, 0, 1)  # hw * b * c

        # Permute text_classifier from [batch, num_queries, dim] to [num_queries, batch, dim]
        text_classifier_permuted = text_classifier.permute(1, 0, 2)

        bias = self.cross_atten(text_classifier_permuted, img_feat, memory_mask=None, memory_key_padding_mask=None, pos=pos, query_pos=None)

        return bias.permute(1, 0, 2)


class Attention(nn.Module):
    """Multi-head Self Attention Module with context projection."""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, key_padding_mask=None):
        batch_size = query.size(0)
        
        q = self.q_proj(query)  # [B, Q_Len, D]
        k = self.k_proj(key)    # [B, K_Len, D]
        v = self.v_proj(value) # [B, V_Len, D]
        
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Q_Len, D/H]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K_Len, D/H]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, V_Len, D/H]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, Q_Len, K_Len]
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K_Len]
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ v  # [B, H, Q_Len, D/H]
        
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, Q_Len, H, D/H]
        attn_output = attn_output.view(batch_size, -1, self.embed_dim)  # [B, Q_Len, D]
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

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
            prompt_encoder_cfg = None
    ):
        assert init_cfg is not None and \
            init_cfg['type'] in ['sam_pretrain', 'Pretrained'], f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__(init_cfg=None)
        self.init_cfg = init_cfg
        self.logger = MMLogger.get_current_instance()
        self.cdt = ContentDependentTransfer(d_model=256, nhead=8)#【修改】
        # self.prototype_projection = nn.Linear(768, 256)
        # self.combine_projection = nn.Linear(512, 256)
        self.prototype_proj_to_roi_dim = nn.Linear(256, 768)
        self.attention = Attention(
            768,
            num_heads=8,
        )

        if prompt_encoder_cfg is not None:
            # Get the prompt encoder configuration
            prompt_encoder = MODELS.build(prompt_encoder_cfg)
            
            # Fix the class_encoder if it's a dictionary
            if hasattr(prompt_encoder, 'class_encoder') and isinstance(prompt_encoder.class_encoder, dict):
                if 'type' in prompt_encoder.class_encoder:
                    encoder_class = prompt_encoder.class_encoder['type']
                    encoder_params = {k: v for k, v in prompt_encoder.class_encoder.items() if k != 'type'}
                    try:
                        prompt_encoder.class_encoder = encoder_class(**encoder_params)
                        print(f"Successfully initialized class_encoder: {type(prompt_encoder.class_encoder)}")
                    except Exception as e:
                        print(f"Error initializing class_encoder: {str(e)}")
                        prompt_encoder.class_encoder = None
            
            # Fix the transformer if it's a dictionary
            if hasattr(prompt_encoder, 'transformer') and isinstance(prompt_encoder.transformer, dict):
                if 'type' in prompt_encoder.transformer:
                    transformer_class = prompt_encoder.transformer['type']
                    transformer_params = {k: v for k, v in prompt_encoder.transformer.items() if k != 'type'}
                    try:
                        prompt_encoder.transformer = transformer_class(**transformer_params)
                        print(f"Successfully initialized transformer: {type(prompt_encoder.transformer)}")
                    except Exception as e:
                        print(f"Error initializing transformer: {str(e)}")
                        # For transformer, we can't use None as it's required
                        # Create a minimal transformer or raise an error
                        raise RuntimeError(f"Failed to initialize transformer: {str(e)}")
            
            self.prompt_encoder = prompt_encoder
        else:
            # Fallback to direct initialization
            self.prompt_encoder = PromptImageEncoder(
                embed_dim=meta_dict[model_name]['prompt_embed_dim'],
                image_embedding_size=(64, 64),
                activation=nn.GELU,
                mask_in_chans=16,
            )
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

    def forward_logit(self, cls_embd, cls_embed=None):
        """Calculate classification logits with optional explicit class embeddings."""
        if cls_embed is not None:
            cls_embd_norm = F.normalize(cls_embd, dim=-1)
            cls_embed_norm = F.normalize(cls_embed, dim=-1)
            cls_pred = torch.bmm(cls_embd_norm, cls_embed_norm.transpose(1, 2))
            cls_pred = cls_pred.squeeze(1)
            cls_pred = cls_pred.unsqueeze(1)
            cls_pred = self.logit_scale.exp() * cls_pred
        else:
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
            backbone=None,
            data_samples=None,
            GCN_Features=None, 
            neck=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        # Get batch size and dimensions
        batch_size = image_embeddings.shape[0]
        embedding_dim = image_embeddings.shape[1]
        image_embedding_height, image_embedding_width = image_embeddings.shape[2], image_embeddings.shape[3]
        
        # Configuration parameters
        num_examples = getattr(self, 'num_reference_examples', 2)
        num_classes = self.num_classes  # Use model's class count
        mask_height = image_embedding_height * 4  # Mask resolution is 4x the feature map
        mask_width = image_embedding_width * 4
        
        # Initialize reference data containers
        reference_image_embeddings = torch.zeros(
            batch_size, num_examples, embedding_dim, image_embedding_height, image_embedding_width
        ).to(image_embeddings.device)
        reference_masks = torch.zeros(
            batch_size, num_examples, num_classes, mask_height, mask_width
        ).to(image_embeddings.device)
        mask_flags = torch.zeros(batch_size, num_examples, num_classes, device=image_embeddings.device)
        flag_examples = torch.zeros(batch_size, num_examples, num_classes, device=image_embeddings.device)
        
        # ===== ReID-based Reference Image Retrieval =====
        # Check if model has ReID components cached (to avoid reloading)
        if not hasattr(self, 'reid_model') or self.reid_model is None:
            # Load ReID model and gallery features (only once)
            config_file = "/data1/Code/wangziwen/Project/123/ovsam-main/seg/models/TransReid/configs/VeRi/vit_transreid_stride_new.yml"
            weights_path = "/data1/Code/wuwentao/pre_model/0921_vehiclemaev2_100W_mask75/checkpoint-100.pth"
            # gallery_dir = "/data1/Datasets/Seg/CarPart/images/valid/"
            gallery_dir = "/data1/Datasets/Seg/3DRealCar_Segment_Dataset/images/train/"

            
            print("Loading ReID model and gallery features...")
            self.reid_model, self.reid_device, self.reid_cfg = load_model(
                config_file, weights_path
            )
            self.gallery_features, self.gallery_paths = load_gallery_features(
                gallery_dir, self.reid_model, self.reid_device, self.reid_cfg
            )
            print(f"Loaded gallery with {len(self.gallery_paths)} images")
        
        # Create instance mappings
        instance_to_img_mapping = {}
        instance_to_label_mapping = {}
        instance_count = 0
        
        # Process data samples if available
        data_samples_list = []
        if data_samples is not None:
            if isinstance(data_samples, (list, tuple)):
                data_samples_list = data_samples
            else:
                data_samples_list = [data_samples]
        
        # Track processed image paths to avoid duplicate ReID queries
        processed_imgs = set()
        
        # First pass: get image paths and create mappings
        for img_idx, sample in enumerate(data_samples_list):
            # Get image path from data sample
            if hasattr(sample, 'img_path'):
                img_path = sample.img_path
            elif hasattr(sample, 'metainfo') and 'img_path' in sample.metainfo:
                img_path = sample.metainfo['img_path']
            else:
                # Try to extract from img_id if available
                if hasattr(sample, 'img_id'):
                    img_path = f"image_{sample.img_id}.png"  # Construct path from ID
                else:
                    print(f"Warning: Can't find image path for sample {img_idx}")
                    continue
            
            # Map instances to images and labels
            if hasattr(sample, 'gt_instances') and hasattr(sample.gt_instances, 'labels'):
                for label_idx, label in enumerate(sample.gt_instances.labels):
                    instance_to_img_mapping[instance_count] = img_idx
                    instance_to_label_mapping[instance_count] = label.item()
                    instance_count += 1
            

            mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
            std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
            pad_size_divisor = 1024
            pad_value = 0


            if img_path not in processed_imgs:
                processed_imgs.add(img_path)
                
                # Use ReID model to query similar images
                try:
                    self.reid_model.eval()

                    # 修改这里：增加获取的相似图片数量
                    match_paths, match_distances = query_image(
                        query_img_path=img_path,
                        model=self.reid_model,
                        gallery_features=self.gallery_features, 
                        gallery_paths=self.gallery_paths,
                        device=self.reid_device,
                        cfg=self.reid_cfg,
                        top_k=num_examples+1  # 增加检索数量匹配num_examples
                    )
                    
                    print(f"Found {len(match_paths)} matches for {img_path}")
                    
                    # match_paths = match_paths[1:]
                    # 修改这里：处理每个参考图像的特征提取
                    for example_idx, match_path in enumerate(match_paths):
                        if example_idx >= num_examples:
                            break
                            
                        try:
                            image = Image.open(match_path)
                            image_np = np.array(image)
                            image_tensor = torch.tensor(image_np, dtype=torch.float32)
                            image_tensor_chw = image_tensor.permute(2, 0, 1)
                            _batch_input = image_tensor_chw.float()
                            normalized_input = (_batch_input - mean) / std
                            batch_inputs = [normalized_input]
                            padded_stacked_input = stack_batch(batch_inputs, pad_size_divisor, pad_value)
                            padded_stacked_input = padded_stacked_input.to(image_embeddings.device)
                            backbone_feats = backbone(padded_stacked_input)
                            ref_embeddings = neck(backbone_feats)
                            for instance_idx, img_map_idx in instance_to_img_mapping.items():
                                if img_map_idx == img_idx:
                                    reference_image_embeddings[instance_idx, example_idx] = ref_embeddings.squeeze(0)
                        
                        except Exception as e:
                            print(f"Error processing reference image {match_path}: {str(e)}")
                            continue
                    
                    for example_idx, match_path in enumerate(match_paths):
                        if example_idx >= num_examples:
                            break
                            
                        # Convert image path to annotation/mask path
                        mask_path = match_path.replace('images', 'annotations')
                        mask_path = os.path.splitext(mask_path)[0] + '.png'
                        
                        print(f"Looking for mask at: {mask_path}")
                        
                        # Load the matched reference data with masks and class labels
                        ref_data = load_reference_data(match_path, mask_path)
                        
                        if ref_data is not None:
                            ref_masks = ref_data['masks']  # List of masks
                            ref_labels = ref_data['labels']  # List of class labels
                            ref_labels = [label - 1 for label in ref_labels]
                            
                            # Register which classes are valid in this reference
                            valid_classes = set(ref_labels)
                            
                            # For each instance mapped to this image, find and assign only its specific class mask
                            for instance_idx, img_map_idx in instance_to_img_mapping.items():
                                if img_map_idx == img_idx:
                                    # Get the label for this instance
                                    label_idx = instance_to_label_mapping.get(instance_idx)
                                    
                                    if label_idx is not None and label_idx in valid_classes:
                                        # Find the mask for this label in the reference data
                                        for mask_idx, (mask, ref_label) in enumerate(zip(ref_masks, ref_labels)):
                                            if ref_label == label_idx:
                                                # Resize mask to required dimensions
                                                resized_mask = torch.nn.functional.interpolate(
                                                    torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0),
                                                    size=(mask_height, mask_width),
                                                    mode='bilinear',
                                                    align_corners=False
                                                ).squeeze().to(image_embeddings.device)
                                                
                                                # Ensure binary mask
                                                resized_mask = (resized_mask > 0.5).float()
                                                
                                                # Assign only the mask for this instance's specific class
                                                reference_masks[instance_idx, example_idx, label_idx] = resized_mask
                                                break  # Only assign the first matching mask
                                    
                                    # Set flags for all valid classes in the image
                                    for valid_label in valid_classes:
                                        if valid_label < num_classes:
                                            mask_flags[instance_idx, example_idx, valid_label] = 1.0
                                            flag_examples[instance_idx, example_idx, valid_label] = 1.0
                
                except Exception as e:
                    print(f"Error processing ReID for {img_path}: {str(e)}")
                    print("Attempting to reset ReID model to evaluation mode and retry...")
                    
                    try:
                        # Force reset model state and try again
                        self.reid_model.eval()
                        
                        # Clear any accumulated gradients
                        if hasattr(self.reid_model, 'zero_grad'):
                            self.reid_model.zero_grad()
                            
                        # Try the query again
                        match_paths, match_distances = query_image(
                            query_img_path=img_path,
                            model=self.reid_model,
                            gallery_features=self.gallery_features, 
                            gallery_paths=self.gallery_paths,
                            device=self.reid_device,
                            cfg=self.reid_cfg,
                            top_k=num_examples+1
                        )
                        
                        print(f"Retry successful: Found {len(match_paths)} matches for {img_path}")
                        
                        
                    except Exception as retry_error:
                        print(f"Retry failed: {str(retry_error)}")
                        print("Continuing without ReID matches for this image")
                        continue  # Skip to next image
        
        # save_reference_masks(reference_masks, save_dir="/home/wangziwen/Project/123/ovsam-main/masks_multi")
        
        # Prepare PromptImageEncoder input
        masks = (reference_masks, mask_flags)
        
        # Call PromptImageEncoder to get class prototypes
        if hasattr(self, 'prompt_encoder'):
            try:
                prompt_encoder_output = self.prompt_encoder(
                    image_embeddings=reference_image_embeddings,
                    points=None,
                    boxes=None,
                    masks=masks,
                    flag_examples=flag_examples
                )
                
                # Get class prototypes
                class_prototypes = prompt_encoder_output.get('class_embeddings', None)
                
                print(f"Generated class prototypes from {num_examples} reference images, shape: {class_prototypes.shape if class_prototypes is not None else 'None'}")
            
            except Exception as e:
                print(f"Error generating class prototypes: {e}")
                print(f"Reference image embeddings shape: {reference_image_embeddings.shape}")
                print(f"Reference masks shape: {reference_masks.shape}")
                print(f"Mask flags shape: {mask_flags.shape}")
                # Continue execution with default prototypes
        else:
            print("No prompt_encoder found, unable to generate class prototypes")
            # 处理无prompt_encoder情况
            # ...
        class_prototypes = self.cdt(image_embeddings, class_prototypes)
        # Original mask prediction code (unchanged)
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

        # Generate class labels if required
        if self.with_label_token:
            device = label_queries.device
            cls_embed_list = []
            for i in range(self.num_classes):
                cls_embed_list.append(self.label_mlp(label_queries[:, i, :]))
            if self.gen_box:
                batch_masks = masks[0]  # Shape: [13, 256, 256]
                
                # Process data samples to get ground truth labels
                gt_labels = []
                if data_samples is not None:
                    if isinstance(data_samples, (list, tuple)):
                        for sample in data_samples:
                            if hasattr(sample, 'gt_instances') and hasattr(sample.gt_instances, 'labels'):
                                gt_labels.extend(sample.gt_instances.labels.cpu().tolist())
                    else:
                        if hasattr(data_samples, 'gt_instances') and hasattr(data_samples.gt_instances, 'labels'):
                            gt_labels = data_samples.gt_instances.labels.cpu().tolist()
                
                # If no ground truth labels available, use sigmoid threshold on all masks
                if not gt_labels:
                    print("No ground truth labels found, using all masks with threshold")
                    combined_mask = masks.sigmoid().max(dim=1)[0] > 0.5
                else:
                    # Select masks corresponding to ground truth labels
                    print(f"Using ground truth labels for mask selection: {gt_labels}")
                    selected_masks = []
                    for label in gt_labels:
                        if 0 <= label < self.num_classes:
                            mask = batch_masks[label].sigmoid() > 0.5
                            selected_masks.append(mask)

                combined_mask = torch.stack(selected_masks, dim=0)
                # Generate bounding boxes from masks
                bboxes = mask2bbox(combined_mask) * 4
                roi_list = bbox2roi([bboxes])       
            
            roi_feats = self.roi(fpn_feats, roi_list)
            roi_feats = self.roi_conv(roi_feats)
            roi_feats = roi_feats.mean(dim=-1).mean(dim=-1)  # Shape: [total_instances, 768]
            
            # Construct ROI features for attention
            num_instances = image_embeddings.size(0)
            feature_dim = roi_feats.size(-1)  # 768
            
            new_roi_feats = torch.zeros(num_instances, num_classes, feature_dim, device=roi_feats.device)
            
            # Map features using instance mappings
            for instance_idx in range(num_instances):
                label_idx = instance_to_label_mapping.get(instance_idx)
                if label_idx is not None:
                    new_roi_feats[instance_idx, label_idx] = roi_feats[instance_idx]
            
            combined_roi_feats = new_roi_feats + cls_embed
            
            # Ensure prototypes have correct shape
            if len(class_prototypes.shape) == 2:
                prototype_features = class_prototypes.unsqueeze(0).expand(num_instances, -1, -1)
            else:
                prototype_features = class_prototypes
            prototype_features = self.prototype_proj_to_roi_dim(prototype_features)
            
            # Apply attention mechanism
            attn_output, attn_weights = self.attention(
                combined_roi_feats.unsqueeze(1),
                prototype_features,
                prototype_features
            )
            
            attention_weights = attn_weights.mean(dim=1)
            attended_prototypes = attention_weights @ prototype_features
            enhanced_roi_feats = combined_roi_feats * attended_prototypes
            
            # Final class prediction
            if hasattr(self, 'cls_embed') and isinstance(self.cls_embed, torch.Tensor):
                cls_embed_tensor = self.cls_embed.view(1, num_classes+1, feature_dim).expand(num_instances, -1, -1)
                cls_pred = self.forward_logit(enhanced_roi_feats, cls_embed_tensor)
            else:
                cls_pred = self.forward_logit(enhanced_roi_feats)
        else:
            cls_pred = None
        
        return masks, None, cls_pred.squeeze(1)

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
            GCN_Feature=None,
            neck=None,  # 传递 self.neck
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
            GCN_Features = GCN_Feature, 
            data_samples = data_samples,
            neck = neck,
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
        GCN_Feature=None,
        neck=None,  # 传递 self.neck
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
            neck = neck,
            data_samples=data_samples,
            GCN_Features = GCN_Feature, 
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
        # print(cls_preds.shape)
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
        # reg_term = 0.0
        # for param in self.parameters():
        #     reg_term = reg_term + param.mean() * 0.0

        reg_term = 0.0
        for param in self.parameters():
            mean_value = param.mean()
            if not torch.isnan(mean_value) and not torch.isinf(mean_value):
                reg_term += mean_value * 0.0
        # print(cls_scores.shape)
        # print(gt_labels_with_bg.shape)
        loss_cls = self.loss_cls(
            cls_scores,
            gt_labels_with_bg,
            valid_mask,
            avg_factor=valid_mask.sum().clamp(min=1.0)
        ) + reg_term

        # Mask loss computation
        loss_dice = 0.0
        loss_mask = 0.0
        mask_count = 0
        
        if 'masks' in gt_instances:
            # Process each batch sample separately
            for batch_idx, data_sample in enumerate(data_samples):
                # Get ground truth masks for this sample
                sample_gt_masks = data_sample.gt_instances.masks.to_tensor(
                    dtype=torch.float, device=device
                )
                sample_gt_labels = data_sample.gt_instances.labels
                
                # For each instance in this sample
                for inst_idx, label in enumerate(sample_gt_labels):
                    if label == -1:
                        continue
                    
                    # Get prediction mask for this class
                    pred_mask = masks[batch_idx, label].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    gt_mask = sample_gt_masks[inst_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    
                    # Resize ground truth if needed
                    if gt_mask.shape[-2:] != pred_mask.shape[-2:]:
                        gt_mask = F.interpolate(
                            gt_mask,
                            size=pred_mask.shape[-2:],
                            mode='nearest'
                        )
                    
                    # Point sampling
                    with torch.no_grad():
                        point_coords = get_uncertain_point_coords_with_randomness(
                            pred_mask,
                            None,
                            self.num_points,
                            self.oversample_ratio,
                            self.importance_sample_ratio
                        )
                    
                    sampled_pred = point_sample(pred_mask, point_coords)
                    sampled_gt = point_sample(gt_mask, point_coords)
                    
                    # Compute losses
                    loss_dice += self.loss_dice(
                        sampled_pred,
                        sampled_gt,
                        avg_factor=1.0
                    )
                    
                    loss_mask += self.loss_mask(
                        sampled_pred.reshape(-1),
                        sampled_gt.reshape(-1),
                        avg_factor=self.num_points
                    )
                    
                    mask_count += 1
            
            # Average losses over number of masks
            if mask_count > 0:
                loss_dice = loss_dice / mask_count
                loss_mask = loss_mask / mask_count
        
        losses = {
            'loss_cls': loss_cls,
            'loss_dice': loss_dice,
            'loss_mask': loss_mask
        }
        
        return losses