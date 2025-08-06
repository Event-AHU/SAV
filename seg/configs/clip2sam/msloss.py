import torch
from torch import nn, Tensor
from torch.nn import functional as F

class MA_Loss(nn.Module):
    def __init__(self,reduction='mean', loss_weight=1.0):
        super().__init__()
        self.sl1 = nn.SmoothL1Loss()
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, cls_scores, pred_masks, gt_labels, gt_masks):
        '''
        input:  cls_scores: torch.Size([40, 1, 13])
                pred_masks: torch.Size([40, 1, 12544])
                gt_labels: torch.Size([40])
                gt_masks: torch.Size([40, 1, 12544])
        
        output: ma_loss
        '''
        
        batch_size = cls_scores.shape[0]
        out_vocab_cls_results = cls_scores  # [40, 1, 13]
        mask_results = pred_masks  # [40, 1, 12544]
        
        logits_per_image = F.softmax(out_vocab_cls_results[...,:-1], dim=-1)  # 40*1*12
        num_classes = logits_per_image.shape[-1]

        logits_per_instance = []  # bn * 1
        labels_per_instance = []  # bn * 12544
        masks_per_instance = []   # bn * 1 * 12544
        
        mask_results = mask_results.sigmoid()
        
        for b in range(batch_size):
            maski = mask_results[b]
            # Add safeguard for gt_labels
            label_index = min(gt_labels[b].item(), num_classes - 1)
            logiti = logits_per_image[b, :, label_index]
            labeli = gt_masks[b]
            logits_per_instance.append(logiti)
            labels_per_instance.append(labeli)
            masks_per_instance.append(maski)
        
        masks_per_instance = torch.stack(masks_per_instance, dim=0)
        labels_per_instance = torch.stack(labels_per_instance, dim=0)
        logits_per_instance = torch.stack(logits_per_instance, dim=0)

        ious = self.get_iou(masks_per_instance, labels_per_instance).detach()  # bs*1
        ious = self.mynorm(ious)
        
        # Ensure logits_per_instance and ious have the same shape
        logits_per_instance = logits_per_instance.squeeze()  # [40]
        ious = ious.squeeze()  # [40]
        
        ma_loss = self.sl1(logits_per_instance, ious)
        # Apply reduction and loss_weight
        if self.reduction == 'mean':
            ma_loss = ma_loss.mean()
        elif self.reduction == 'sum':
            ma_loss = ma_loss.sum()

        return self.loss_weight * ma_loss


    def get_iou(self, pred, target):
        b, c, h = pred.shape
        
        pred = pred.view(b, -1)
        target = target.view(b, -1)
        
        # compute the IoU of the foreground
        Iand1 = torch.sum(target * pred, dim=-1)
        Ior1 = torch.sum(target, dim=-1) + torch.sum(pred, dim=-1) - Iand1 + 0.0000001
        IoU1 = Iand1 / Ior1

        return IoU1.unsqueeze(1)  # [b, 1]

    def mynorm(self, embeding):
        assert len(embeding.shape) == 2, embeding.shape
        min_em, _ = torch.min(embeding, dim=-1, keepdim=True)
        max_em, _ = torch.max(embeding, dim=-1, keepdim=True)
        embeding = (embeding - min_em) / (max_em - min_em + 0.00000001)
        return embeding