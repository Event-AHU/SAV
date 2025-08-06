import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, reduction='mean', loss_weight=1.0):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, text_features, roi_features, avg_factor=None):
        text_features = F.normalize(text_features, dim=1)
        roi_features = F.normalize(roi_features, dim=1)
        logits = torch.matmul(text_features, roi_features.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels, reduction='none')
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss