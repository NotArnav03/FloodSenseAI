"""
Loss functions for flood segmentation training.

Combines Binary Cross-Entropy and Dice Loss for handling
class imbalance (flood pixels are often minority class).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation — handles class imbalance well."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal loss — down-weights easy examples, focuses on hard pixels."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """BCE + Dice + Focal combined loss for robust segmentation training."""

    def __init__(self, bce_weight=0.3, dice_weight=0.5, focal_weight=0.2):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        loss_focal = self.focal(logits, targets)

        total = (
            self.bce_weight * loss_bce +
            self.dice_weight * loss_dice +
            self.focal_weight * loss_focal
        )
        return total
