import torch
import torch.nn as nn
import torch.nn.functional as F

""" Loss Functions -------------------------------------- """

# Per-class CrossEntropy weights, derived per-dataset from the actual per-class
# pixel frequency in each dataset's TRAINING masks. Base scheme is median-frequency
# balancing (power p=1):
#     w_c = (median(class_pixel_counts) / class_pixel_count_c) ** p   (normalized: benign = 1.0)
# Background (class 0) is ~90%+ (BUSI) / ~99.7% (CBIS) of all pixels, so it is
# heavily down-weighted; the rare lesion classes carry the CE gradient. These are
# backstopped by the foreground-only Dice term (see DICE_INCLUDE_BACKGROUND).
# Recompute if the training split changes. CBIS counts are over MASS-only masks
# (the mass filter is what training consumes). Tunable.
#
# BUSI: pure median-frequency (p=1). Pixels: bg 143,157,846 / benign 7,737,471 / malignant 7,062,221.
# CBIS: median-freq at p=1 gave bg=0.00172, which OVER-SEGMENTED in a 3-epoch smoke test
#       (foreground recall ~0.58 but precision pinned at ~0.01 with no upward trend). CBIS is
#       therefore power-TEMPERED to p~=0.72, lifting bg 0.00172 -> ~0.010 so background contributes
#       ~74% of the CE loss (vs ~33% at p=1). Pixels: bg 9,524,082,349 / benign 16,375,443 /
#       malignant 14,672,983.
#                     [background, benign, malignant]
BUSI_CLASS_WEIGHTS = [0.0540, 1.0, 1.096]
CBIS_CLASS_WEIGHTS = [0.010, 1.0, 1.10]

DICE_INCLUDE_BACKGROUND = False


class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1e-5, include_background=DICE_INCLUDE_BACKGROUND):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, inputs, targets):
        # 1. Apply Softmax to model outputs to get probabilities
        inputs = torch.softmax(inputs, dim=1)

        # 2. PROPER BITMASKING: Convert target (Batch, H, W) to one-hot
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # 3. Calculate Dice per class -> (Batch, Classes)
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        # 4. Average the loss across classes. Excluding background (class 0)
        #    stops its trivially-high Dice from softening the lesion gradient.
        if not self.include_background and self.num_classes > 1:
            dice_score = dice_score[:, 1:]

        return 1.0 - dice_score.mean()

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes=3,
        class_weights=None,
        dice_include_background=DICE_INCLUDE_BACKGROUND,
        smooth=1e-5,
    ):
        super(CombinedLoss, self).__init__()
        self.dice = MultiClassDiceLoss(
            num_classes=num_classes,
            smooth=smooth,
            include_background=dice_include_background,
        )

        # Weighted CrossEntropy counters background pixel dominance. nn.CrossEntropyLoss
        # registers `weight` as a buffer (via _WeightedLoss), so calling
        # `CombinedLoss.to(device)` moves this weight tensor along with the module.
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)
        else:
            weight = None
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets):
        # inputs: [Batch, Classes, H, W] (Raw Logits)
        # targets: [Batch, H, W] (Class Indices)
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return ce_loss + dice_loss

""" Metrics ------------------------------------------ """
def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)

def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)

def F2(y_true, y_pred, beta=2):
    p = precision(y_true,y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2.) *(p*r) / float(beta**2*p + r + 1e-15)

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)
