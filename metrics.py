import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_dynamic_class_weights(class_pixel_counts, power, reference_class=1):
    """Compute fold-local class weights using the BUSI v2 power law.

    The returned weights are normalized to reference_class (benign by default):

        w_c = (n_reference / n_c) ** power

    Plain Python values are returned so this helper can resolve a run
    configuration before a device is selected.
    """

    counts = [float(count) for count in class_pixel_counts]
    if len(counts) < 2:
        raise ValueError("class_pixel_counts must contain at least two classes")
    if not 0 <= int(reference_class) < len(counts):
        raise ValueError("reference_class is outside class_pixel_counts")
    if float(power) < 0:
        raise ValueError("power must be non-negative")
    if any(not torch.isfinite(torch.tensor(count)) or count <= 0 for count in counts):
        raise ValueError("all class pixel counts must be finite and positive")

    reference_count = counts[int(reference_class)]
    return [(reference_count / count) ** float(power) for count in counts]

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


class BinaryForegroundOverlapLoss(nn.Module):
    """Binary foreground Dice or Tversky loss for lesion-positive samples.

    Background-only BUSI samples are supervised by cross entropy, but do not
    enter this overlap loss. Returning a differentiable zero for an all-normal
    batch keeps composite training finite and avoids inventing a free Dice score.

    fp_weight and fn_weight name their roles explicitly. For example,
    fp_weight=0.7, fn_weight=0.3 penalizes false-positive foreground more.
    """

    def __init__(
        self,
        mode="dice",
        smooth=1e-5,
        fp_weight=0.6,
        fn_weight=0.4,
    ):
        super().__init__()
        if mode not in {"dice", "tversky"}:
            raise ValueError("mode must be 'dice' or 'tversky'")
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        if fp_weight < 0 or fn_weight < 0 or fp_weight + fn_weight <= 0:
            raise ValueError("Tversky FP/FN weights must be non-negative with a positive sum")
        self.mode = mode
        self.smooth = float(smooth)
        self.fp_weight = float(fp_weight)
        self.fn_weight = float(fn_weight)

    def forward(self, inputs, targets):
        probabilities = torch.softmax(inputs, dim=1)
        predicted_foreground = probabilities[:, 1:].sum(dim=1)
        target_foreground = targets.ne(0).to(dtype=predicted_foreground.dtype)

        positive_samples = target_foreground.flatten(1).sum(dim=1).gt(0)
        if not torch.any(positive_samples):
            return predicted_foreground.sum() * 0.0

        predicted_foreground = predicted_foreground[positive_samples].flatten(1)
        target_foreground = target_foreground[positive_samples].flatten(1)

        true_positive = (predicted_foreground * target_foreground).sum(dim=1)
        false_positive = (predicted_foreground * (1.0 - target_foreground)).sum(dim=1)
        false_negative = ((1.0 - predicted_foreground) * target_foreground).sum(dim=1)

        if self.mode == "dice":
            score = (2.0 * true_positive + self.smooth) / (
                2.0 * true_positive + false_positive + false_negative + self.smooth
            )
        else:
            score = (true_positive + self.smooth) / (
                true_positive
                + self.fp_weight * false_positive
                + self.fn_weight * false_negative
                + self.smooth
            )
        return 1.0 - score.mean()


class PresenceAwareClassDiceLoss(nn.Module):
    """Batch-aggregated foreground Dice over classes present in ground truth."""

    def __init__(self, num_classes=3, smooth=1e-5):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include background and foreground")
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        self.num_classes = int(num_classes)
        self.smooth = float(smooth)

    def forward(self, inputs, targets):
        probabilities = torch.softmax(inputs, dim=1)
        target_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).to(probabilities.dtype)

        probabilities = probabilities[:, 1:]
        target_one_hot = target_one_hot[:, 1:]
        reduce_dims = (0, 2, 3)
        target_pixels = target_one_hot.sum(dim=reduce_dims)
        present = target_pixels.gt(0)
        if not torch.any(present):
            return probabilities.sum() * 0.0

        intersection = (probabilities * target_one_hot).sum(dim=reduce_dims)
        denominator = probabilities.sum(dim=reduce_dims) + target_pixels
        score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - score[present].mean()


class BUSIStageLoss(nn.Module):
    """Control or composite loss for one Modified Double U-Net output.

    mode="control" retains weighted CE plus the existing per-sample foreground
    Dice. mode="composite" uses CE plus binary overlap and presence-aware Dice.
    """

    def __init__(
        self,
        num_classes=3,
        class_weights=None,
        mode="control",
        localization="dice",
        localization_weight=0.7,
        class_dice_weight=0.3,
        fp_weight=0.6,
        fn_weight=0.4,
        smooth=1e-5,
    ):
        super().__init__()
        if mode not in {"control", "composite"}:
            raise ValueError("mode must be 'control' or 'composite'")
        if localization_weight < 0 or class_dice_weight < 0:
            raise ValueError("loss component weights must be non-negative")

        self.mode = mode
        self.localization_weight = float(localization_weight)
        self.class_dice_weight = float(class_dice_weight)
        weight = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        if weight is not None and weight.numel() != num_classes:
            raise ValueError("class_weights length must equal num_classes")
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.control_dice = MultiClassDiceLoss(
            num_classes=num_classes,
            smooth=smooth,
            include_background=False,
        )
        self.localization = BinaryForegroundOverlapLoss(
            mode=localization,
            smooth=smooth,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
        )
        self.presence_dice = PresenceAwareClassDiceLoss(
            num_classes=num_classes,
            smooth=smooth,
        )

    def forward(self, inputs, targets, return_components=False):
        ce_loss = self.ce(inputs, targets)
        if self.mode == "control":
            class_dice_loss = self.control_dice(inputs, targets)
            localization_loss = inputs.sum() * 0.0
            total = ce_loss + class_dice_loss
        else:
            localization_loss = self.localization(inputs, targets)
            class_dice_loss = self.presence_dice(inputs, targets)
            total = (
                ce_loss
                + self.localization_weight * localization_loss
                + self.class_dice_weight * class_dice_loss
            )

        if not return_components:
            return total
        return total, {
            "total": total,
            "cross_entropy": ce_loss,
            "binary_localization": localization_loss,
            "class_dice": class_dice_loss,
        }


class DeepSupervisionLoss(nn.Module):
    """Apply one stage loss to P1/P2 using a configurable normalized ratio."""

    def __init__(self, stage_loss, p1_weight=0.5, p2_weight=0.5):
        super().__init__()
        if p1_weight < 0 or p2_weight < 0 or p1_weight + p2_weight <= 0:
            raise ValueError("P1/P2 weights must be non-negative with a positive sum")
        total_weight = float(p1_weight + p2_weight)
        self.stage_loss = stage_loss
        self.p1_weight = float(p1_weight) / total_weight
        self.p2_weight = float(p2_weight) / total_weight

    def forward(self, p1, p2, targets, return_components=False):
        if return_components:
            p1_loss, p1_components = self.stage_loss(p1, targets, return_components=True)
            p2_loss, p2_components = self.stage_loss(p2, targets, return_components=True)
        else:
            p1_loss = self.stage_loss(p1, targets)
            p2_loss = self.stage_loss(p2, targets)

        total = self.p1_weight * p1_loss + self.p2_weight * p2_loss
        if not return_components:
            return total

        components = {
            "total": total,
            "p1_stage": p1_loss,
            "p2_stage": p2_loss,
            "p1_weight": self.p1_weight,
            "p2_weight": self.p2_weight,
        }
        components.update({f"p1_{name}": value for name, value in p1_components.items()})
        components.update({f"p2_{name}": value for name, value in p2_components.items()})
        return total, components


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
