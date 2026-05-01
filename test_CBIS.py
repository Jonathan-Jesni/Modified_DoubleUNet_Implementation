import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from glob import glob
from operator import add

import cv2
import numpy as np
import torch
from tqdm import tqdm

from CBIS_model import build_doubleunet
from utils import create_dir, seeding


NUM_CLASSES = 3
SIZE = (256, 256)

DATASET_PATH = "dataset_seg_CBIS"
CHECKPOINT_PATH = "files/CBIS_checkpoint.pth.zip"
SAVE_PATH = "results_CBIS"


COLOR_MAP = {
    0: (0, 0, 0),       # background = black
    1: (0, 255, 0),     # benign = green
    2: (0, 0, 255),     # malignant = red
}


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("\n================ MODEL INFO ================")
    print(f"Total Parameters        : {total_params:,}")
    print(f"Trainable Parameters    : {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    print("===========================================\n")


def load_data(path):
    def get_split_data(split_name):
        img_dir = os.path.join(path, split_name, "images")
        mask_dir = os.path.join(path, split_name, "masks")

        images = sorted(glob(os.path.join(img_dir, "*.png")))
        masks = sorted(glob(os.path.join(mask_dir, "*.png")))

        image_dict = {os.path.basename(x): x for x in images}
        mask_dict = {os.path.basename(y): y for y in masks}

        common_names = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

        images = [image_dict[name] for name in common_names]
        masks = [mask_dict[name] for name in common_names]

        return images, masks

    train_x, train_y = get_split_data("train")
    valid_x, valid_y = get_split_data("val")
    test_x, test_y = get_split_data("test")

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def class_to_gray(mask):
    mask = (mask * 127).astype(np.uint8)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=2)
    return mask


def class_to_color(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id, color in COLOR_MAP.items():
        color_mask[mask == class_id] = color

    return color_mask


def make_overlay(image_rgb, mask_classes, alpha=0.45):
    mask_color = class_to_color(mask_classes)
    overlay = image_rgb.copy()

    lesion = mask_classes > 0

    overlay[lesion] = (
        image_rgb[lesion].astype(np.float32) * (1 - alpha)
        + mask_color[lesion].astype(np.float32) * alpha
    ).astype(np.uint8)

    return overlay


def calculate_multiclass_metrics(y_true, y_pred, num_classes=3):
    y_true = y_true.detach().cpu().numpy().astype(np.uint8)
    y_pred = y_pred.detach().cpu().numpy().astype(np.uint8)

    jaccards = []
    dices = []
    recalls = []
    precisions = []

    for cls in range(1, num_classes):
        true_cls = y_true == cls
        pred_cls = y_pred == cls

        tp = np.logical_and(true_cls, pred_cls).sum()
        fp = np.logical_and(~true_cls, pred_cls).sum()
        fn = np.logical_and(true_cls, ~pred_cls).sum()

        if true_cls.sum() == 0 and pred_cls.sum() == 0:
            continue

        jaccard = tp / (tp + fp + fn + 1e-7)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)

        jaccards.append(jaccard)
        dices.append(dice)
        recalls.append(recall)
        precisions.append(precision)

    if len(jaccards) == 0:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        float(np.mean(jaccards)),
        float(np.mean(dices)),
        float(np.mean(recalls)),
        float(np.mean(precisions)),
    ]


def dice_per_class(pred, target, class_id, eps=1e-7):
    pred_c = pred == class_id
    target_c = target == class_id

    intersection = np.logical_and(pred_c, target_c).sum()
    total = pred_c.sum() + target_c.sum()

    if total == 0:
        return np.nan

    return (2 * intersection + eps) / (total + eps)


def iou_per_class(pred, target, class_id, eps=1e-7):
    pred_c = pred == class_id
    target_c = target == class_id

    intersection = np.logical_and(pred_c, target_c).sum()
    union = np.logical_or(pred_c, target_c).sum()

    if union == 0:
        return np.nan

    return (intersection + eps) / (union + eps)


def print_score(metrics_score, num_samples):
    jaccard = metrics_score[0] / num_samples
    f1 = metrics_score[1] / num_samples
    recall = metrics_score[2] / num_samples
    precision = metrics_score[3] / num_samples

    print(
        f"Jaccard/IoU: {jaccard:1.4f} - "
        f"Dice/F1: {f1:1.4f} - "
        f"Recall: {recall:1.4f} - "
        f"Precision: {precision:1.4f}"
    )


def evaluate(model, save_path, test_x, test_y, size, device):
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0]
    metrics_score_2 = [0.0, 0.0, 0.0, 0.0]

    class_dice = {0: [], 1: [], 2: []}
    class_iou = {0: [], 1: [], 2: []}

    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = os.path.basename(x)

        image_gray = cv2.imread(x, cv2.IMREAD_GRAYSCALE)

        if image_gray is None:
            raise ValueError(f"Failed to read image: {x}")

        image_gray = cv2.resize(image_gray, size, interpolation=cv2.INTER_LINEAR)
        image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

        save_img = image_rgb.copy()

        # FIXED: Just divide by 255.0 to match training normalizer
        image = image_rgb.astype(np.float32) / 255.0
        # REMOVED: image = (image - 0.5) / 0.5 
        
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        image_tensor = torch.from_numpy(image).float().to(device)

        mask_raw = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        if mask_raw is None:
            raise ValueError(f"Failed to read mask: {y}")

        mask_raw = cv2.resize(mask_raw, size, interpolation=cv2.INTER_NEAREST)

        final_mask = np.clip(mask_raw, 0, 2).astype(np.uint8)
        mask_tensor = torch.from_numpy(final_mask).long().to(device)

        with torch.no_grad():
            start_time = time.time()

            y_pred1, y_pred2 = model(image_tensor)

            end_time = time.time() - start_time
            time_taken.append(end_time)

            y_pred1 = torch.softmax(y_pred1, dim=1)
            y_pred2 = torch.softmax(y_pred2, dim=1)

            y_pred1_classes = torch.argmax(y_pred1, dim=1)[0]
            y_pred2_classes = torch.argmax(y_pred2, dim=1)[0]

            score_1 = calculate_multiclass_metrics(mask_tensor, y_pred1_classes, NUM_CLASSES)
            score_2 = calculate_multiclass_metrics(mask_tensor, y_pred2_classes, NUM_CLASSES)

            metrics_score_1 = list(map(add, metrics_score_1, score_1))
            metrics_score_2 = list(map(add, metrics_score_2, score_2))

            pred1_np = y_pred1_classes.detach().cpu().numpy().astype(np.uint8)
            pred2_np = y_pred2_classes.detach().cpu().numpy().astype(np.uint8)

            for class_id in [0, 1, 2]:
                d = dice_per_class(pred2_np, final_mask, class_id)
                j = iou_per_class(pred2_np, final_mask, class_id)

                if not np.isnan(d):
                    class_dice[class_id].append(d)
                if not np.isnan(j):
                    class_iou[class_id].append(j)

        gt_gray = class_to_gray(final_mask)
        pred1_gray = class_to_gray(pred1_np)
        pred2_gray = class_to_gray(pred2_np)

        gt_color = class_to_color(final_mask)
        pred1_color = class_to_color(pred1_np)
        pred2_color = class_to_color(pred2_np)

        gt_overlay = make_overlay(save_img, final_mask)
        pred_overlay = make_overlay(save_img, pred2_np)

        line = np.ones((size[1], 10, 3), dtype=np.uint8) * 255

        joint_gray = np.concatenate(
            [save_img, line, gt_gray, line, pred1_gray, line, pred2_gray],
            axis=1
        )

        joint_color = np.concatenate(
            [save_img, line, gt_color, line, pred1_color, line, pred2_color],
            axis=1
        )

        joint_overlay = np.concatenate(
            [save_img, line, gt_overlay, line, pred_overlay],
            axis=1
        )

        cv2.imwrite(f"{save_path}/joint_gray/{name}", joint_gray)
        cv2.imwrite(f"{save_path}/joint_color/{name}", joint_color)
        cv2.imwrite(f"{save_path}/joint_overlay/{name}", joint_overlay)

        cv2.imwrite(f"{save_path}/gt_overlay/{name}", gt_overlay)
        cv2.imwrite(f"{save_path}/pred_overlay/{name}", pred_overlay)

        cv2.imwrite(f"{save_path}/mask1_gray/{name}", pred1_gray)
        cv2.imwrite(f"{save_path}/mask2_gray/{name}", pred2_gray)

        cv2.imwrite(f"{save_path}/mask1_color/{name}", pred1_color)
        cv2.imwrite(f"{save_path}/mask2_color/{name}", pred2_color)

    print("\n--- Output 1 Scores ---")
    print_score(metrics_score_1, len(test_x))

    print("\n--- Output 2 Scores Final Output ---")
    print_score(metrics_score_2, len(test_x))

    print("\n--- Per-Class Final Output Scores ---")

    class_names = {
        0: "Background",
        1: "Benign lesion",
        2: "Malignant lesion",
    }

    for class_id, class_name in class_names.items():
        mean_dice = np.mean(class_dice[class_id]) if class_dice[class_id] else 0.0
        mean_iou = np.mean(class_iou[class_id]) if class_iou[class_id] else 0.0

        print(f"{class_name}: Dice = {mean_dice:.4f}, IoU = {mean_iou:.4f}")

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1 / mean_time_taken

    print(f"\nMean FPS: {mean_fps:1.2f}")
    print(f"Results saved at: {save_path}")


if __name__ == "__main__":
    seeding(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = build_doubleunet()
    model = model.to(device)

    print_model_parameters(model)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(DATASET_PATH)

    # ADDED: Filter only MASS images for test evaluation to match predictions
    test_x_filtered, test_y_filtered = [], []
    for tx, ty in zip(test_x, test_y):
        if "mass" in os.path.basename(tx).lower():
            test_x_filtered.append(tx)
            test_y_filtered.append(ty)
            
    test_x, test_y = test_x_filtered, test_y_filtered

    print(f"Train images: {len(train_x)}")
    print(f"Val images  : {len(valid_x)}")
    print(f"Test images : {len(test_x)} (Filtered for MASS)")
    print(f"Test masks  : {len(test_y)} (Filtered for MASS)")

    if len(test_x) == 0:
        raise RuntimeError("No MASS test images found.")

    for item in [
        "mask1_gray",
        "mask2_gray",
        "mask1_color",
        "mask2_color",
        "joint_gray",
        "joint_color",
        "joint_overlay",
        "gt_overlay",
        "pred_overlay",
    ]:
        create_dir(f"{SAVE_PATH}/{item}")

    evaluate(model, SAVE_PATH, test_x, test_y, SIZE, device)