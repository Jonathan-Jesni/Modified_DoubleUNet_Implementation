import os
import csv
from glob import glob

import cv2
import numpy as np
import torch

from CBIS_model import build_doubleunet


CHECKPOINT_PATH = "files/CBIS_checkpoint.pth"

IMAGE_DIR = "dataset_seg_CBIS/test/images"
MASK_DIR = "dataset_seg_CBIS/test/masks"

OUTPUT_DIR = "files/predictions"
MASK_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pred_masks")
PROB_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "prob_maps")
OVERLAY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "overlays")
PANEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "panels")
CSV_PATH = os.path.join(OUTPUT_DIR, "metrics.csv")

IMAGE_SIZE = (256, 256)
NUM_CLASSES = 3
USE_P1_TOO = False


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROB_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OVERLAY_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PANEL_OUTPUT_DIR, exist_ok=True)


def calculate_multiclass_metrics(y_true, y_pred, num_classes=3):
    jaccards, dices, recalls, precisions = [], [], [], []

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
        return 0.0, 0.0, 0.0, 0.0

    return (
        float(np.mean(jaccards)),
        float(np.mean(dices)),
        float(np.mean(recalls)),
        float(np.mean(precisions)),
    )


def load_image(image_path, size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    original_gray = image.copy()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_resized = cv2.resize(image_rgb, size)

    image_norm = image_resized.astype(np.float32) / 255.0
    image_norm = (image_norm - 0.5) / 0.5

    image_tensor = torch.from_numpy(np.transpose(image_norm, (2, 0, 1))).unsqueeze(0).float()

    return original_gray, image_resized, image_tensor


def load_gt_mask(mask_path, size):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return np.clip(mask, 0, NUM_CLASSES - 1).astype(np.uint8)


def tensor_to_prediction(pred_tensor):
    prob = torch.softmax(pred_tensor, dim=1).cpu().numpy()[0]
    pred_classes = np.argmax(prob, axis=0).astype(np.uint8)
    return prob, pred_classes


def colorize_mask(mask):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color[mask == 1] = (0, 255, 0)
    color[mask == 2] = (0, 0, 255)
    return color


def make_overlay(base_gray, mask, alpha=0.45):
    base = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    mask_color = colorize_mask(mask)

    overlay = base.copy()
    lesion = mask > 0

    overlay[lesion] = (
        base[lesion] * (1 - alpha) + mask_color[lesion] * alpha
    ).astype(np.uint8)

    return overlay


def make_panel(img, gt, prob, pred, metrics, name):
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gt_col = colorize_mask(gt)
    pred_col = colorize_mask(pred)
    prob_col = cv2.applyColorMap(prob, cv2.COLORMAP_JET)

    top = np.hstack([base, gt_col, prob_col])
    bottom = np.hstack([pred_col, make_overlay(img, gt), make_overlay(img, pred)])

    return np.vstack([top, bottom])


def write_csv(rows):
    if not rows:
        return

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def find_matching_mask(name):
    path = os.path.join(MASK_DIR, name)
    return path if os.path.exists(path) else None


def main():
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_doubleunet()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 🔥 MASS FILTER APPLIED HERE
    image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))
    image_paths = [p for p in image_paths if "mass" in os.path.basename(p).lower()]

    if not image_paths:
        raise RuntimeError("No MASS images found")

    print(f"[INFO] MASS images: {len(image_paths)}")

    results = []

    with torch.no_grad():
        for i, path in enumerate(image_paths):
            name = os.path.basename(path)

            mask_path = find_matching_mask(name)
            if mask_path is None:
                continue

            gray, _, tensor = load_image(path, IMAGE_SIZE)
            tensor = tensor.to(device)

            gt = load_gt_mask(mask_path, IMAGE_SIZE)

            _, p2 = model(tensor)
            prob, pred = tensor_to_prediction(p2)

            jac, f1, rec, prec = calculate_multiclass_metrics(gt, pred)

            prob_vis = (np.sum(prob[1:], axis=0) * 255).astype(np.uint8)

            cv2.imwrite(os.path.join(MASK_OUTPUT_DIR, name), pred)
            cv2.imwrite(os.path.join(PROB_OUTPUT_DIR, name), prob_vis)
            cv2.imwrite(os.path.join(OVERLAY_OUTPUT_DIR, name), make_overlay(gray, pred))

            panel = make_panel(gray, gt, prob_vis, pred, {"dice": f1, "iou": jac}, name)
            cv2.imwrite(os.path.join(PANEL_OUTPUT_DIR, name), panel)

            results.append({
                "image": name,
                "dice": f1,
                "iou": jac,
                "recall": rec,
                "precision": prec
            })

            print(f"[{i+1}/{len(image_paths)}] {name} | Dice: {f1:.4f}")

    write_csv(results)


if __name__ == "__main__":
    main()