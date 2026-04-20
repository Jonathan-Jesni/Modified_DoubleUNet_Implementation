import os
import csv
from glob import glob

import cv2
import numpy as np
import torch

from model import build_doubleunet


# =========================================================
# CONFIG
# =========================================================
CHECKPOINT_PATH = "files/checkpoint.pth.zip"
IMAGE_DIR = "dataset_seg/test/images"
MASK_DIR = "dataset_seg/test/masks"

OUTPUT_DIR = "files/predictions"
MASK_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pred_masks")
PROB_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "prob_maps")
OVERLAY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "overlays")
PANEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "panels")
CSV_PATH = os.path.join(OUTPUT_DIR, "metrics.csv")

IMAGE_SIZE = (256, 256)   # must match training
THRESHOLD = 0.5
USE_P1_TOO = False        # set True if you also want output from p1 saved


# =========================================================
# UTILS
# =========================================================
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROB_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OVERLAY_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PANEL_OUTPUT_DIR, exist_ok=True)
    if USE_P1_TOO:
        os.makedirs(os.path.join(OUTPUT_DIR, "pred_masks_p1"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "prob_maps_p1"), exist_ok=True)


def load_image(image_path, size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    original_gray = image.copy()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_resized = cv2.resize(image_rgb, size, interpolation=cv2.INTER_LINEAR)

    image_norm = image_resized.astype(np.float32) / 255.0
    image_chw = np.transpose(image_norm, (2, 0, 1))
    image_tensor = torch.from_numpy(image_chw).unsqueeze(0).float()

    return original_gray, image_resized, image_tensor


def load_mask(mask_path, size):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")

    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask > 127).astype(np.uint8)
    return mask_bin


def tensor_to_prediction(pred_tensor, threshold=0.5):
    pred_prob = torch.sigmoid(pred_tensor).detach().cpu().numpy()[0, 0]
    pred_bin = (pred_prob >= threshold).astype(np.uint8)

    pred_prob_u8 = (pred_prob * 255).clip(0, 255).astype(np.uint8)
    pred_bin_u8 = (pred_bin * 255).astype(np.uint8)

    return pred_prob, pred_bin, pred_prob_u8, pred_bin_u8


def compute_metrics(y_true, y_pred):
    """
    y_true, y_pred: binary arrays of shape [H, W] with values 0/1
    """
    y_true = y_true.astype(np.uint8).flatten()
    y_pred = y_pred.astype(np.uint8).flatten()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    eps = 1e-7

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    recall = (tp + eps) / (tp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "recall": float(recall),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def colorize_mask(mask_bin_u8):
    """
    Binary mask 0/255 -> BGR visualization
    """
    color = np.zeros((mask_bin_u8.shape[0], mask_bin_u8.shape[1], 3), dtype=np.uint8)
    color[mask_bin_u8 > 0] = (0, 255, 0)  # green
    return color


def make_overlay(base_gray, pred_bin_u8, alpha=0.45):
    """
    base_gray: original grayscale resized to model size
    pred_bin_u8: binary mask uint8 0/255
    """
    base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    mask_color = colorize_mask(pred_bin_u8)

    overlay = base_bgr.copy()
    lesion = pred_bin_u8 > 0
    overlay[lesion] = cv2.addWeighted(base_bgr[lesion], 1 - alpha, mask_color[lesion], alpha, 0)
    return overlay


def make_gt_overlay(base_gray, gt_bin):
    base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    gt_color = np.zeros_like(base_bgr)
    gt_color[gt_bin > 0] = (255, 0, 0)  # blue
    overlay = base_bgr.copy()
    lesion = gt_bin > 0
    overlay[lesion] = cv2.addWeighted(base_bgr[lesion], 0.55, gt_color[lesion], 0.45, 0)
    return overlay


def add_title(img, title):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (30, 30, 30), -1)
    cv2.putText(
        img,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


def make_panel(image_gray_resized, gt_bin, prob_u8, pred_bin_u8, metrics, name):
    """
    Creates a 2x3 research-style panel
    """
    original_bgr = cv2.cvtColor(image_gray_resized, cv2.COLOR_GRAY2BGR)
    gt_u8 = (gt_bin * 255).astype(np.uint8)

    gt_bgr = colorize_mask(gt_u8)
    pred_bgr = colorize_mask(pred_bin_u8)
    prob_bgr = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    pred_overlay = make_overlay(image_gray_resized, pred_bin_u8)
    gt_overlay = make_gt_overlay(image_gray_resized, gt_bin)

    a = add_title(original_bgr, f"Image: {name}")
    b = add_title(gt_bgr, "Ground Truth")
    c = add_title(prob_bgr, "Probability Map")
    d = add_title(pred_bgr, "Predicted Mask")
    e = add_title(gt_overlay, "GT Overlay")
    f = add_title(
        pred_overlay,
        f"Pred Overlay | Dice:{metrics['dice']:.4f} IoU:{metrics['iou']:.4f}"
    )

    top = np.hstack([a, b, c])
    bottom = np.hstack([d, e, f])
    panel = np.vstack([top, bottom])

    return panel


def write_csv(rows, csv_path):
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading model...")
    model = build_doubleunet()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in: {IMAGE_DIR}")

    print(f"[INFO] Found {len(image_paths)} test images")

    all_rows = []
    dice_scores = []
    iou_scores = []
    recall_scores = []
    precision_scores = []
    accuracy_scores = []

    with torch.no_grad():
        for idx, image_path in enumerate(image_paths, start=1):
            name = os.path.basename(image_path)
            mask_path = os.path.join(MASK_DIR, name)

            if not os.path.exists(mask_path):
                print(f"[WARNING] Mask not found for {name}, skipping")
                continue

            # Load image and mask
            original_gray, image_resized_rgb, image_tensor = load_image(image_path, IMAGE_SIZE)
            image_tensor = image_tensor.to(device)

            gt_bin = load_mask(mask_path, IMAGE_SIZE)

            image_gray_resized = cv2.resize(original_gray, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

            # Model prediction
            p1, p2 = model(image_tensor)

            # Final output from p2
            _, pred_bin, prob_u8, pred_bin_u8 = tensor_to_prediction(p2, threshold=THRESHOLD)

            # Optional save p1 too
            if USE_P1_TOO:
                _, pred_bin_p1, prob_u8_p1, pred_bin_u8_p1 = tensor_to_prediction(p1, threshold=THRESHOLD)
                cv2.imwrite(os.path.join(OUTPUT_DIR, "pred_masks_p1", name), pred_bin_u8_p1)
                cv2.imwrite(os.path.join(OUTPUT_DIR, "prob_maps_p1", name), prob_u8_p1)

            # Metrics
            metrics = compute_metrics(gt_bin, pred_bin)

            # Save outputs
            cv2.imwrite(os.path.join(MASK_OUTPUT_DIR, name), pred_bin_u8)
            cv2.imwrite(os.path.join(PROB_OUTPUT_DIR, name), prob_u8)

            overlay = make_overlay(image_gray_resized, pred_bin_u8)
            cv2.imwrite(os.path.join(OVERLAY_OUTPUT_DIR, name), overlay)

            panel = make_panel(
                image_gray_resized=image_gray_resized,
                gt_bin=gt_bin,
                prob_u8=prob_u8,
                pred_bin_u8=pred_bin_u8,
                metrics=metrics,
                name=name
            )
            cv2.imwrite(os.path.join(PANEL_OUTPUT_DIR, name), panel)

            row = {
                "image_name": name,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "accuracy": metrics["accuracy"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
            all_rows.append(row)

            dice_scores.append(metrics["dice"])
            iou_scores.append(metrics["iou"])
            recall_scores.append(metrics["recall"])
            precision_scores.append(metrics["precision"])
            accuracy_scores.append(metrics["accuracy"])

            print(
                f"[{idx:03d}/{len(image_paths):03d}] {name} | "
                f"Dice: {metrics['dice']:.4f} | "
                f"IoU: {metrics['iou']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"Precision: {metrics['precision']:.4f}"
            )

    # Save CSV
    write_csv(all_rows, CSV_PATH)

    # Print summary
    if len(all_rows) > 0:
        print("\n" + "=" * 60)
        print("[SUMMARY]")
        print(f"Images evaluated : {len(all_rows)}")
        print(f"Mean Dice        : {np.mean(dice_scores):.4f}")
        print(f"Mean IoU         : {np.mean(iou_scores):.4f}")
        print(f"Mean Recall      : {np.mean(recall_scores):.4f}")
        print(f"Mean Precision   : {np.mean(precision_scores):.4f}")
        print(f"Mean Accuracy    : {np.mean(accuracy_scores):.4f}")
        print(f"CSV saved at     : {CSV_PATH}")
        print(f"Panels saved at  : {PANEL_OUTPUT_DIR}")
        print("=" * 60)
    else:
        print("[INFO] No images were evaluated.")


if __name__ == "__main__":
    main()