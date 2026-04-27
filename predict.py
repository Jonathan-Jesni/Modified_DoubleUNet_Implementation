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


def tensor_to_prediction(pred_tensor):
    # Use Softmax because model output is multi-class logits
    prob = torch.softmax(pred_tensor, dim=1).detach().cpu().numpy()[0] 
    # Get discrete class indices (0, 1, or 2)
    pred_classes = np.argmax(prob, axis=0).astype(np.uint8)
    return prob, pred_classes


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


def colorize_mask(mask_classes):
    """
    BGR Color Mapping:
    Class 0 (Background): Black [0, 0, 0]
    Class 1 (Benign): Green [0, 255, 0]
    Class 2 (Malignant): Red [0, 0, 255]
    """
    h, w = mask_classes.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[mask_classes == 1] = (0, 255, 0) 
    color[mask_classes == 2] = (0, 0, 255) 
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
    # Ensure this matches your multiclass checkpoint (3 output channels)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))
    if len(image_paths) == 0:
        # Fallback for .jpg if your test set uses them
        image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")))
        
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in: {IMAGE_DIR}")

    print(f"[INFO] Found {len(image_paths)} test images")

    all_rows = []
    # Using lists to track multi-class averages
    summary_metrics = {"jaccard": [], "f1": [], "recall": [], "precision": []}

    with torch.no_grad():
        for idx, image_path in enumerate(image_paths, start=1):
            name = os.path.basename(image_path)
            # Find matching mask (usually .png)
            mask_name = os.path.splitext(name)[0] + "_mask.png"
            mask_path = os.path.join(MASK_DIR, mask_name)

            if not os.path.exists(mask_path):
                print(f"[WARNING] Mask not found for {name}, skipping")
                continue

            # 1. Load image and mask
            original_gray, image_resized_rgb, image_tensor = load_image(image_path, IMAGE_SIZE)
            image_tensor = image_tensor.to(device)
            
            # Load GT and map to 0, 1, 2
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_raw = cv2.resize(mask_raw, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
            gt_classes = np.zeros(mask_raw.shape, dtype=np.int64)
            tumor_pixels = (mask_raw > 127)
            
            if "benign" in name.lower():
                gt_classes[tumor_pixels] = 1
            elif "malignant" in name.lower():
                gt_classes[tumor_pixels] = 2
            # "normal" stays 0

            image_gray_resized = cv2.resize(original_gray, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

            # 2. Model prediction
            p1, p2 = model(image_tensor)

            # 3. Multi-class conversion using Softmax + Argmax
            prob_map, pred_classes = tensor_to_prediction(p2) 
            
            # 4. Metrics using updated multi-class logic
            # Convert to tensors for calculate_metrics
            m_jaccard, m_f1, m_recall, m_precision = calculate_metrics(
                torch.from_numpy(gt_classes).to(device), 
                torch.from_numpy(pred_classes).to(device)
            )

            # 5. Save Visuals
            # pred_bin_u8 scaled for visibility (0, 127, 254)
            pred_bin_u8 = (pred_classes * 127).astype(np.uint8)
            cv2.imwrite(os.path.join(MASK_OUTPUT_DIR, name), pred_bin_u8)
            
            # Prob map: use the channel with the highest non-background probability for visualization
            prob_vis = (np.max(prob_map[1:], axis=0) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(PROB_OUTPUT_DIR, name), prob_vis)

            overlay = make_overlay(image_gray_resized, pred_classes) # Updated for multi-color
            cv2.imwrite(os.path.join(OVERLAY_OUTPUT_DIR, name), overlay)

            # Prepare metrics dict for the panel
            panel_metrics = {"dice": m_f1, "iou": m_jaccard}
            panel = make_panel(
                image_gray_resized=image_gray_resized,
                gt_bin=gt_classes,
                prob_u8=prob_vis,
                pred_bin_u8=pred_classes,
                metrics=panel_metrics,
                name=name
            )
            cv2.imwrite(os.path.join(PANEL_OUTPUT_DIR, name), panel)

            # 6. Logging
            row = {
                "image_name": name,
                "jaccard": m_jaccard,
                "f1_score": m_f1,
                "recall": m_recall,
                "precision": m_precision
            }
            all_rows.append(row)
            
            summary_metrics["jaccard"].append(m_jaccard)
            summary_metrics["f1"].append(m_f1)
            summary_metrics["recall"].append(m_recall)
            summary_metrics["precision"].append(m_precision)

            print(
                f"[{idx:03d}/{len(image_paths):03d}] {name} | "
                f"Jaccard: {m_jaccard:.4f} | F1: {m_f1:.4f}"
            )

    # Save CSV
    write_csv(all_rows, CSV_PATH)

    # Print summary
    if len(all_rows) > 0:
        print("\n" + "=" * 60)
        print("[SUMMARY - MULTI-CLASS]")
        print(f"Images evaluated : {len(all_rows)}")
        print(f"Mean Jaccard     : {np.mean(summary_metrics['jaccard']):.4f}")
        print(f"Mean F1 (Dice)   : {np.mean(summary_metrics['f1']):.4f}")
        print(f"Mean Recall      : {np.mean(summary_metrics['recall']):.4f}")
        print(f"Mean Precision   : {np.mean(summary_metrics['precision']):.4f}")
        print(f"CSV saved at     : {CSV_PATH}")
        print("=" * 60)
    else:
        print("[INFO] No images were evaluated.")


if __name__ == "__main__":
    main()