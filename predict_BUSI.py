import os
import csv
from glob import glob

import cv2
import numpy as np
import torch

from BUSI_model import build_doubleunet
from utils import calculate_metrics


# =========================================================
# CONFIG
# =========================================================
CHECKPOINT_PATH = "files/BUSI_checkpoint.pth.zip"

IMAGE_DIR = "dataset_seg_BUSI/test/images"
MASK_DIR = "dataset_seg_BUSI/test/masks"

OUTPUT_DIR = "files/predictions"
MASK_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pred_masks")
PROB_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "prob_maps")
OVERLAY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "overlays")
PANEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "panels")
CSV_PATH = os.path.join(OUTPUT_DIR, "metrics.csv")

IMAGE_SIZE = (256, 256)
USE_P1_TOO = False


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


def tensor_to_prediction(pred_tensor):
    prob = torch.softmax(pred_tensor, dim=1).detach().cpu().numpy()[0]
    pred_classes = np.argmax(prob, axis=0).astype(np.uint8)
    return prob, pred_classes


def colorize_mask(mask_classes):
    """
    Class 0: Background - black
    Class 1: Benign     - green
    Class 2: Malignant  - red
    """
    h, w = mask_classes.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    color[mask_classes == 1] = (0, 255, 0)
    color[mask_classes == 2] = (0, 0, 255)

    return color


def make_overlay(base_gray, mask_classes, alpha=0.45):
    base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    mask_color = colorize_mask(mask_classes)

    overlay = base_bgr.copy()
    lesion = mask_classes > 0

    # Safe blending instead of cv2.addWeighted on boolean-indexed arrays
    overlay[lesion] = (
        base_bgr[lesion].astype(np.float32) * (1 - alpha)
        + mask_color[lesion].astype(np.float32) * alpha
    ).astype(np.uint8)

    return overlay


def make_gt_overlay(base_gray, gt_classes, alpha=0.45):
    base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)

    gt_color = np.zeros_like(base_bgr)
    gt_color[gt_classes == 1] = (255, 0, 0)      # blue for benign GT
    gt_color[gt_classes == 2] = (255, 0, 255)    # magenta for malignant GT

    overlay = base_bgr.copy()
    lesion = gt_classes > 0

    # Safe blending instead of cv2.addWeighted on boolean-indexed arrays
    overlay[lesion] = (
        base_bgr[lesion].astype(np.float32) * (1 - alpha)
        + gt_color[lesion].astype(np.float32) * alpha
    ).astype(np.uint8)

    return overlay


def add_title(img, title):
    img = img.copy()

    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (30, 30, 30), -1)
    cv2.putText(
        img,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return img


def make_panel(image_gray_resized, gt_classes, prob_u8, pred_classes, metrics, name):
    original_bgr = cv2.cvtColor(image_gray_resized, cv2.COLOR_GRAY2BGR)

    gt_bgr = colorize_mask(gt_classes)
    pred_bgr = colorize_mask(pred_classes)

    prob_bgr = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)

    gt_overlay = make_gt_overlay(image_gray_resized, gt_classes)
    pred_overlay = make_overlay(image_gray_resized, pred_classes)

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

    return np.vstack([top, bottom])


def write_csv(rows, csv_path):
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_gt_classes(mask_path, image_name, size):
    mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise ValueError(f"Failed to read mask: {mask_path}")

    mask_raw = cv2.resize(mask_raw, size, interpolation=cv2.INTER_NEAREST)

    gt_classes = np.zeros(mask_raw.shape, dtype=np.int64)
    tumor_pixels = mask_raw > 127

    lower_name = image_name.lower()

    if lower_name.startswith("benign"):
        gt_classes[tumor_pixels] = 1
    elif lower_name.startswith("malignant"):
        gt_classes[tumor_pixels] = 2
    else:
        gt_classes[tumor_pixels] = 0

    return gt_classes


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print("[INFO] Loading model...")
    model = build_doubleunet()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))

    if len(image_paths) == 0:
        image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in: {IMAGE_DIR}")

    print(f"[INFO] Found {len(image_paths)} test images")

    all_rows = []

    summary_metrics = {
        "jaccard": [],
        "f1": [],
        "recall": [],
        "precision": [],
    }

    with torch.no_grad():
        for idx, image_path in enumerate(image_paths, start=1):
            name = os.path.basename(image_path)

            # FIXED: prepared dataset uses same image and mask filename
            mask_path = os.path.join(MASK_DIR, name)

            if not os.path.exists(mask_path):
                print(f"[WARNING] Mask not found for {name}, skipping")
                continue

            original_gray, image_resized_rgb, image_tensor = load_image(
                image_path,
                IMAGE_SIZE
            )

            image_tensor = image_tensor.to(device)

            image_gray_resized = cv2.resize(
                original_gray,
                IMAGE_SIZE,
                interpolation=cv2.INTER_LINEAR
            )

            gt_classes = build_gt_classes(
                mask_path=mask_path,
                image_name=name,
                size=IMAGE_SIZE
            )

            p1, p2 = model(image_tensor)

            prob_map, pred_classes = tensor_to_prediction(p2)

            gt_tensor = torch.from_numpy(gt_classes).to(device)
            pred_tensor = torch.from_numpy(pred_classes).to(device)

            m_jaccard, m_f1, m_recall, m_precision = calculate_metrics(
                gt_tensor,
                pred_tensor
            )

            pred_save = (pred_classes * 127).astype(np.uint8)
            cv2.imwrite(os.path.join(MASK_OUTPUT_DIR, name), pred_save)

            prob_vis = (np.max(prob_map[1:], axis=0) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(PROB_OUTPUT_DIR, name), prob_vis)

            overlay = make_overlay(image_gray_resized, pred_classes)
            cv2.imwrite(os.path.join(OVERLAY_OUTPUT_DIR, name), overlay)

            panel_metrics = {
                "dice": m_f1,
                "iou": m_jaccard,
            }

            panel = make_panel(
                image_gray_resized=image_gray_resized,
                gt_classes=gt_classes,
                prob_u8=prob_vis,
                pred_classes=pred_classes,
                metrics=panel_metrics,
                name=name
            )

            cv2.imwrite(os.path.join(PANEL_OUTPUT_DIR, name), panel)

            row = {
                "image_name": name,
                "jaccard": m_jaccard,
                "f1_score": m_f1,
                "recall": m_recall,
                "precision": m_precision,
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

    write_csv(all_rows, CSV_PATH)

    if len(all_rows) > 0:
        print("\n" + "=" * 60)
        print("[SUMMARY - MULTI-CLASS]")
        print(f"Images evaluated : {len(all_rows)}")
        print(f"Mean Jaccard     : {np.mean(summary_metrics['jaccard']):.4f}")
        print(f"Mean F1 / Dice   : {np.mean(summary_metrics['f1']):.4f}")
        print(f"Mean Recall      : {np.mean(summary_metrics['recall']):.4f}")
        print(f"Mean Precision   : {np.mean(summary_metrics['precision']):.4f}")
        print(f"CSV saved at     : {CSV_PATH}")
        print("=" * 60)
    else:
        print("[INFO] No images were evaluated.")


if __name__ == "__main__":
    main()