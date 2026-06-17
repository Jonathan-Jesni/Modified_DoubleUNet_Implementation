import os
import csv
from glob import glob

import cv2
import numpy as np
import torch

from CBIS_model import build_doubleunet
from utils import calculate_metrics


CHECKPOINT_PATH = "files/CBIS_checkpoint.pth.zip"

IMAGE_DIR = "dataset_seg_CBIS/test/images"
MASK_DIR = "dataset_seg_CBIS/test/masks"

OUTPUT_DIR = "files/predictions_CBIS"
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


def load_image(image_path, size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Resize to 256x256 immediately so the visualization matches prediction size
    image_resized = cv2.resize(image, size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)

    # FIXED: Just divide by 255.0. 
    # REMOVED: (image_norm - 0.5) / 0.5
    image_norm = image_rgb.astype(np.float32) / 255.0

    image_tensor = torch.from_numpy(np.transpose(image_norm, (2, 0, 1))).unsqueeze(0).float()

    # Return the resized grayscale version for the panel visualization
    return image_resized, image_resized, image_tensor


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
    # Resize the high-res original image to match the 256x256 mask
    base_resized = cv2.resize(base_gray, (mask.shape[1], mask.shape[0]))
    base = cv2.cvtColor(base_resized, cv2.COLOR_GRAY2BGR)
    
    mask_color = colorize_mask(mask)
    overlay = base.copy()
    lesion = mask > 0

    # Ensure float math for blending to avoid overflow/underflow
    overlay[lesion] = (
        base[lesion].astype(float) * (1 - alpha) + 
        mask_color[lesion].astype(float) * alpha
    ).astype(np.uint8)

    return overlay


def make_panel(img, gt, prob, pred, metrics, name):
    # 'img' is now 256x256 from the updated load_image above
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gt_col = colorize_mask(gt)
    pred_col = colorize_mask(pred)
    
    # Use a Heatmap for the probability map so it looks professional
    prob_col = cv2.applyColorMap(prob, cv2.COLORMAP_JET)

    # Now all components are identical in size (256, 256, 3)
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
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading model...")
    model = build_doubleunet()
    # Loading weights
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))
    image_paths = [p for p in image_paths if "mass" in os.path.basename(p).lower()]

    if not image_paths:
        raise RuntimeError("No MASS images found")

    print(f"[INFO] Found {len(image_paths)} test images")

    results = []

    summary_metrics = {
        "jaccard": [],
        "f1": [],
        "recall": [],
        "precision": [],
    }

    with torch.no_grad():
        for i, path in enumerate(image_paths, start=1):
            name = os.path.basename(path)

            mask_path = find_matching_mask(name)
            if mask_path is None:
                continue

            gray, _, tensor = load_image(path, IMAGE_SIZE)
            tensor = tensor.to(device)

            gt = load_gt_mask(mask_path, IMAGE_SIZE)

            # DoubleU-Net returns two outputs, we use the final one (p2)
            _, p2 = model(tensor)
            prob, pred = tensor_to_prediction(p2)

            gt_tensor = torch.from_numpy(gt).to(device)
            pred_tensor = torch.from_numpy(pred).to(device)
            jac, f1, rec, prec = calculate_metrics(gt_tensor, pred_tensor)

            # Sum probabilities of lesion classes (1 and 2) for heat map
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
            
            summary_metrics["jaccard"].append(jac)
            summary_metrics["f1"].append(f1)
            summary_metrics["recall"].append(rec)
            summary_metrics["precision"].append(prec)

            print(f"[{i:03d}/{len(image_paths):03d}] {name} | Jaccard: {jac:.4f} | F1: {f1:.4f}")

    write_csv(results)

    if len(results) > 0:
        print("\n" + "=" * 60)
        print("[SUMMARY - MULTI-CLASS]")
        print(f"Images evaluated : {len(results)}")
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