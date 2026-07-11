from pathlib import Path
import json
import base64
import zlib
import re

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent
CBIS_ROOT = ROOT / "CBIS_dataset"
OUT_ROOT = ROOT / "dataset_seg_CBIS"

RANDOM_STATE = 42

# Mask classes:
# 0 = background
# 1 = benign/default lesion
# 2 = malignant


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def extract_patient_id(image_path: Path) -> str:
    match = re.search(r"P_\d+", str(image_path))
    if match is None:
        raise ValueError(f"Could not extract a CBIS patient ID from: {image_path}")
    return match.group(0)


def split_by_patient(samples):
    groups = [extract_patient_id(sample["image_path"]) for sample in samples]
    all_indices = list(range(len(samples)))

    outer_split = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_STATE
    )
    train_indices, temp_indices = next(outer_split.split(all_indices, groups=groups))
    train_samples = [samples[index] for index in train_indices]
    temp_samples = [samples[index] for index in temp_indices]

    temp_groups = [groups[index] for index in temp_indices]
    inner_split = GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=RANDOM_STATE
    )
    val_indices, test_indices = next(
        inner_split.split(list(range(len(temp_samples))), groups=temp_groups)
    )
    val_samples = [temp_samples[index] for index in val_indices]
    test_samples = [temp_samples[index] for index in test_indices]

    return train_samples, val_samples, test_samples


def verify_patient_split(train_samples, val_samples, test_samples):
    patient_sets = {
        "train": {extract_patient_id(sample["image_path"]) for sample in train_samples},
        "val": {extract_patient_id(sample["image_path"]) for sample in val_samples},
        "test": {extract_patient_id(sample["image_path"]) for sample in test_samples},
    }
    overlaps = {
        "train/val": patient_sets["train"] & patient_sets["val"],
        "train/test": patient_sets["train"] & patient_sets["test"],
        "val/test": patient_sets["val"] & patient_sets["test"],
    }
    print(
        "Patient split verification: "
        f"train/val={len(overlaps['train/val'])}, "
        f"train/test={len(overlaps['train/test'])}, "
        f"val/test={len(overlaps['val/test'])}"
    )
    if any(overlaps.values()):
        raise RuntimeError(f"Patient leakage detected: {overlaps}")

    return patient_sets


def get_ann_path(split_dir: Path, image_path: Path):
    return split_dir / "ann" / f"{image_path.name}.json"


def infer_class_id(ann_data, ann_path: Path):
    tag_names = []

    for tag in ann_data.get("tags", []):
        tag_names.append(str(tag.get("name", "")).lower())
        tag_names.append(str(tag.get("value", "")).lower())

    text = " ".join(tag_names).lower()

    if "malignant" in text:
        return 2

    if "benign" in text:
        return 1

    print(f"[WARN] No benign/malignant tag found in: {ann_path.name}. Using class 1.")
    return 1


def decode_bitmap(bitmap_obj):
    encoded = bitmap_obj.get("data", "")
    origin = bitmap_obj.get("origin", [0, 0])

    decoded = base64.b64decode(encoded)
    decompressed = zlib.decompress(decoded)

    arr = np.frombuffer(decompressed, dtype=np.uint8)
    bitmap_img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if bitmap_img is None:
        return None, origin

    if len(bitmap_img.shape) == 3:
        if bitmap_img.shape[2] == 4:
            bitmap_mask = bitmap_img[:, :, 3]
        else:
            bitmap_mask = cv2.cvtColor(bitmap_img, cv2.COLOR_BGR2GRAY)
    else:
        bitmap_mask = bitmap_img

    bitmap_mask = (bitmap_mask > 0).astype(np.uint8)

    return bitmap_mask, origin


def draw_polygon(mask, points, class_id, scale_x, scale_y):
    exterior = points.get("exterior", [])

    if len(exterior) < 3:
        return

    pts = np.array(
        [[int(x * scale_x), int(y * scale_y)] for x, y in exterior],
        dtype=np.int32
    )

    cv2.fillPoly(mask, [pts], class_id)

    for interior in points.get("interior", []):
        if len(interior) < 3:
            continue

        hole = np.array(
            [[int(x * scale_x), int(y * scale_y)] for x, y in interior],
            dtype=np.int32
        )

        cv2.fillPoly(mask, [hole], 0)


def build_mask_from_json(ann_path: Path, image_shape):
    h, w = image_shape[:2]

    with open(ann_path, "r", encoding="utf-8") as f:
        ann_data = json.load(f)

    mask = np.zeros((h, w), dtype=np.uint8)
    class_id = infer_class_id(ann_data, ann_path)

    objects = ann_data.get("objects", [])

    for obj in objects:
        geometry_type = str(obj.get("geometryType", "")).lower()

        if geometry_type == "bitmap" and "bitmap" in obj:
            bitmap_mask, origin = decode_bitmap(obj["bitmap"])

            if bitmap_mask is None:
                print(f"[WARN] Could not decode bitmap in: {ann_path.name}")
                continue

            ox, oy = int(origin[0]), int(origin[1])
            bh, bw = bitmap_mask.shape[:2]

            x1, y1 = ox, oy
            x2, y2 = min(ox + bw, w), min(oy + bh, h)

            crop_w = x2 - x1
            crop_h = y2 - y1

            if crop_w <= 0 or crop_h <= 0:
                continue

            roi = bitmap_mask[:crop_h, :crop_w]
            mask[y1:y2, x1:x2][roi > 0] = class_id

        elif "polygon" in geometry_type and "points" in obj:
            draw_polygon(mask, obj["points"], class_id, 1.0, 1.0)

        elif "rectangle" in geometry_type and "points" in obj:
            exterior = obj["points"].get("exterior", [])

            if len(exterior) >= 2:
                x1, y1 = exterior[0]
                x2, y2 = exterior[1]

                cv2.rectangle(
                    mask,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    class_id,
                    -1
                )

    return mask


def find_breast_bbox(img, margin_frac=0.02):
    """Bounding box of the breast tissue in a full-resolution grayscale mammogram.

    Otsu-thresholds the image to separate breast tissue from the black scanner
    margin, then takes the LARGEST connected component (to ignore small scanner
    artifacts / embedded text labels), expands its bbox by a small margin so the
    crop doesn't clip the breast edge, and falls back to the full image if Otsu
    finds no foreground at all.
    """
    h, w = img.shape[:2]

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)

    # Label 0 is the background component; skip it when picking the largest.
    if num_labels <= 1:
        return 0, 0, w, h

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[largest_label]

    if area <= 0:
        return 0, 0, w, h

    margin_x = int(bw * margin_frac)
    margin_y = int(bh * margin_frac)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + bw + margin_x)
    y2 = min(h, y + bh + margin_y)

    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h

    return x1, y1, x2, y2


def colorize_mask(mask):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    color[mask == 1] = (0, 255, 0)   # benign = green
    color[mask == 2] = (0, 0, 255)   # malignant = red

    return color


def make_overlay(image_gray, mask, alpha=0.4):
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    mask_color = colorize_mask(mask)

    overlay = image_bgr.copy()
    lesion = mask > 0

    overlay[lesion] = (
        image_bgr[lesion].astype(np.float32) * (1 - alpha)
        + mask_color[lesion].astype(np.float32) * alpha
    ).astype(np.uint8)

    return overlay


def gather_samples_from_split(split_name):
    samples = []

    split_dir = CBIS_ROOT / split_name
    img_dir = split_dir / "img"

    if not img_dir.exists():
        print(f"[WARN] Missing image folder: {img_dir}")
        return samples

    image_files = sorted(img_dir.glob("*.png"))

    for image_path in image_files:
        ann_path = get_ann_path(split_dir, image_path)

        if not ann_path.exists():
            print(f"[WARN] Missing annotation for: {image_path.name}")
            continue

        samples.append({
            "source_split": split_name,
            "image_path": image_path,
            "ann_path": ann_path,
        })

    return samples


def gather_samples():
    train_samples = gather_samples_from_split("train")
    test_samples = gather_samples_from_split("test")

    return train_samples, test_samples


def save_split(split_name, samples):
    out_img_dir = OUT_ROOT / split_name / "images"
    out_mask_dir = OUT_ROOT / split_name / "masks"
    out_color_mask_dir = OUT_ROOT / split_name / "color_masks"
    out_overlay_dir = OUT_ROOT / split_name / "overlays"

    ensure_dir(out_img_dir)
    ensure_dir(out_mask_dir)
    ensure_dir(out_color_mask_dir)
    ensure_dir(out_overlay_dir)

    for idx, sample in enumerate(samples):
        image_path = sample["image_path"]
        ann_path = sample["ann_path"]

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        mask = build_mask_from_json(ann_path, image_shape=img.shape)

        if mask is None:
            print(f"[WARN] Could not build mask for: {image_path}")
            continue

        # Crop to the breast ROI (Otsu + largest connected component) so the
        # downstream 256x256 resize in train/test/predict spends its resolution
        # budget on breast tissue instead of the empty black scanner margin.
        # img and mask share the exact same bbox so they stay pixel-aligned.
        x1, y1, x2, y2 = find_breast_bbox(img)
        img = img[y1:y2, x1:x2]
        mask = mask[y1:y2, x1:x2]

        color_mask = colorize_mask(mask)
        overlay = make_overlay(img, mask)

        file_name = f"{image_path.stem}_{idx:04d}.png"

        cv2.imwrite(str(out_img_dir / file_name), img)
        cv2.imwrite(str(out_mask_dir / file_name), mask)
        cv2.imwrite(str(out_color_mask_dir / file_name), color_mask)
        cv2.imwrite(str(out_overlay_dir / file_name), overlay)


def main():
    ensure_dir(OUT_ROOT)

    original_train_samples, original_test_samples = gather_samples()

    # 1. Combine all valid samples from both folders into one big list
    all_samples = original_train_samples + original_test_samples

    print(f"Total usable samples found: {len(all_samples)}")

    patient_examples = [
        f"{sample['image_path'].name} -> {extract_patient_id(sample['image_path'])}"
        for sample in all_samples[:5]
    ]
    print("Patient ID examples: " + "; ".join(patient_examples))

    # Keep the original 80/10/10 target while keeping each patient's images in one split.
    train_samples, val_samples, test_samples = split_by_patient(all_samples)
    patient_sets = verify_patient_split(train_samples, val_samples, test_samples)

    print(f"Unique patients: {len(set().union(*patient_sets.values()))}")
    print(f"Train: {len(train_samples)}")
    print(f"Val  : {len(val_samples)}")
    print(f"Test : {len(test_samples)}")

    save_split("train", train_samples)
    save_split("val", val_samples)
    save_split("test", test_samples)

    print(f"\nFinished. Output at: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()