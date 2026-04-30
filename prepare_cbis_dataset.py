from pathlib import Path
import json
import base64
import zlib

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
CBIS_ROOT = ROOT / "CBIS_dataset"
OUT_ROOT = ROOT / "dataset_seg"

RANDOM_STATE = 42

# Mask classes:
# 0 = background
# 1 = benign/default lesion
# 2 = malignant


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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


# NEW: color mask function
def colorize_mask(mask):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    color[mask == 1] = (0, 255, 0)   # benign = green
    color[mask == 2] = (0, 0, 255)   # malignant = red

    return color


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

    ensure_dir(out_img_dir)
    ensure_dir(out_mask_dir)
    ensure_dir(out_color_mask_dir)

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

        color_mask = colorize_mask(mask)

        file_name = f"{image_path.stem}_{idx:04d}.png"

        cv2.imwrite(str(out_img_dir / file_name), img)
        cv2.imwrite(str(out_mask_dir / file_name), mask)
        cv2.imwrite(str(out_color_mask_dir / file_name), color_mask)


def main():
    ensure_dir(OUT_ROOT)

    original_train_samples, original_test_samples = gather_samples()

    print(f"Original train samples: {len(original_train_samples)}")
    print(f"Original test samples : {len(original_test_samples)}")

    train_samples, val_samples = train_test_split(
        original_train_samples,
        test_size=0.10,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    test_samples = original_test_samples

    print(f"Train: {len(train_samples)}")
    print(f"Val  : {len(val_samples)}")
    print(f"Test : {len(test_samples)}")

    save_split("train", train_samples)
    save_split("val", val_samples)
    save_split("test", test_samples)

    print(f"\nFinished. Output at: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()