# RUN THIS SCRIPT FROM PROJECT ROOT
# Example:
# C:\Users\User\Projects\Modified_DoubleUNet_Implementation>
# python prepare_cbis_masks.py

import os
import io
import json
import zlib
import base64
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Project root = folder containing this script
ROOT = Path(__file__).resolve().parent

# Original downloaded Kaggle dataset lives inside project_root/dataset/
DATASET_ROOT = ROOT / "dataset"

# Input splits inside dataset/
SRC_SPLITS = ["train", "test"]

# Output folder generated in root dir
OUT_ROOT = ROOT / "dataset_seg"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def decode_supervisely_bitmap(b64_data: str) -> np.ndarray:
    """
    Decode Supervisely / DatasetNinja bitmap annotation
    into binary uint8 mask {0,255}
    """
    compressed = base64.b64decode(b64_data)
    decompressed = zlib.decompress(compressed)

    img = Image.open(io.BytesIO(decompressed)).convert("RGBA")
    arr = np.array(img)

    if arr.shape[-1] == 4:
        alpha = arr[:, :, 3]
        mask = (alpha > 0).astype(np.uint8) * 255
    else:
        mask = (np.any(arr[:, :, :3] > 0, axis=-1)).astype(np.uint8) * 255

    return mask


def build_full_mask(json_path: Path) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    h = ann["size"]["height"]
    w = ann["size"]["width"]

    full_mask = np.zeros((h, w), dtype=np.uint8)

    for obj in ann.get("objects", []):
        if obj.get("geometryType") != "bitmap":
            continue

        bmp = obj.get("bitmap", {})
        data = bmp.get("data")
        origin = bmp.get("origin", [0, 0])

        if data is None:
            continue

        patch = decode_supervisely_bitmap(data)

        x, y = origin
        ph, pw = patch.shape[:2]

        x2 = min(x + pw, w)
        y2 = min(y + ph, h)

        patch = patch[: y2 - y, : x2 - x]

        full_mask[y:y2, x:x2] = np.maximum(
            full_mask[y:y2, x:x2],
            patch
        )

    return full_mask


def process_split(split: str):
    img_dir = DATASET_ROOT / split / "img"
    ann_dir = DATASET_ROOT / split / "ann"

    out_img_dir = OUT_ROOT / split / "images"
    out_mask_dir = OUT_ROOT / split / "masks"

    ensure_dir(out_img_dir)
    ensure_dir(out_mask_dir)

    if not img_dir.exists():
        print(f"[ERROR] Missing image folder: {img_dir}")
        return

    if not ann_dir.exists():
        print(f"[ERROR] Missing annotation folder: {ann_dir}")
        return

    json_files = sorted(ann_dir.glob("*.json"))

    print(f"\nProcessing split: {split}")
    print(f"Images folder: {img_dir}")
    print(f"Annotations: {len(json_files)}")

    ok = 0
    missing_img = 0
    unreadable = 0

    for json_path in json_files:
        image_name = json_path.name.replace(".json", "")
        image_path = img_dir / image_name

        if not image_path.exists():
            print(f"[WARN] Missing image: {image_name}")
            missing_img += 1
            continue

        try:
            mask = build_full_mask(json_path)
        except Exception as e:
            print(f"[WARN] Failed mask build: {json_path.name} | {e}")
            continue

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[WARN] Could not read image: {image_path}")
            unreadable += 1
            continue

        out_img_path = out_img_dir / image_name
        out_mask_path = out_mask_dir / image_name

        cv2.imwrite(str(out_img_path), img)
        cv2.imwrite(str(out_mask_path), mask)

        ok += 1

        if ok % 100 == 0:
            print(f"Saved {ok} samples...")

    print(f"\nDone {split}")
    print(f"Saved       : {ok}")
    print(f"Missing img : {missing_img}")
    print(f"Unreadable  : {unreadable}")


def main():
    print("=" * 60)
    print("CBIS-DDSM MASK PREPARATION")
    print("=" * 60)
    print(f"Project Root : {ROOT}")
    print(f"Dataset Root : {DATASET_ROOT}")
    print(f"Output Root  : {OUT_ROOT}")

    ensure_dir(OUT_ROOT)

    for split in SRC_SPLITS:
        process_split(split)

    print("\nFinished.")
    print(f"Generated dataset at:\n{OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()