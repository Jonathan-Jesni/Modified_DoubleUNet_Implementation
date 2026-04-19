from pathlib import Path
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
BUSI_ROOT = ROOT / "BUSI_dataset"
OUT_ROOT = ROOT / "dataset_seg"

CLASSES_TO_USE = ["benign", "malignant"]   # add "normal" later if needed
RANDOM_STATE = 42


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def is_base_image(path: Path) -> bool:
    name = path.stem.lower()
    return "_mask" not in name


def collect_mask_paths(class_dir: Path, image_path: Path):
    """
    For image 'benign (100).png', collect:
      benign (100)_mask.png
      benign (100)_mask_1.png
      benign (100)_mask_2.png
      ...
    """
    base = image_path.stem
    mask_paths = sorted(class_dir.glob(f"{base}_mask*.png"))
    return mask_paths


def merge_masks(mask_paths, target_shape=None):
    merged = None

    for mp in mask_paths:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue

        m = (m > 0).astype(np.uint8) * 255

        if merged is None:
            merged = np.zeros_like(m, dtype=np.uint8)

        merged = np.maximum(merged, m)

    if merged is None and target_shape is not None:
        merged = np.zeros(target_shape, dtype=np.uint8)

    return merged


def gather_samples():
    samples = []

    for cls in CLASSES_TO_USE:
        class_dir = BUSI_ROOT / cls
        if not class_dir.exists():
            print(f"[WARN] Missing class folder: {class_dir}")
            continue

        image_files = sorted([p for p in class_dir.glob("*.png") if is_base_image(p)])

        for image_path in image_files:
            mask_paths = collect_mask_paths(class_dir, image_path)

            # Skip samples with no mask for now
            if len(mask_paths) == 0:
                continue

            samples.append({
                "class_name": cls,
                "image_path": image_path,
                "mask_paths": mask_paths,
            })

    return samples


def save_split(split_name, samples):
    out_img_dir = OUT_ROOT / split_name / "images"
    out_mask_dir = OUT_ROOT / split_name / "masks"
    ensure_dir(out_img_dir)
    ensure_dir(out_mask_dir)

    for idx, sample in enumerate(samples):
        image_path = sample["image_path"]
        mask_paths = sample["mask_paths"]
        cls = sample["class_name"]

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        merged_mask = merge_masks(mask_paths, target_shape=img.shape[:2])
        if merged_mask is None:
            print(f"[WARN] Could not build mask for: {image_path}")
            continue

        file_name = f"{cls}_{idx:04d}.png"
        cv2.imwrite(str(out_img_dir / file_name), img)
        cv2.imwrite(str(out_mask_dir / file_name), merged_mask)


def main():
    ensure_dir(OUT_ROOT)

    samples = gather_samples()
    print(f"Total usable samples: {len(samples)}")

    # 80 / 10 / 10 split
    train_samples, temp_samples = train_test_split(
        samples, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
    )
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.50, random_state=RANDOM_STATE, shuffle=True
    )

    print(f"Train: {len(train_samples)}")
    print(f"Val  : {len(val_samples)}")
    print(f"Test : {len(test_samples)}")

    save_split("train", train_samples)
    save_split("val", val_samples)
    save_split("test", test_samples)

    print(f"\nFinished. Output at: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()