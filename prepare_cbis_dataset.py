from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
CBIS_ROOT = ROOT / "CBIS_dataset"
CSV_ROOT = CBIS_ROOT / "csv"
JPEG_ROOT = CBIS_ROOT / "jpeg"
OUT_ROOT = ROOT / "dataset_seg"

RANDOM_STATE = 42

CLASS_MAP = {
    "BENIGN": 1,
    "BENIGN_WITHOUT_CALLBACK": 1,
    "MALIGNANT": 2,
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clean_output():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    for split in ["train", "val", "test"]:
        ensure_dir(OUT_ROOT / split / "images")
        ensure_dir(OUT_ROOT / split / "masks")          # grayscale 0/1/2 for training
        ensure_dir(OUT_ROOT / split / "masks_color")    # colored copy for viewing


def colorize_mask(mask):
    """
    Visualization only:
        0 = background = black
        1 = benign     = green
        2 = malignant  = red
    """
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color[mask == 1] = (0, 255, 0)
    color[mask == 2] = (0, 0, 255)
    return color


def norm_text(x):
    return str(x).strip().replace("\\", "/") if pd.notna(x) else ""


def find_jpeg_from_path(path_text):
    path_text = norm_text(path_text)
    if not path_text:
        return None

    parts = [p for p in path_text.split("/") if p]

    for part in reversed(parts):
        candidate_dir = JPEG_ROOT / part
        if candidate_dir.exists():
            jpgs = sorted(candidate_dir.glob("*.jpg"))
            if jpgs:
                return jpgs[0]

    last = Path(parts[-1]).stem if parts else ""
    if last:
        matches = sorted(JPEG_ROOT.rglob(f"*{last}*.jpg"))
        if matches:
            return matches[0]

    return None


def read_mask(mask_path):
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None

    return (m > 0).astype(np.uint8)


def build_multiclass_mask(mask_paths, class_id, target_shape):
    final = np.zeros(target_shape, dtype=np.uint8)

    for mp in mask_paths:
        m = read_mask(mp)
        if m is None:
            continue

        if m.shape != target_shape:
            m = cv2.resize(
                m,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        final[m > 0] = class_id

    return final


def load_case_csvs():
    files = [
        CSV_ROOT / "calc_case_description_train_set.csv",
        CSV_ROOT / "calc_case_description_test_set.csv",
        CSV_ROOT / "mass_case_description_train_set.csv",
        CSV_ROOT / "mass_case_description_test_set.csv",
    ]

    dfs = []

    for f in files:
        if not f.exists():
            print(f"[WARN] Missing CSV: {f}")
            continue

        df = pd.read_csv(f)
        df["source_csv"] = f.name
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No CBIS case description CSV files found.")

    return pd.concat(dfs, ignore_index=True)


def pick_column(df, possible_names):
    cols = {c.lower().strip(): c for c in df.columns}

    for name in possible_names:
        key = name.lower().strip()
        if key in cols:
            return cols[key]

    return None


def gather_samples():
    df = load_case_csvs()

    patient_col = pick_column(df, ["patient_id", "patient id"])
    pathology_col = pick_column(df, ["pathology"])
    image_col = pick_column(df, ["image file path", "image_file_path"])
    roi_col = pick_column(df, ["roi mask file path", "roi_mask_file_path"])
    cropped_col = pick_column(df, ["cropped image file path", "cropped_image_file_path"])

    if patient_col is None or pathology_col is None or image_col is None:
        raise ValueError(f"Required columns missing. Found columns: {list(df.columns)}")

    samples = []

    for row_idx, row in df.iterrows():
        pathology = str(row[pathology_col]).strip().upper()

        if pathology not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[pathology]
        patient_id = str(row[patient_col]).strip()

        image_path = find_jpeg_from_path(row[image_col])
        if image_path is None:
            print(f"[WARN] Image not found for row {row_idx}")
            continue

        mask_paths = []

        if roi_col is not None:
            mp = find_jpeg_from_path(row[roi_col])
            if mp is not None:
                mask_paths.append(mp)

        if len(mask_paths) == 0 and cropped_col is not None:
            mp = find_jpeg_from_path(row[cropped_col])
            if mp is not None:
                mask_paths.append(mp)

        if len(mask_paths) == 0:
            print(f"[WARN] ROI mask not found for row {row_idx}")
            continue

        samples.append({
            "patient_id": patient_id,
            "class_id": class_id,
            "pathology": pathology,
            "image_path": image_path,
            "mask_paths": mask_paths,
        })

    return samples


def save_split(split_name, samples):
    out_img_dir = OUT_ROOT / split_name / "images"
    out_mask_dir = OUT_ROOT / split_name / "masks"
    out_mask_color_dir = OUT_ROOT / split_name / "masks_color"

    for idx, s in enumerate(samples):
        img = cv2.imread(str(s["image_path"]), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[WARN] Could not read image: {s['image_path']}")
            continue

        mask = build_multiclass_mask(
            mask_paths=s["mask_paths"],
            class_id=s["class_id"],
            target_shape=img.shape,
        )

        mask_color = colorize_mask(mask)

        prefix = "benign" if s["class_id"] == 1 else "malignant"
        safe_patient = s["patient_id"].replace("/", "_").replace("\\", "_")
        name = f"{prefix}_{safe_patient}_{idx:05d}.png"

        cv2.imwrite(str(out_img_dir / name), img)

        # Training mask: keep as grayscale class IDs 0/1/2
        cv2.imwrite(str(out_mask_dir / name), mask)

        # Visualization mask: colored copy only
        cv2.imwrite(str(out_mask_color_dir / name), mask_color)


def patientwise_split(samples):
    patients = sorted(list({s["patient_id"] for s in samples}))

    train_p, temp_p = train_test_split(
        patients,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    val_p, test_p = train_test_split(
        temp_p,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_p, val_p, test_p = set(train_p), set(val_p), set(test_p)

    train_s = [s for s in samples if s["patient_id"] in train_p]
    val_s = [s for s in samples if s["patient_id"] in val_p]
    test_s = [s for s in samples if s["patient_id"] in test_p]

    return train_s, val_s, test_s


def main():
    clean_output()

    samples = gather_samples()

    print(f"Total usable ROI samples: {len(samples)}")

    benign = sum(1 for s in samples if s["class_id"] == 1)
    malignant = sum(1 for s in samples if s["class_id"] == 2)

    print(f"Benign samples   : {benign}")
    print(f"Malignant samples: {malignant}")

    train_s, val_s, test_s = patientwise_split(samples)

    print(f"Train: {len(train_s)}")
    print(f"Val  : {len(val_s)}")
    print(f"Test : {len(test_s)}")

    save_split("train", train_s)
    save_split("val", val_s)
    save_split("test", test_s)

    print(f"\nFinished. Output at: {OUT_ROOT.resolve()}")
    print("Training masks saved in: dataset_seg/<split>/masks")
    print("Colored masks saved in : dataset_seg/<split>/masks_color")


if __name__ == "__main__":
    main()