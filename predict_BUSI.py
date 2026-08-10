"""Generate BUSI masks, probability maps, overlays, and panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from busi_evaluation import apply_foreground_threshold
from busi_runtime import (
    ManifestSegmentationDataset,
    atomic_json_dump,
    load_json,
    sha256_file,
    split_membership_hash,
    validate_dataset_contract,
)


COLORS = {0: (0, 0, 0), 1: (0, 200, 0), 2: (0, 0, 230)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/BUSI")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _colorize(mask: np.ndarray) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in COLORS.items():
        output[mask == class_id] = color
    return output


def _display_image(tensor, profile) -> np.ndarray:
    image = tensor.numpy().transpose(1, 2, 0)
    mean = np.asarray(profile["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(profile["std"], dtype=np.float32).reshape(1, 1, 3)
    image = np.clip((image * std + mean) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    output = image.copy()
    foreground = mask != 0
    colors = _colorize(mask)
    output[foreground] = (
        (1.0 - alpha) * image[foreground].astype(np.float32)
        + alpha * colors[foreground].astype(np.float32)
    ).astype(np.uint8)
    return output


def _title(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (25, 25, 25), -1)
    cv2.putText(
        output,
        text,
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def _next_output_dir(requested: str | None) -> Path:
    if requested:
        path = Path(requested).resolve()
        if path.exists():
            raise FileExistsError(f"Output directory already exists: {path}")
        return path
    path = Path("files/predictions_BUSI").resolve()
    suffix = 2
    while path.exists():
        path = Path(f"files/predictions_BUSI_{suffix}").resolve()
        suffix += 1
    return path


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "resolved_config.json"
    results_path = run_dir / "test_results.json"
    if not config_path.is_file():
        raise FileNotFoundError("Run python train_BUSI.py first")
    if not results_path.is_file():
        raise FileNotFoundError("Run python test_BUSI.py first")
    config = load_json(config_path)
    results = load_json(results_path)
    probability_value = Path(results["probabilities"])
    probability_path = (
        probability_value.resolve()
        if probability_value.is_absolute()
        else (run_dir / probability_value).resolve()
    )
    if not probability_path.is_file():
        raise FileNotFoundError(probability_path)
    if sha256_file(probability_path) != results.get("probabilities_sha256"):
        raise RuntimeError("Saved BUSI probability archive hash does not match the report")

    dataset_metadata = validate_dataset_contract(
        config["dataset"]["manifest"],
        require_completed_review=True,
        expected_counts=config["dataset"].get("expected_counts"),
    )
    if dataset_metadata["dataset_fingerprint"] != results.get("dataset_fingerprint"):
        raise RuntimeError("Prediction report belongs to a different BUSI dataset")

    dataset = ManifestSegmentationDataset(
        config["dataset"]["manifest"],
        split="outer",
        outer_fold=int(config["outer_fold"]),
        preprocessing_profile=config["dataset"]["preprocessing_profile"],
        input_size=config["dataset"]["input_size"],
        augmentation=None,
    )
    with np.load(probability_path, allow_pickle=False) as archive:
        probabilities = archive["p2"].copy()
        sample_ids = archive["sample_ids"].astype(str)
        if str(archive["dataset_fingerprint"].item()) != results["dataset_fingerprint"]:
            raise RuntimeError("Probability archive dataset fingerprint mismatch")
        if str(archive["outer_split_hash"].item()) != results["outer_split_hash"]:
            raise RuntimeError("Probability archive split fingerprint mismatch")
    expected_ids = np.asarray([str(row["sample_id"]) for row in dataset.rows])
    if not np.array_equal(sample_ids, expected_ids):
        raise RuntimeError("Saved probabilities do not match the current test dataset")
    if split_membership_hash(dataset.rows) != results["outer_split_hash"]:
        raise RuntimeError("Current BUSI test split differs from the evaluated split")

    threshold = float(results["threshold"])
    predictions = apply_foreground_threshold(probabilities, threshold)
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    if count <= 0:
        raise ValueError("--limit must be positive")
    output_dir = _next_output_dir(args.output_dir)
    for name in ("masks", "probabilities", "overlays", "panels"):
        (output_dir / name).mkdir(parents=True, exist_ok=False)

    rows = []
    profile = config["dataset"]["preprocessing"]
    for index in range(count):
        image_tensor, target_tensor, metadata = dataset[index]
        sample_id = str(metadata["sample_id"])
        safe_name = sample_id.replace("/", "_").replace("\\", "_")
        image = _display_image(image_tensor, profile)
        target = target_tensor.numpy().astype(np.uint8)
        prediction = predictions[index].astype(np.uint8)
        foreground_probability = probabilities[index, 1:].sum(axis=0)
        probability_u8 = np.clip(foreground_probability * 255.0, 0, 255).astype(np.uint8)
        probability_color = cv2.applyColorMap(probability_u8, cv2.COLORMAP_VIRIDIS)
        overlay = _overlay(image, prediction)
        separator = np.full((image.shape[0], 6, 3), 255, dtype=np.uint8)
        panel = np.concatenate(
            [
                _title(image, sample_id),
                separator,
                _title(_colorize(target), "Ground truth"),
                separator,
                _title(probability_color, f"p_fg; t={threshold:.2f}"),
                separator,
                _title(_colorize(prediction), "Prediction"),
                separator,
                _title(overlay, "Overlay"),
            ],
            axis=1,
        )
        artifacts = {
            output_dir / "masks" / f"{safe_name}.png": prediction,
            output_dir / "probabilities" / f"{safe_name}.png": probability_u8,
            output_dir / "overlays" / f"{safe_name}.png": overlay,
            output_dir / "panels" / f"{safe_name}.png": panel,
        }
        for path, artifact in artifacts.items():
            if not cv2.imwrite(str(path), artifact):
                raise OSError(f"Failed to write {path}")
        rows.append(
            {
                "sample_id": sample_id,
                "predicted_foreground_fraction": float((prediction != 0).mean()),
                "ground_truth_foreground_fraction": float((target != 0).mean()),
            }
        )

    atomic_json_dump(
        {
            "checkpoint": results["checkpoint"],
            "threshold": threshold,
            "sample_count": count,
            "samples": rows,
        },
        output_dir / "prediction_manifest.json",
    )
    print(f"Saved {count} predictions to {output_dir}")


if __name__ == "__main__":
    main()
