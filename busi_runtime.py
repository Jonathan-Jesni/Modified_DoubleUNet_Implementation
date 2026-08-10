"""Shared, deterministic runtime utilities for the BUSI v2 study.

This module deliberately contains no model-specific logic.  Dataset labels come
from the immutable manifest; filenames are never interpreted as diagnoses.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


CLASS_MAPPING = {"background": 0, "normal": 0, "benign": 1, "malignant": 2}
LOCKED_BUSI_DATASET = {
    "dataset_fingerprint": "9e1c292e7a2d484b696a41cfda9721931d3e36288c0bbcd6d9ccf3e773f05a0c",
    "manifest_sha256": "3034ff281b6a0259c11e8a07722961cc80ae3b4f066fd732b6380c2d28a23557",
    "dataset_contract_hash": "25925af89068253135eeac18a18218910306d84252170df975e8217791957323",
    "artifact_inventory_hash": "b027732502c0680204ced5c8e034cd2a77982ff83a1f9f20fd1814a059f78700",
    "artifact_file_count": 1556,
    "raw": 780,
    "eligible": 778,
    "excluded": 2,
    "development": 676,
    "sealed": 102,
    "fold_counts": [136, 135, 134, 136, 135],
    "calibration_count": 68,
}
PROFILE_ALIASES = {
    "legacy_imagenet": "legacy_256_imagenet",
    "legacy": "legacy_256_imagenet",
    "xception": "xception_256",
    "padded_256": "padded_256_imagenet",
    "padded_320": "padded_320_imagenet",
}
PREPROCESSING_PROFILES = {
    "legacy_256_imagenet": {
        "input_size": 256,
        "resize": "square_warp",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "image_interpolation": "linear",
        "mask_interpolation": "nearest",
        "padding_value": 0,
    },
    "xception_256": {
        "input_size": 256,
        "resize": "square_warp",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "image_interpolation": "linear",
        "mask_interpolation": "nearest",
        "padding_value": 0,
    },
    "padded_256_imagenet": {
        "input_size": 256,
        "resize": "aspect_pad",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "image_interpolation": "linear",
        "mask_interpolation": "nearest",
        "padding_value": 0,
    },
    "padded_320_imagenet": {
        "input_size": 320,
        "resize": "aspect_pad",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "image_interpolation": "linear",
        "mask_interpolation": "nearest",
        "padding_value": 0,
    },
    "padded_256_xception": {
        "input_size": 256,
        "resize": "aspect_pad",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "image_interpolation": "linear",
        "mask_interpolation": "nearest",
        "padding_value": 0,
    },
    "padded_320_xception": {
        "input_size": 320,
        "resize": "aspect_pad",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "image_interpolation": "linear",
        "mask_interpolation": "nearest",
        "padding_value": 0,
    },
}


def canonical_profile(name: str, input_size: int | None = None) -> dict[str, Any]:
    key = PROFILE_ALIASES.get(name, name)
    if key not in PREPROCESSING_PROFILES:
        raise ValueError(f"Unknown preprocessing profile {name!r}")
    profile = deepcopy(PREPROCESSING_PROFILES[key])
    if input_size is not None:
        profile["input_size"] = int(input_size)
    size = int(profile["input_size"])
    if size <= 0 or size % 16:
        raise ValueError("BUSI input size must be positive and divisible by 16")
    profile["name"] = key
    return profile


def deep_update(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json_dump(value: Any, path: str | os.PathLike[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _coerce_manifest_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"", "none", "null"}:
        return None if lowered != "" else ""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_manifest_rows(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    manifest = Path(path)
    if manifest.suffix.lower() == ".csv":
        with open(manifest, "r", encoding="utf-8-sig", newline="") as handle:
            return [
                {key: _coerce_manifest_value(value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    if manifest.suffix.lower() in {".jsonl", ".ndjson"}:
        with open(manifest, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    payload = load_json(manifest)
    if isinstance(payload, dict):
        payload = payload.get("samples", payload.get("rows"))
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported manifest payload in {manifest}")
    return [dict(row) for row in payload]


def _calibration_folds(row: Mapping[str, Any]) -> set[int]:
    value = row.get("calibration_for_folds", "")
    if isinstance(value, list):
        return {int(item) for item in value}
    return {int(item) for item in str(value).replace(",", ";").split(";") if item.strip()}


def select_rows(
    rows: Sequence[Mapping[str, Any]], split: str, outer_fold: int | None = None
) -> list[dict[str, Any]]:
    split = split.lower()
    eligible = [
        dict(row)
        for row in rows
        if not row.get("quarantine_reason") and not row.get("exclusion_reason")
    ]
    if split == "sealed":
        return [row for row in eligible if row.get("partition") == "sealed"]
    development = [row for row in eligible if row.get("partition") == "development"]
    if split == "development":
        return development
    if outer_fold is None:
        raise ValueError(f"outer_fold is required for split={split!r}")
    if split == "outer":
        return [row for row in development if int(row["fold"]) == int(outer_fold)]
    if split == "calibration":
        return [row for row in development if int(outer_fold) in _calibration_folds(row)]
    if split == "fit":
        return [
            row
            for row in development
            if int(row["fold"]) != int(outer_fold)
            and int(outer_fold) not in _calibration_folds(row)
        ]
    raise ValueError(f"Unknown BUSI split {split!r}")


def resolve_manifest_path(manifest_path: Path, stored_path: str) -> Path:
    path = Path(stored_path)
    if path.is_absolute():
        return path
    candidates = (manifest_path.parent / path, manifest_path.parent.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def make_augmentation(name: str | None):
    if name in {None, "", "none"}:
        return None
    import albumentations as A

    if name == "legacy":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=25,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            ]
        )
    if name == "conservative_ultrasound":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.10,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(0.10, 0.10, p=1.0),
                        A.RandomGamma(gamma_limit=(90, 110), p=1.0),
                    ],
                    p=0.35,
                ),
                A.GaussNoise(var_limit=(1.0, 10.0), p=0.15),
            ]
        )
    if name == "clahe_only":
        return A.Compose([A.HorizontalFlip(p=0.5), A.CLAHE(2.0, (8, 8), p=0.3)])
    raise ValueError(f"Unknown augmentation profile {name!r}")


def preprocess_pair(
    image: np.ndarray, mask: np.ndarray, profile: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    size = int(profile["input_size"])
    original_h, original_w = mask.shape[:2]
    if profile["resize"] == "square_warp":
        processed_image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        processed_mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        geometry = {"top": 0, "left": 0, "height": size, "width": size}
    elif profile["resize"] == "aspect_pad":
        scale = min(size / original_h, size / original_w)
        height = max(1, min(size, int(round(original_h * scale))))
        width = max(1, min(size, int(round(original_w * scale))))
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        top, left = (size - height) // 2, (size - width) // 2
        processed_image = np.zeros((size, size, 3), dtype=image.dtype)
        processed_mask = np.zeros((size, size), dtype=mask.dtype)
        processed_image[top : top + height, left : left + width] = resized_image
        processed_mask[top : top + height, left : left + width] = resized_mask
        geometry = {"top": top, "left": left, "height": height, "width": width}
    else:
        raise ValueError(f"Unknown resize policy {profile['resize']!r}")
    values = set(np.unique(processed_mask).tolist())
    if not values.issubset({0, 1, 2}):
        raise ValueError(f"Mask interpolation produced invalid class IDs: {sorted(values)}")
    image_float = processed_image.astype(np.float32) / 255.0
    mean = np.asarray(profile["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(profile["std"], dtype=np.float32).reshape(1, 1, 3)
    image_float = (image_float - mean) / std
    return (
        np.ascontiguousarray(image_float.transpose(2, 0, 1)),
        np.ascontiguousarray(processed_mask.astype(np.int64)),
        geometry,
    )


class ManifestSegmentationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        split: str,
        outer_fold: int | None,
        preprocessing_profile: str,
        input_size: int | None = None,
        augmentation: str | None = None,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.rows = select_rows(load_manifest_rows(self.manifest_path), split, outer_fold)
        self.profile = canonical_profile(preprocessing_profile, input_size)
        self.augmentation = make_augmentation(augmentation)
        if not self.rows:
            raise ValueError(f"No manifest rows selected for {split=} {outer_fold=}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = resolve_manifest_path(self.manifest_path, str(row["image_path"]))
        mask_path = resolve_manifest_path(self.manifest_path, str(row["mask_path"]))
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise FileNotFoundError(f"Cannot decode BUSI pair {image_path}, {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        image, mask, geometry = preprocess_pair(image, mask, self.profile)
        expected_class = int(row["class_id"])
        if expected_class == 0 and np.any(mask):
            raise ValueError(f"Normal sample {row['sample_id']} has a non-empty mask")
        if expected_class in {1, 2} and any(v not in {0, expected_class} for v in np.unique(mask)):
            raise ValueError(f"Mask/manifest class mismatch for {row['sample_id']}")
        metadata = {
            "sample_id": str(row["sample_id"]),
            "class_id": expected_class,
            "diagnosis": str(row["diagnosis"]),
            "group_id": str(row.get("group_id", "")),
            "original_height": int(row.get("original_height", image.shape[1])),
            "original_width": int(row.get("original_width", image.shape[2])),
            "lesion_size_quartile": (
                "normal"
                if expected_class == 0
                else f"Q{int(row.get('lesion_size_quartile', 0))}"
            ),
            **geometry,
        }
        return torch.from_numpy(image), torch.from_numpy(mask), metadata


def class_pixel_counts(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    counts = [0, 0, 0]
    for row in rows:
        height = int(row["original_height"])
        width = int(row["original_width"])
        lesion = int(row.get("lesion_pixels", 0))
        class_id = int(row["class_id"])
        counts[0] += height * width - lesion
        if class_id:
            counts[class_id] += lesion
    if any(value <= 0 for value in counts):
        raise ValueError(f"Every fit-pool class must contain pixels; got {counts}")
    return counts


def preprocessed_class_pixel_counts(
    dataset: ManifestSegmentationDataset,
) -> list[int]:
    """Count fit pixels after the recipe's deterministic resize/padding policy."""
    counts = np.zeros(3, dtype=np.int64)
    for row in dataset.rows:
        mask_path = resolve_manifest_path(
            dataset.manifest_path, str(row["mask_path"])
        )
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        dummy = np.zeros((*mask.shape, 3), dtype=np.uint8)
        _, processed_mask, _ = preprocess_pair(dummy, mask, dataset.profile)
        counts += np.bincount(processed_mask.reshape(-1), minlength=3)[:3]
    resolved = [int(value) for value in counts]
    if any(value <= 0 for value in resolved):
        raise ValueError(
            f"Every preprocessed fit-pool class must contain pixels; got {resolved}"
        )
    return resolved


def diagnosis_balanced_sampler(
    rows: Sequence[Mapping[str, Any]],
    target_probabilities: Sequence[float] = (0.25, 0.50, 0.25),
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    targets = np.asarray(target_probabilities, dtype=np.float64)
    if targets.shape != (3,) or np.any(targets < 0) or not np.isclose(targets.sum(), 1.0):
        raise ValueError("target_probabilities must be three non-negative values summing to one")
    labels = np.asarray([int(row["class_id"]) for row in rows])
    frequencies = np.bincount(labels, minlength=3)
    if np.any(frequencies == 0):
        raise ValueError("Balanced sampling requires all three diagnoses in the fit pool")
    weights = [targets[label] / frequencies[label] for label in labels]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(rows),
        replacement=True,
        generator=generator,
    )


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_fingerprint(path: str | os.PathLike[str]) -> str:
    manifest = Path(path)
    sidecars = [
        manifest.parent / "dataset_fingerprint.json",
        manifest.parent / "dataset_fingerprint.sha256",
    ]
    hashes = {"manifest": sha256_file(manifest)}
    for sidecar in sidecars:
        if sidecar.exists():
            hashes[sidecar.name] = sha256_file(sidecar)
    return canonical_sha256(hashes)


def verify_generated_artifacts(
    manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Hash every eligible generated image/mask and bind the exact byte inventory."""

    manifest = Path(manifest_path).resolve()
    rows = load_manifest_rows(manifest)
    inventory = []
    for row in sorted(rows, key=lambda item: str(item.get("sample_id", ""))):
        if row.get("quarantine_reason") or row.get("exclusion_reason"):
            continue
        if "eligible" in row and not bool(row["eligible"]):
            continue
        sample_id = str(row.get("sample_id", ""))
        expected_image = str(row.get("generated_image_sha256", ""))
        expected_mask = str(row.get("generated_mask_sha256", ""))
        if not sample_id or not expected_image or not expected_mask:
            raise RuntimeError("BUSI artifact inventory is missing sample IDs or hashes")
        image_path = resolve_manifest_path(manifest, str(row["image_path"]))
        mask_path = resolve_manifest_path(manifest, str(row["mask_path"]))
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing generated BUSI artifact for {sample_id}: "
                f"{image_path}, {mask_path}"
            )
        if sha256_file(image_path) != expected_image:
            raise RuntimeError(f"Generated BUSI image hash mismatch: {sample_id}")
        if sha256_file(mask_path) != expected_mask:
            raise RuntimeError(f"Generated BUSI mask hash mismatch: {sample_id}")
        inventory.append(
            {
                "sample_id": sample_id,
                "image_sha256": expected_image,
                "mask_sha256": expected_mask,
            }
        )
    if not inventory:
        raise RuntimeError("BUSI generated artifact inventory is empty")
    return {
        "artifact_inventory_hash": canonical_sha256(inventory),
        "artifact_file_count": 2 * len(inventory),
        "sample_count": len(inventory),
    }


def split_membership_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    membership = [
        {
            "sample_id": str(row["sample_id"]),
            "content_group": str(row.get("content_group", row.get("group_id", ""))),
            "class_id": int(row["class_id"]),
        }
        for row in sorted(rows, key=lambda item: str(item["sample_id"]))
    ]
    if not membership:
        raise ValueError("Cannot hash an empty BUSI split")
    return canonical_sha256(membership)


def validate_dataset_contract(
    manifest_path: str | os.PathLike[str],
    require_completed_review: bool = True,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Fail closed on a changed, incomplete, or unexpectedly sized BUSI dataset."""
    manifest = Path(manifest_path).resolve()
    metadata_path = manifest.parent / "dataset_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing BUSI dataset metadata: {metadata_path}")
    metadata = load_json(metadata_path)
    actual_manifest_hash = sha256_file(manifest)
    if metadata.get("manifest_sha256") != actual_manifest_hash:
        raise RuntimeError("BUSI manifest hash differs from dataset metadata")
    rows = load_manifest_rows(manifest)
    fingerprints = {str(row.get("dataset_fingerprint", "")) for row in rows}
    if fingerprints != {str(metadata.get("dataset_fingerprint", ""))}:
        raise RuntimeError("BUSI manifest rows do not share the frozen dataset fingerprint")
    review = metadata.get("duplicate_review", {})
    if require_completed_review:
        if not review.get("complete", False):
            pending = review.get("state_counts", {})
            raise RuntimeError(
                "BUSI duplicate review is incomplete; full study training is locked. "
                f"Review states: {pending}"
            )
        review_value = review.get("review_csv")
        if not review_value:
            raise RuntimeError("BUSI completed duplicate-review record is missing")
        # Dataset metadata may have been generated on Windows.  Normalizing the
        # separator keeps the frozen dataset portable to Linux/ROCm hosts.
        review_path = (
            manifest.parent / Path(str(review_value).replace("\\", "/"))
        ).resolve()
        try:
            review_path.relative_to(manifest.parent)
        except ValueError as exc:
            raise RuntimeError(
                "BUSI duplicate-review record must be embedded in the dataset"
            ) from exc
        if not review_path.is_file():
            raise FileNotFoundError(f"Missing completed review record: {review_path}")
        if sha256_file(review_path) != review.get("review_csv_sha256"):
            raise RuntimeError("BUSI completed duplicate-review hash mismatch")
        reviewer_paths = review.get("reviewer_decision_csvs", {})
        reviewer_hashes = review.get("reviewer_decision_sha256s", {})
        if set(reviewer_paths) != {"1", "2"} or set(reviewer_hashes) != {"1", "2"}:
            raise RuntimeError("BUSI requires two embedded reviewer decision records")
        for reviewer in ("1", "2"):
            reviewer_path = (
                manifest.parent
                / Path(str(reviewer_paths[reviewer]).replace("\\", "/"))
            ).resolve()
            try:
                reviewer_path.relative_to(manifest.parent)
            except ValueError as exc:
                raise RuntimeError("Reviewer record escaped the BUSI dataset root") from exc
            if not reviewer_path.is_file():
                raise FileNotFoundError(reviewer_path)
            if sha256_file(reviewer_path) != reviewer_hashes[reviewer]:
                raise RuntimeError(f"BUSI reviewer {reviewer} decision hash mismatch")
    if expected_counts:
        actual = metadata.get("counts", {})
        for name, expected in expected_counts.items():
            observed = (
                actual.get(name)
                if name in actual
                else actual.get("by_diagnosis", {}).get(name)
            )
            if int(observed if observed is not None else -1) != int(expected):
                raise RuntimeError(
                    f"BUSI {name} count is {observed!r}, expected {expected}"
                )
    if int(metadata.get("split_seed", -1)) != 20260717:
        raise RuntimeError("BUSI split seed is not the prespecified 20260717")
    if int(metadata.get("folds", -1)) != 5:
        raise RuntimeError("BUSI manifest must contain five development folds")
    if not metadata.get("include_normal"):
        raise RuntimeError("BUSI normals must be included")
    locked = LOCKED_BUSI_DATASET
    if str(metadata.get("dataset_fingerprint")) != locked["dataset_fingerprint"]:
        raise RuntimeError("BUSI dataset fingerprint differs from the locked final study")
    if actual_manifest_hash != locked["manifest_sha256"]:
        raise RuntimeError("BUSI manifest differs from the locked final study")
    if manifest_fingerprint(manifest) != locked["dataset_contract_hash"]:
        raise RuntimeError("BUSI dataset contract differs from the locked final study")
    if not np.isclose(float(metadata.get("sealed_fraction_requested", -1)), 0.15):
        raise RuntimeError("BUSI sealed fraction is not the locked 0.15")
    if not np.isclose(
        float(metadata.get("calibration_fraction_requested", -1)), 0.125
    ):
        raise RuntimeError("BUSI calibration fraction is not the locked 0.125")
    counts = metadata.get("counts", {})
    for name in ("raw", "eligible", "excluded", "development", "sealed"):
        if int(counts.get(name, -1)) != int(locked[name]):
            raise RuntimeError(f"BUSI {name} count differs from the locked final study")
    observed_folds = [
        int(counts.get("by_fold", {}).get(str(fold), -1)) for fold in range(5)
    ]
    if observed_folds != locked["fold_counts"]:
        raise RuntimeError("BUSI fold counts differ from the locked final study")
    for fold in range(5):
        if len(select_rows(rows, "calibration", fold)) != locked["calibration_count"]:
            raise RuntimeError(
                f"BUSI calibration count differs for outer fold {fold}"
            )
    prior_test = [
        row
        for row in rows
        if not row.get("exclusion_reason")
        and "test" in str(row.get("previous_split", "")).split(";")
    ]
    if len(prior_test) != 65 or any(
        row.get("partition") != "development" for row in prior_test
    ):
        raise RuntimeError("BUSI prior inspected test membership is not locked to development")
    review_states = review.get("state_counts", {})
    if int(review.get("candidate_count", -1)) != 239 or review_states != {
        "accepted_by_both_reviewers": 238,
        "rejected_by_both_reviewers": 1,
    }:
        raise RuntimeError("BUSI duplicate-review inventory differs from the locked study")
    return metadata


def git_state(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = str(Path(repo_root).resolve())

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    status = run("status", "--porcelain=v1", "--untracked-files=no")
    diff = run("diff", "--binary", "--", "*.py", "*.json", "requirements.txt")
    root_path = Path(root)
    code_files = list(root_path.glob("*.py"))
    code_files.extend((root_path / "tests").glob("**/*.py"))
    code_files.extend((root_path / "configs").glob("**/*.json"))
    code_files.extend(root_path.glob("requirements*.txt"))
    code_digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in code_files if item.is_file()}):
        code_digest.update(path.relative_to(root_path).as_posix().encode("utf-8"))
        code_digest.update(sha256_file(path).encode("ascii"))
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status and status != "unavailable"),
        "dirty_fingerprint": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "workspace_code_fingerprint": code_digest.hexdigest(),
    }


def _gpu_driver_version() -> str:
    for name in (
        "GPU_DRIVER_VERSION",
        "ROCM_DRIVER_VERSION",
        "AMD_DRIVER_VERSION",
        "NVIDIA_DRIVER_VERSION",
    ):
        value = os.environ.get(name)
        if value:
            return f"{name}={value.strip()}"
    amdgpu_version = Path("/sys/module/amdgpu/version")
    if amdgpu_version.is_file():
        try:
            return f"amdgpu={amdgpu_version.read_text(encoding='utf-8').strip()}"
        except OSError:
            pass
    commands = (
        ["rocm-smi", "--showdriverversion", "--json"],
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    )
    for command in commands:
        try:
            output = subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        if output:
            return " ".join(output.split())
    return "unavailable"


def environment_fingerprint() -> dict[str, Any]:
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = "unavailable"
    try:
        import timm
        timm_version = timm.__version__
    except Exception:
        timm_version = "unavailable"
    try:
        import albumentations

        albumentations_version = albumentations.__version__
    except Exception:
        albumentations_version = "unavailable"
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = "unavailable"
    try:
        import scipy

        scipy_version = scipy.__version__
    except Exception:
        scipy_version = "unavailable"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "timm": timm_version,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "albumentations": albumentations_version,
        "scikit_learn": sklearn_version,
        "scipy": scipy_version,
        "gpu_driver": _gpu_driver_version(),
        "cuda_or_rocm_runtime": getattr(torch.version, "cuda", None)
        or getattr(torch.version, "hip", None),
        "hip": getattr(torch.version, "hip", None),
        "gpu": gpu,
        "container_digest": os.environ.get("CONTAINER_IMAGE_DIGEST", "unavailable"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
    }
    payload["digest"] = canonical_sha256(payload)
    return payload


def capture_rng_state(
    dataloader_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_gpu": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if dataloader_generator is not None:
        state["dataloader_generator"] = dataloader_generator.get_state()
    return state


def restore_rng_state(
    state: Mapping[str, Any], dataloader_generator: torch.Generator | None = None
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_gpu"):
        torch.cuda.set_rng_state_all(state["torch_gpu"])
    if dataloader_generator is not None and "dataloader_generator" in state:
        dataloader_generator.set_state(state["dataloader_generator"])


def atomic_torch_save(payload: Any, path: str | os.PathLike[str]) -> None:
    """Write, reload-verify, and atomically retain the current and previous file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    verified = torch.load(temporary, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and isinstance(verified, Mapping):
        required = {"schema_version", "model_state_dict"}
        if required.issubset(payload) and not required.issubset(verified):
            raise RuntimeError(f"Checkpoint verification failed for {temporary}")
    if target.exists():
        previous = target.with_name(f"{target.stem}.prev{target.suffix}")
        os.replace(target, previous)
    os.replace(temporary, target)


def append_jsonl(path: str | os.PathLike[str], row: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    else:
        torch.use_deterministic_algorithms(False)


def worker_seed_init(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
