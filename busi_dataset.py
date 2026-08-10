"""Canonical BUSI v2 generation and manifest-backed dataset access."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    class Dataset:  # type: ignore[no-redef]
        pass


MANIFEST_VERSION = "busi-v2.0"
CLASS_IDS = {"normal": 0, "benign": 1, "malignant": 2}
KNOWN_CONTRADICTORY_FILES = {
    ("benign", "benign (433).png"),
    ("malignant", "malignant (145).png"),
}
MANIFEST_FIELDS = [
    "sample_id", "diagnosis", "class_id", "eligible", "exclusion_reason",
    "source_image_path", "source_mask_paths", "image_path", "mask_path",
    "original_height", "original_width", "aspect_ratio", "lesion_pixels",
    "lesion_fraction", "connected_components", "bounding_box_xywh",
    "lesion_size_quartile", "aspect_ratio_bin", "mask_count", "multi_mask",
    "image_read_ok", "mask_qc_ok", "qc_notes", "source_image_sha256",
    "decoded_pixel_sha256", "source_mask_sha256s", "union_mask_sha256",
    "generated_image_sha256", "generated_mask_sha256", "exact_duplicate_group",
    "near_duplicate_group", "content_group", "grouping_evidence",
    "previous_split", "previous_names", "partition", "fold",
    "calibration_for_folds", "split_seed", "manifest_version",
    "dataset_fingerprint",
]


@dataclass(frozen=True)
class GenerationConfig:
    raw_root: Path
    output_root: Path
    prior_dataset_root: Path | None = None
    prior_manifest: Path | None = None
    include_normal: bool = True
    split_seed: int = 20260717
    sealed_fraction: float = 0.15
    folds: int = 5
    calibration_fraction: float = 0.125
    expected_prior_test_count: int | None = 65
    review_csv: Path | None = None

    def validate(self) -> None:
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"BUSI raw root does not exist: {self.raw_root}")
        if self.output_root.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {self.output_root}. "
                "Choose a new versioned --output-root."
            )
        if self.expected_prior_test_count is not None:
            if self.expected_prior_test_count < 0:
                raise ValueError("expected_prior_test_count must be non-negative")
            has_prior_dataset = (
                self.prior_dataset_root is not None
                and self.prior_dataset_root.is_dir()
            )
            has_prior_manifest = (
                self.prior_manifest is not None and self.prior_manifest.is_file()
            )
            if not has_prior_dataset and not has_prior_manifest:
                raise FileNotFoundError(
                    "Locked BUSI regeneration requires either the prior prepared "
                    "dataset root or a canonical prior manifest"
                )
        if not 0.0 < self.sealed_fraction < 0.5:
            raise ValueError("sealed_fraction must be between 0 and 0.5")
        if not 0.0 < self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must be between 0 and 0.5")
        if self.folds < 2:
            raise ValueError("folds must be at least 2")
        if self.review_csv is not None and not self.review_csv.is_file():
            raise FileNotFoundError(f"Review CSV does not exist: {self.review_csv}")


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1

    def groups(self) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for value in self.parent:
            grouped[self.find(value)].append(value)
        return [sorted(values) for values in grouped.values()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_id(diagnosis: str, image_path: Path) -> str:
    match = re.search(r"\((\d+)\)", image_path.stem)
    if match:
        return f"{diagnosis}_{int(match.group(1)):04d}"
    slug = re.sub(r"[^a-z0-9]+", "_", image_path.stem.lower()).strip("_")
    return f"{diagnosis}_{slug}"


def _collect_masks(class_dir: Path, image_path: Path) -> list[Path]:
    return sorted(class_dir.glob(f"{image_path.stem}_mask*.png"))


def _merge_masks(
    mask_paths: Sequence[Path], shape: tuple[int, int]
) -> tuple[np.ndarray, list[str]]:
    merged = np.zeros(shape, dtype=np.uint8)
    notes: list[str] = []
    for path in mask_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            notes.append(f"unreadable_mask:{path.name}")
        elif mask.shape != shape:
            notes.append(f"mask_shape_mismatch:{path.name}")
        else:
            merged = np.maximum(merged, (mask > 0).astype(np.uint8))
    return merged, notes


def _prior_membership(
    prior_root: Path | None,
    prior_manifest: Path | None = None,
) -> dict[str, list[tuple[str, str]]]:
    result: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if prior_manifest is not None and prior_manifest.is_file():
        with prior_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "decoded_pixel_sha256",
                "previous_split",
                "previous_names",
                "dataset_fingerprint",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise RuntimeError(
                    f"Prior BUSI manifest is missing required columns: {sorted(missing)}"
                )
            rows = list(reader)
        fingerprints = {
            str(row["dataset_fingerprint"]).strip()
            for row in rows
            if str(row["dataset_fingerprint"]).strip()
        }
        if len(fingerprints) != 1:
            raise RuntimeError("Prior BUSI manifest must have one shared dataset fingerprint")
        metadata_path = prior_manifest.parent / "dataset_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Prior BUSI manifest metadata is missing: {metadata_path}"
            )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("manifest_sha256") != sha256_file(prior_manifest):
            raise RuntimeError("Prior BUSI manifest hash differs from its metadata")
        if metadata.get("dataset_fingerprint") not in fingerprints:
            raise RuntimeError("Prior BUSI manifest fingerprint differs from its metadata")
        for row in rows:
            pixel_hash = str(row.get("decoded_pixel_sha256", "")).strip()
            if not pixel_hash:
                continue
            splits = {
                value.strip()
                for value in str(row.get("previous_split", "")).split(";")
                if value.strip()
            }
            names = {
                value.strip()
                for value in str(row.get("previous_names", "")).split(";")
                if value.strip()
            }
            if splits or names:
                result[pixel_hash].extend((split, "") for split in sorted(splits))
                result[pixel_hash].extend(("", name) for name in sorted(names))
        return result
    if prior_root is None or not prior_root.is_dir():
        return result
    for split in ("train", "val", "test"):
        directory = prior_root / split / "images"
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.png")):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                result[sha256_array(image)].append((split, path.name))
    return result


def _scan_raw(config: GenerationConfig) -> list[dict[str, Any]]:
    prior = _prior_membership(config.prior_dataset_root, config.prior_manifest)
    diagnoses = ["benign", "malignant"]
    if config.include_normal:
        diagnoses.insert(0, "normal")
    records: list[dict[str, Any]] = []
    for diagnosis in diagnoses:
        directory = config.raw_root / diagnosis
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing BUSI class directory: {directory}")
        images = sorted(
            path for path in directory.glob("*.png") if "_mask" not in path.stem.lower()
        )
        for image_path in images:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            mask_paths = _collect_masks(directory, image_path)
            record: dict[str, Any] = {
                "sample_id": _sample_id(diagnosis, image_path),
                "diagnosis": diagnosis,
                "class_id": CLASS_IDS[diagnosis],
                "source_image": image_path,
                "source_masks": mask_paths,
                "image": image,
                "eligible": True,
                "exclusion_reason": "",
                "qc_notes": [],
            }
            if image is None:
                record.update(
                    eligible=False,
                    exclusion_reason="unreadable_source_image",
                    binary_mask=np.zeros((1, 1), dtype=np.uint8),
                    decoded_pixel_sha256="",
                    previous_split="",
                    previous_names="",
                )
                records.append(record)
                continue
            pixel_hash = sha256_array(image)
            previous = prior.get(pixel_hash, [])
            record["decoded_pixel_sha256"] = pixel_hash
            record["previous_split"] = ";".join(
                sorted({item[0] for item in previous if item[0]})
            )
            record["previous_names"] = ";".join(
                sorted({item[1] for item in previous if item[1]})
            )
            if diagnosis == "normal":
                if mask_paths:
                    source_mask, notes = _merge_masks(mask_paths, image.shape)
                    record["qc_notes"].extend(notes)
                    if notes:
                        record["eligible"] = False
                        record["exclusion_reason"] = "normal_source_mask_qc_failure"
                    elif np.any(source_mask):
                        record["eligible"] = False
                        record["exclusion_reason"] = "normal_source_mask_not_empty"
                    else:
                        record["qc_notes"].append("source_normal_masks_ignored")
                else:
                    record["qc_notes"].append("normal_source_mask_absent")
                record["binary_mask"] = np.zeros(image.shape, dtype=np.uint8)
            else:
                if not mask_paths:
                    record["eligible"] = False
                    record["exclusion_reason"] = "missing_positive_mask"
                mask, notes = _merge_masks(mask_paths, image.shape)
                record["binary_mask"] = mask
                record["qc_notes"].extend(notes)
                if notes:
                    record["eligible"] = False
                    record["exclusion_reason"] = (
                        record["exclusion_reason"] or "positive_source_mask_qc_failure"
                    )
                if not np.any(mask):
                    record["eligible"] = False
                    record["exclusion_reason"] = (
                        record["exclusion_reason"] or "empty_positive_mask"
                    )
            records.append(record)
    ids = [record["sample_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Stable sample IDs are not unique")
    return records


def _assign_exact_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["decoded_pixel_sha256"]:
            by_hash[record["decoded_pixel_sha256"]].append(record)
    rows: list[dict[str, Any]] = []
    index = 0
    for pixel_hash, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue
        group_id = f"exact_{index:04d}"
        index += 1
        diagnoses = sorted({member["diagnosis"] for member in members})
        contradictory = len(diagnoses) > 1
        for member in members:
            member["exact_duplicate_group"] = group_id
            if contradictory:
                member["eligible"] = False
                member["exclusion_reason"] = "contradictory_exact_duplicate"
        rows.append({
            "exact_group": group_id,
            "decoded_pixel_sha256": pixel_hash,
            "sample_ids": ";".join(sorted(member["sample_id"] for member in members)),
            "diagnoses": ";".join(diagnoses),
            "contradictory": int(contradictory),
            "action": "quarantine" if contradictory else "group",
        })
    for record in records:
        record.setdefault("exact_duplicate_group", "")
        if (record["diagnosis"], record["source_image"].name) in KNOWN_CONTRADICTORY_FILES:
            record["eligible"] = False
            record["exclusion_reason"] = "known_contradictory_exact_duplicate"
    return rows


def _phash(image: np.ndarray) -> int:
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    values = cv2.dct(resized)[:8, :8].flatten()
    threshold = float(np.median(values[1:]))
    result = 0
    for bit in values > threshold:
        result = (result << 1) | int(bit)
    return result


def _thumbnail(image: np.ndarray, size: int = 64) -> np.ndarray:
    result = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    result -= float(result.mean())
    norm = float(np.linalg.norm(result))
    if norm:
        result /= norm
    return result


def _overlap(
    left: np.ndarray, right: np.ndarray, dy: int, dx: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = left.shape
    ly0, ly1 = max(0, dy), min(height, height + dy)
    lx0, lx1 = max(0, dx), min(width, width + dx)
    ry0, ry1 = max(0, -dy), min(height, height - dy)
    rx0, rx1 = max(0, -dx), min(width, width - dx)
    return left[ly0:ly1, lx0:lx1], right[ry0:ry1, rx0:rx1]


def _aligned_correlation(
    left: np.ndarray, right: np.ndarray, max_shift: int = 2
) -> tuple[float, int, int]:
    best = (-1.0, 0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            a, b = _overlap(left, right, dy, dx)
            a = a - float(a.mean())
            b = b - float(b.mean())
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            score = float(np.sum(a * b) / denominator) if denominator else 0.0
            if score > best[0]:
                best = (score, dy, dx)
    return best


def _all_aligned_correlations(
    thumbnails: np.ndarray, max_shift: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact best translation-aligned correlation for every image pair."""
    count, height, width = thumbnails.shape
    best = np.full((count, count), -np.inf, dtype=np.float32)
    best_dy = np.zeros((count, count), dtype=np.int8)
    best_dx = np.zeros((count, count), dtype=np.int8)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            ly0, ly1 = max(0, dy), min(height, height + dy)
            lx0, lx1 = max(0, dx), min(width, width + dx)
            ry0, ry1 = max(0, -dy), min(height, height - dy)
            rx0, rx1 = max(0, -dx), min(width, width - dx)
            left = thumbnails[:, ly0:ly1, lx0:lx1].reshape(count, -1).copy()
            right = thumbnails[:, ry0:ry1, rx0:rx1].reshape(count, -1).copy()
            left -= left.mean(axis=1, keepdims=True)
            right -= right.mean(axis=1, keepdims=True)
            left_norm = np.linalg.norm(left, axis=1, keepdims=True)
            right_norm = np.linalg.norm(right, axis=1, keepdims=True)
            np.divide(left, left_norm, out=left, where=left_norm != 0)
            np.divide(right, right_norm, out=right, where=right_norm != 0)
            correlation = left @ right.T
            update = correlation > best
            best[update] = correlation[update]
            best_dy[update] = dy
            best_dx[update] = dx
    return best, best_dy, best_dx


def _mask_dice(left: np.ndarray, right: np.ndarray, dy: int, dx: int) -> float:
    a, b = _overlap(left, right, dy, dx)
    a, b = a > 0, b > 0
    denominator = int(a.sum()) + int(b.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(a, b).sum() / denominator)


def _review_decisions(path: Path | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}
    result: dict[str, tuple[str, str]] = {}
    allowed = {"", "group", "accept", "same", "separate", "reject", "different"}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            candidate_id = row.get("candidate_id", "").strip()
            if candidate_id:
                if candidate_id in result:
                    raise ValueError(f"duplicate review row for {candidate_id}")
                decisions = (
                    row.get("reviewer_1_decision", "").strip().lower(),
                    row.get("reviewer_2_decision", "").strip().lower(),
                )
                if any(decision not in allowed for decision in decisions):
                    raise ValueError(
                        f"invalid duplicate-review decision for {candidate_id}: {decisions}"
                    )
                result[candidate_id] = decisions
    return result


def _review_outcome(left: str, right: str) -> tuple[bool, str]:
    group_values = {"group", "accept", "same"}
    separate_values = {"separate", "reject", "different"}
    if left in separate_values and right in separate_values:
        return False, "rejected_by_both_reviewers"
    if left in group_values and right in group_values:
        return True, "accepted_by_both_reviewers"
    if left or right:
        return True, "disputed_conservative_group"
    return True, "pending_conservative_group"


def _near_candidates(
    records: list[dict[str, Any]], review_csv: Path | None
) -> list[dict[str, Any]]:
    readable = [record for record in records if record["image"] is not None]
    if not readable:
        return []
    thumbs = np.stack([_thumbnail(record["image"]) for record in readable])
    vectors = thumbs.reshape(len(readable), -1)
    correlations = np.clip(vectors @ vectors.T, -1.0, 1.0)
    aligned_correlations, alignment_dys, alignment_dxs = (
        _all_aligned_correlations(thumbs)
    )
    phashes = [_phash(record["image"]) for record in readable]
    masks = np.stack([
        cv2.resize(record["binary_mask"], (64, 64), interpolation=cv2.INTER_NEAREST)
        for record in readable
    ])
    reviews = _review_decisions(review_csv)
    candidates: list[dict[str, Any]] = []
    for i in range(len(readable)):
        for j in range(i + 1, len(readable)):
            left, right = readable[i], readable[j]
            if left["decoded_pixel_sha256"] == right["decoded_pixel_sha256"]:
                continue
            distance = (phashes[i] ^ phashes[j]).bit_count()
            zero_corr = float(correlations[i, j])
            aligned_corr = float(aligned_correlations[i, j])
            if distance > 6 and aligned_corr < 0.90:
                continue
            dy = int(alignment_dys[i, j])
            dx = int(alignment_dxs[i, j])
            dice = _mask_dice(masks[i], masks[j], dy, dx)
            evidence: list[str] = []
            if distance <= 6:
                evidence.append("phash_le_6")
            if aligned_corr >= 0.95:
                evidence.append("aligned_corr_ge_0.95")
            if aligned_corr >= 0.90 and dice >= 0.70:
                evidence.append("corr_ge_0.90_and_mask_dice_ge_0.70")
            if not evidence:
                continue
            sample_a, sample_b = sorted((left["sample_id"], right["sample_id"]))
            candidate_id = "near_" + hashlib.sha256(
                f"{sample_a}|{sample_b}".encode()
            ).hexdigest()[:12]
            reviewer_1, reviewer_2 = reviews.get(candidate_id, ("", ""))
            should_group, review_state = _review_outcome(reviewer_1, reviewer_2)
            candidates.append({
                "candidate_id": candidate_id,
                "sample_a": sample_a,
                "sample_b": sample_b,
                "phash_distance": distance,
                "zero_shift_correlation": round(zero_corr, 8),
                "aligned_correlation": round(aligned_corr, 8),
                "alignment_dy": dy,
                "alignment_dx": dx,
                "registered_mask_dice": round(dice, 8),
                "evidence": ";".join(evidence),
                "reviewer_1_decision": reviewer_1,
                "reviewer_2_decision": reviewer_2,
                "review_state": review_state,
                "group_conservatively": int(should_group),
            })
    return sorted(candidates, key=lambda row: row["candidate_id"])


def _assign_content_groups(
    records: list[dict[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> None:
    eligible = [record["sample_id"] for record in records if record["eligible"]]
    union = UnionFind(eligible)
    exact_members: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record["eligible"] and record["exact_duplicate_group"]:
            exact_members[record["exact_duplicate_group"]].append(record["sample_id"])
    for members in exact_members.values():
        for member in members[1:]:
            union.union(members[0], member)
    eligible_set = set(eligible)
    for candidate in candidates:
        if int(candidate["group_conservatively"]):
            left, right = str(candidate["sample_a"]), str(candidate["sample_b"])
            if left in eligible_set and right in eligible_set:
                union.union(left, right)
    group_by_sample: dict[str, str] = {}
    for index, members in enumerate(sorted(union.groups(), key=lambda group: group[0])):
        for sample_id in members:
            group_by_sample[sample_id] = f"content_{index:04d}"
    evidence: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        if int(candidate["group_conservatively"]):
            value = f"{candidate['candidate_id']}:{candidate['review_state']}"
            evidence[str(candidate["sample_a"])].append(value)
            evidence[str(candidate["sample_b"])].append(value)
    for record in records:
        sample_id = record["sample_id"]
        record["content_group"] = group_by_sample.get(sample_id, "")
        record["near_duplicate_group"] = group_by_sample.get(sample_id, "") if evidence[sample_id] else ""
        record["grouping_evidence"] = ";".join(sorted(evidence[sample_id]))


def _populate_geometry(records: list[dict[str, Any]]) -> None:
    for record in records:
        image, mask = record["image"], record["binary_mask"]
        if image is None:
            record["aspect_ratio"] = 0.0
            record["lesion_fraction"] = 0.0
        else:
            record["aspect_ratio"] = image.shape[1] / image.shape[0]
            record["lesion_fraction"] = float(np.count_nonzero(mask) / mask.size)


def _assign_bins(records: list[dict[str, Any]]) -> None:
    for diagnosis in ("benign", "malignant"):
        positive = [r for r in records if r["eligible"] and r["diagnosis"] == diagnosis]
        if positive:
            cuts = np.quantile([r["lesion_fraction"] for r in positive], [0.25, 0.5, 0.75])
            for record in positive:
                record["lesion_size_quartile"] = int(
                    np.searchsorted(cuts, record["lesion_fraction"], side="right") + 1
                )
    for record in records:
        record.setdefault("lesion_size_quartile", 0)
        ratio = record["aspect_ratio"]
        record["aspect_ratio_bin"] = (
            "portrait" if ratio < 0.8 else "landscape" if ratio > 1.25 else "square"
        )


def _stratum(record: Mapping[str, Any]) -> str:
    return f"{record['diagnosis']}|q{record['lesion_size_quartile']}|{record['aspect_ratio_bin']}"


def _group_records(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["eligible"]:
            groups[record["content_group"]].append(record)
    return groups


def _score(
    counts: Counter[str], total: int, targets: Mapping[str, float], target_total: float
) -> float:
    result = ((total - target_total) / max(target_total, 1.0)) ** 2
    return result + sum(
        ((counts[key] - target) / max(target, 1.0)) ** 2
        for key, target in targets.items()
    )


def _select_groups(
    groups: Mapping[str, Sequence[dict[str, Any]]],
    fraction: float,
    seed: int,
    excluded: set[str] | None = None,
) -> set[str]:
    available = {k: v for k, v in groups.items() if k not in (excluded or set())}
    all_records = [record for members in available.values() for record in members]
    target_total = round(len(all_records) * fraction)
    totals = Counter(_stratum(record) for record in all_records)
    targets = {key: value * fraction for key, value in totals.items()}
    rng = random.Random(seed)
    ties = {key: rng.random() for key in available}
    selected: set[str] = set()
    counts: Counter[str] = Counter()
    size = 0
    while available and size < target_total:
        best = min(available, key=lambda key: (
            _score(
                counts + Counter(_stratum(r) for r in available[key]),
                size + len(available[key]), targets, target_total,
            ), ties[key], key,
        ))
        members = available.pop(best)
        selected.add(best)
        counts.update(_stratum(record) for record in members)
        size += len(members)
    return selected


def _assign_folds(
    groups: Mapping[str, Sequence[dict[str, Any]]], folds: int, seed: int
) -> dict[str, int]:
    records = [record for members in groups.values() for record in members]
    target_size = len(records) / folds
    totals = Counter(_stratum(record) for record in records)
    targets = {key: value / folds for key, value in totals.items()}
    rng = random.Random(seed)
    ties = {key: rng.random() for key in groups}
    rarity = {
        key: sum(1.0 / totals[_stratum(record)] for record in members)
        for key, members in groups.items()
    }
    ordered = sorted(groups, key=lambda key: (-len(groups[key]), -rarity[key], ties[key], key))
    counts = [Counter() for _ in range(folds)]
    sizes = [0] * folds
    assignment: dict[str, int] = {}

    def imbalance(candidate_fold: int, members: Sequence[dict[str, Any]]) -> float:
        """Minimize variance across all folds, not distance of one fold to target.

        A one-fold target-distance objective rewards putting different strata in
        the same fold. Global normalized variance balances both sample counts and
        every diagnosis/size/aspect stratum.
        """

        trial_sizes = sizes.copy()
        trial_sizes[candidate_fold] += len(members)
        trial_counts = [value.copy() for value in counts]
        trial_counts[candidate_fold].update(_stratum(record) for record in members)
        size_variance = float(np.var(np.asarray(trial_sizes) / target_size))
        stratum_variances = [
            float(np.var([
                trial_counts[fold][key] / max(target, 1.0)
                for fold in range(folds)
            ]))
            for key, target in targets.items()
        ]
        return 2.0 * size_variance + float(np.mean(stratum_variances))

    for group_id in ordered:
        members = groups[group_id]
        member_counts = Counter(_stratum(record) for record in members)
        fold = min(range(folds), key=lambda value: (
            imbalance(value, members),
            sizes[value], value,
        ))
        assignment[group_id] = fold
        counts[fold].update(member_counts)
        sizes[fold] += len(members)
    return assignment


def _assign_splits(records: list[dict[str, Any]], config: GenerationConfig) -> None:
    groups = _group_records(records)
    forced_development = {
        record["content_group"] for record in records
        if record["eligible"] and "test" in record["previous_split"].split(";")
    }
    sealed = _select_groups(groups, config.sealed_fraction, config.split_seed, forced_development)
    development = {key: value for key, value in groups.items() if key not in sealed}
    fold_by_group = _assign_folds(development, config.folds, config.split_seed + 1)
    calibration: dict[int, set[str]] = {}
    for outer in range(config.folds):
        pool = {key: value for key, value in development.items() if fold_by_group[key] != outer}
        calibration[outer] = _select_groups(
            pool, config.calibration_fraction, config.split_seed + 1000 + outer
        )
    for record in records:
        group = record["content_group"]
        if not record["eligible"]:
            record.update(partition="excluded", fold=-1, calibration_for_folds="")
        elif group in sealed:
            record.update(partition="sealed", fold=-1, calibration_for_folds="")
        else:
            record["partition"] = "development"
            record["fold"] = fold_by_group[group]
            record["calibration_for_folds"] = ";".join(
                str(outer) for outer in range(config.folds) if group in calibration[outer]
            )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _review_panels(
    output_root: Path,
    candidates: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> None:
    panel_dir = output_root / "review" / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    by_id = {record["sample_id"]: record for record in records}
    for candidate in candidates:
        tiles: list[np.ndarray] = []
        for field in ("sample_a", "sample_b"):
            record = by_id[str(candidate[field])]
            image = cv2.resize(record["image"], (320, 240), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(record["binary_mask"], (320, 240), interpolation=cv2.INTER_NEAREST)
            tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            overlay = np.zeros_like(tile)
            overlay[..., 2] = (mask > 0).astype(np.uint8) * 255
            tile = cv2.addWeighted(tile, 0.75, overlay, 0.25, 0)
            cv2.putText(
                tile, str(record["sample_id"]), (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 1, cv2.LINE_AA,
            )
            tiles.append(tile)
        cv2.imwrite(
            str(panel_dir / f"{candidate['candidate_id']}.png"),
            np.concatenate(tiles, axis=1),
        )


def _materialize(output_root: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        prefix = Path() if record["eligible"] else Path("quarantine")
        image_relative = prefix / "images" / f"{record['sample_id']}.png"
        mask_relative = prefix / "masks" / f"{record['sample_id']}.png"
        image_destination = output_root / image_relative
        mask_destination = output_root / mask_relative
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        mask_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["source_image"], image_destination)
        class_mask = record["binary_mask"].astype(np.uint8) * int(record["class_id"])
        if not cv2.imwrite(str(mask_destination), class_mask):
            raise OSError(f"Failed to write mask: {mask_destination}")
        record["image_path"] = image_relative.as_posix()
        record["mask_path"] = mask_relative.as_posix()
        record["generated_image_sha256"] = sha256_file(image_destination)
        record["generated_mask_sha256"] = sha256_file(mask_destination)


def _mask_geometry(mask: np.ndarray) -> tuple[int, str]:
    binary = (mask > 0).astype(np.uint8)
    count, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return max(0, int(count) - 1), ""
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return max(0, int(count) - 1), f"{x0},{y0},{x1 - x0 + 1},{y1 - y0 + 1}"


def _finalize(record: dict[str, Any], config: GenerationConfig) -> dict[str, Any]:
    image, mask = record["image"], record["binary_mask"]
    height, width = image.shape if image is not None else (0, 0)
    components, bbox = _mask_geometry(mask)
    source_masks = record["source_masks"]
    positive_mask_ok = record["diagnosis"] == "normal" or bool(np.any(mask))
    return {
        "sample_id": record["sample_id"],
        "diagnosis": record["diagnosis"],
        "class_id": record["class_id"],
        "eligible": int(record["eligible"]),
        "exclusion_reason": record["exclusion_reason"],
        "source_image_path": record["source_image"].relative_to(config.raw_root).as_posix(),
        "source_mask_paths": ";".join(
            path.relative_to(config.raw_root).as_posix() for path in source_masks
        ),
        "image_path": record["image_path"],
        "mask_path": record["mask_path"],
        "original_height": height,
        "original_width": width,
        "aspect_ratio": round(width / height, 8) if height else 0.0,
        "lesion_pixels": int(np.count_nonzero(mask)),
        "lesion_fraction": round(float(np.count_nonzero(mask) / mask.size), 10),
        "connected_components": components,
        "bounding_box_xywh": bbox,
        "lesion_size_quartile": record["lesion_size_quartile"],
        "aspect_ratio_bin": record["aspect_ratio_bin"],
        "mask_count": len(source_masks),
        "multi_mask": int(len(source_masks) > 1),
        "image_read_ok": int(image is not None),
        "mask_qc_ok": int(positive_mask_ok and not any(
            note.startswith(("unreadable_mask", "mask_shape_mismatch"))
            for note in record["qc_notes"]
        )),
        "qc_notes": ";".join(record["qc_notes"]),
        "source_image_sha256": sha256_file(record["source_image"]),
        "decoded_pixel_sha256": record["decoded_pixel_sha256"],
        "source_mask_sha256s": ";".join(sha256_file(path) for path in source_masks),
        "union_mask_sha256": sha256_array(mask.astype(np.uint8) * int(record["class_id"])),
        "generated_image_sha256": record["generated_image_sha256"],
        "generated_mask_sha256": record["generated_mask_sha256"],
        "exact_duplicate_group": record["exact_duplicate_group"],
        "near_duplicate_group": record["near_duplicate_group"],
        "content_group": record["content_group"],
        "grouping_evidence": record["grouping_evidence"],
        "previous_split": record["previous_split"],
        "previous_names": record["previous_names"],
        "partition": record["partition"],
        "fold": record["fold"],
        "calibration_for_folds": record["calibration_for_folds"],
        "split_seed": config.split_seed,
        "manifest_version": MANIFEST_VERSION,
        "dataset_fingerprint": "",
    }


def generate_busi_dataset(config: GenerationConfig) -> dict[str, Any]:
    """Generate a non-destructive BUSI v2 tree and return its summary."""

    config.validate()
    records = _scan_raw(config)
    if config.expected_prior_test_count is not None:
        prior_test_count = sum(
            "test" in str(record.get("previous_split", "")).split(";")
            for record in records
            if record.get("eligible")
        )
        if prior_test_count != int(config.expected_prior_test_count):
            raise RuntimeError(
                "Recovered prior BUSI test membership differs from the locked count: "
                f"{prior_test_count} != {config.expected_prior_test_count}"
            )
    _populate_geometry(records)
    exact_rows = _assign_exact_groups(records)
    _assign_bins(records)
    candidates = _near_candidates(records, config.review_csv)
    _assign_content_groups(records, candidates)
    _assign_splits(records, config)
    config.output_root.mkdir(parents=True, exist_ok=False)
    _materialize(config.output_root, records)

    exact_fields = [
        "exact_group", "decoded_pixel_sha256", "sample_ids", "diagnoses",
        "contradictory", "action",
    ]
    candidate_fields = [
        "candidate_id", "sample_a", "sample_b", "phash_distance",
        "zero_shift_correlation", "aligned_correlation", "alignment_dy",
        "alignment_dx", "registered_mask_dice", "evidence",
        "reviewer_1_decision", "reviewer_2_decision", "review_state",
        "group_conservatively",
    ]
    _write_csv(config.output_root / "review" / "exact_duplicates.csv", exact_rows, exact_fields)
    _write_csv(
        config.output_root / "review" / "near_duplicate_candidates.csv",
        candidates, candidate_fields,
    )
    _write_csv(
        config.output_root / "review" / "two_reviewer_template.csv",
        candidates, candidate_fields,
    )
    completed_review_path = None
    reviewer_paths = {}
    if config.review_csv is not None:
        completed_review_path = (
            config.output_root / "review" / "completed_two_reviewer_review.csv"
        )
        _write_csv(completed_review_path, candidates, candidate_fields)
        for reviewer_number in (1, 2):
            reviewer_path = (
                config.output_root
                / "review"
                / f"reviewer_{reviewer_number}_decisions.csv"
            )
            source_reviewer = (
                config.review_csv.parent
                / f"reviewer_{reviewer_number}_decisions.csv"
            )
            if source_reviewer.is_file():
                shutil.copy2(source_reviewer, reviewer_path)
            else:
                decision_field = f"reviewer_{reviewer_number}_decision"
                reviewer_rows = [
                    {
                        "candidate_id": row["candidate_id"],
                        decision_field: row[decision_field],
                    }
                    for row in candidates
                ]
                _write_csv(
                    reviewer_path,
                    reviewer_rows,
                    ["candidate_id", decision_field],
                )
            reviewer_paths[str(reviewer_number)] = reviewer_path
    _review_panels(config.output_root, candidates, records)

    finalized = [_finalize(record, config) for record in records]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in finalized:
        if int(row["eligible"]):
            groups[row["content_group"]].append(row)
    group_rows = [{
        "content_group": group_id,
        "sample_ids": ";".join(sorted(row["sample_id"] for row in members)),
        "diagnoses": ";".join(sorted({row["diagnosis"] for row in members})),
        "partition": members[0]["partition"],
        "fold": members[0]["fold"],
        "calibration_for_folds": members[0]["calibration_for_folds"],
        "member_count": len(members),
    } for group_id, members in sorted(groups.items())]
    _write_csv(
        config.output_root / "review" / "content_groups.csv",
        group_rows,
        ["content_group", "sample_ids", "diagnoses", "partition", "fold",
         "calibration_for_folds", "member_count"],
    )

    fingerprint_payload = [
        {key: value for key, value in row.items() if key != "dataset_fingerprint"}
        for row in sorted(finalized, key=lambda item: item["sample_id"])
    ]
    fingerprint = _canonical_hash(fingerprint_payload)
    for row in finalized:
        row["dataset_fingerprint"] = fingerprint
    manifest_path = config.output_root / "manifest.csv"
    _write_csv(manifest_path, finalized, MANIFEST_FIELDS)
    manifest_hash = sha256_file(manifest_path)
    (config.output_root / "manifest.sha256").write_text(
        f"{manifest_hash}  manifest.csv\n", encoding="utf-8"
    )
    eligible = [row for row in finalized if int(row["eligible"])]
    development_count = sum(row["partition"] == "development" for row in eligible)
    sealed_count = sum(row["partition"] == "sealed" for row in eligible)
    review_state_counts = dict(Counter(row["review_state"] for row in candidates))
    review_complete = all(
        bool(row["reviewer_1_decision"]) and bool(row["reviewer_2_decision"])
        for row in candidates
    )
    metadata = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_fingerprint": fingerprint,
        "manifest_sha256": manifest_hash,
        "split_seed": config.split_seed,
        "sealed_fraction_requested": config.sealed_fraction,
        "calibration_fraction_requested": config.calibration_fraction,
        "folds": config.folds,
        "include_normal": config.include_normal,
        "split_description": "content-grouped; patient IDs unavailable",
        "sealed_description": (
            "prospectively sealed for this study but not historically pristine; "
            "BUSI informed earlier development"
        ),
        "review_policy": (
            "two explicit separate decisions release a pair; pending or disputed "
            "near-duplicate candidates remain conservatively grouped"
        ),
        "duplicate_review": {
            "candidate_count": len(candidates),
            "state_counts": review_state_counts,
            "complete": review_complete,
            "review_csv": (
                completed_review_path.relative_to(config.output_root).as_posix()
                if completed_review_path is not None
                else None
            ),
            "review_csv_sha256": (
                sha256_file(completed_review_path)
                if completed_review_path is not None
                else None
            ),
            "reviewer_decision_csvs": {
                reviewer: path.relative_to(config.output_root).as_posix()
                for reviewer, path in reviewer_paths.items()
            },
            "reviewer_decision_sha256s": {
                reviewer: sha256_file(path)
                for reviewer, path in reviewer_paths.items()
            },
        },
        "counts": {
            "raw": len(finalized),
            "eligible": len(eligible),
            "excluded": len(finalized) - len(eligible),
            "development": development_count,
            "sealed": sealed_count,
            "by_diagnosis": dict(Counter(row["diagnosis"] for row in eligible)),
            "by_fold": dict(Counter(
                str(row["fold"]) for row in eligible if row["partition"] == "development"
            )),
        },
    }
    (config.output_root / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "eligible_samples": len(eligible),
        "development_samples": development_count,
        "sealed_samples": sealed_count,
        "excluded_samples": len(finalized) - len(eligible),
        "near_duplicate_candidates": len(candidates),
        "duplicate_review_complete": review_complete,
        "dataset_fingerprint": fingerprint,
        "manifest_sha256": manifest_hash,
    }


_INT_FIELDS = {
    "class_id", "eligible", "original_height", "original_width", "lesion_pixels",
    "connected_components", "lesion_size_quartile", "mask_count", "multi_mask",
    "image_read_ok", "mask_qc_ok", "fold", "split_seed",
}
_FLOAT_FIELDS = {"aspect_ratio", "lesion_fraction"}


def load_manifest(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load a generated manifest and convert its numeric columns."""

    manifest_path = Path(path)
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in _INT_FIELDS:
            if field in row and row[field] != "":
                row[field] = int(row[field])
        for field in _FLOAT_FIELDS:
            if field in row and row[field] != "":
                row[field] = float(row[field])
    return rows


def resolve_manifest_path(
    manifest_path: str | os.PathLike[str], relative_path: str | os.PathLike[str]
) -> Path:
    """Resolve an artifact path relative to the manifest's dataset root."""

    value = Path(relative_path)
    return value if value.is_absolute() else Path(manifest_path).resolve().parent / value


def resolve_sample_paths(
    manifest_path: str | os.PathLike[str], row: Mapping[str, Any]
) -> tuple[Path, Path]:
    """Resolve one manifest row's canonical image and mask paths."""

    return (
        resolve_manifest_path(manifest_path, str(row["image_path"])),
        resolve_manifest_path(manifest_path, str(row["mask_path"])),
    )


def select_manifest_rows(
    rows: Sequence[Mapping[str, Any]], split: str, outer_fold: int | None = None
) -> list[dict[str, Any]]:
    """Select development, sealed, fit, calibration, or outer rows."""

    valid = {"development", "sealed", "fit", "calibration", "outer"}
    if split not in valid:
        raise ValueError(f"split must be one of {sorted(valid)}, got {split!r}")
    if split in {"fit", "calibration", "outer"} and outer_fold is None:
        raise ValueError(f"outer_fold is required for split={split!r}")
    selected: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if not int(row.get("eligible", 0)):
            continue
        partition = row.get("partition")
        if split in {"development", "sealed"}:
            if partition == split:
                selected.append(row)
            continue
        if partition != "development":
            continue
        fold = int(row["fold"])
        calibration = {
            int(value)
            for value in str(row.get("calibration_for_folds", "")).split(";")
            if value != ""
        }
        if split == "outer" and fold == outer_fold:
            selected.append(row)
        elif split == "calibration" and fold != outer_fold and outer_fold in calibration:
            selected.append(row)
        elif split == "fit" and fold != outer_fold and outer_fold not in calibration:
            selected.append(row)
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def compute_class_pixel_counts(
    rows: Sequence[Mapping[str, Any]],
    manifest_path: str | os.PathLike[str],
    preprocess: Callable[[np.ndarray, np.ndarray], Any] | None = None,
) -> np.ndarray:
    """Count class pixels after optional image/mask preprocessing.

    The callable may return ``(image, mask)`` or a mapping containing ``mask``.
    """

    counts = np.zeros(3, dtype=np.int64)
    for row in rows:
        image_path, mask_path = resolve_sample_paths(manifest_path, row)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Unable to read image/mask for {row['sample_id']}")
        if preprocess is not None:
            result = preprocess(image, mask)
            mask = np.asarray(result["mask"] if isinstance(result, Mapping) else result[1])
        values = set(int(value) for value in np.unique(mask))
        if not values.issubset({0, 1, 2}):
            raise ValueError(f"Invalid class IDs for {row['sample_id']}: {sorted(values)}")
        counts += np.bincount(mask.astype(np.int64).ravel(), minlength=3)[:3]
    return counts


class BUSIManifestDataset(Dataset):
    """Manifest-backed raw dataset returning image, class-ID mask, and metadata."""

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        split: str,
        outer_fold: int | None = None,
        transform: Callable[..., Mapping[str, Any]] | None = None,
        image_loader: Callable[[Path], np.ndarray | None] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = select_manifest_rows(
            load_manifest(self.manifest_path), split=split, outer_fold=outer_fold
        )
        self.transform = transform
        self.image_loader = image_loader or (
            lambda path: cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        row = self.rows[index]
        image_path, mask_path = resolve_sample_paths(self.manifest_path, row)
        image_gray = self.image_loader(image_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_gray is None or mask is None:
            raise ValueError(f"Unable to read sample {row['sample_id']}")
        image = (
            cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
            if image_gray.ndim == 2
            else cv2.cvtColor(image_gray, cv2.COLOR_BGR2RGB)
        )
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask shape mismatch for {row['sample_id']}")
        values = set(int(value) for value in np.unique(mask))
        if not values.issubset({0, 1, 2}):
            raise ValueError(f"Invalid mask values for {row['sample_id']}: {sorted(values)}")
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        else:
            mask = mask.astype(np.int64, copy=False)
        metadata = {
            "sample_id": row["sample_id"],
            "diagnosis": row["diagnosis"],
            "class_id": int(row["class_id"]),
            "content_group": row["content_group"],
            "partition": row["partition"],
            "fold": int(row["fold"]),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
        return image, mask, metadata
