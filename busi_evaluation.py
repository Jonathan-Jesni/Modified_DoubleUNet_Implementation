"""Stable BUSI v2 evaluation and validation-only threshold calibration.

This module preserves legacy helpers in utils.py and provides sample-aware
aggregation that is invariant to DataLoader batch size and filename order.
"""

from __future__ import annotations

from math import fsum

import numpy as np
import torch

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    _SCIPY_SURFACE_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    binary_erosion = None
    distance_transform_edt = None
    _SCIPY_SURFACE_AVAILABLE = False


CLASS_NAMES = {0: "normal", 1: "benign", 2: "malignant"}
_CLASS_IDS = {name: class_id for class_id, name in CLASS_NAMES.items()}


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_batched_labels(value, name):
    array = _to_numpy(value)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [N,H,W] or [H,W]")
    if not np.issubdtype(array.dtype, np.integer):
        rounded = np.rint(array)
        if not np.allclose(array, rounded):
            raise ValueError(f"{name} must contain integer class IDs")
        array = rounded
    array = array.astype(np.int64, copy=False)
    if np.any((array < 0) | (array > 2)):
        raise ValueError(f"{name} contains a class outside {{0,1,2}}")
    return array


def _as_batched_probabilities(value):
    array = _to_numpy(value)
    single = array.ndim == 3
    if single:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[1] != 3:
        raise ValueError("probabilities must have shape [N,3,H,W] or [3,H,W]")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("probabilities must be finite")
    if np.any(array < -1e-6) or np.any(array > 1.0 + 1e-6):
        raise ValueError("probabilities must lie in [0,1]")
    probability_sum = array.sum(axis=1)
    if not np.allclose(probability_sum, 1.0, atol=1e-3, rtol=1e-3):
        raise ValueError("probabilities must sum to one over the class axis")
    array = array / probability_sum[:, None, :, :]
    return array, single


def _normalize_class_id(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized not in _CLASS_IDS:
            raise ValueError(f"unknown BUSI diagnosis {value!r}")
        return _CLASS_IDS[normalized]
    class_id = int(value)
    if class_id not in CLASS_NAMES:
        raise ValueError(f"class_id must be one of {tuple(CLASS_NAMES)}")
    return class_id


def _resolve_class_ids(class_ids, targets):
    count = targets.shape[0]
    if class_ids is None:
        resolved = []
        for target in targets:
            foreground = np.unique(target[target != 0])
            if foreground.size == 0:
                resolved.append(0)
            elif foreground.size == 1:
                resolved.append(int(foreground[0]))
            else:
                raise ValueError(
                    "class_ids are required when a target has multiple foreground classes"
                )
        return resolved

    if isinstance(class_ids, (str, int, np.integer)):
        class_ids = [class_ids]
    elif isinstance(class_ids, torch.Tensor):
        class_ids = class_ids.detach().cpu().reshape(-1).tolist()
    else:
        class_ids = list(class_ids)
    if len(class_ids) != count:
        raise ValueError("class_ids length must equal the batch size")
    return [_normalize_class_id(value) for value in class_ids]


def _validate_target_diagnosis(target, class_id):
    foreground_classes = np.unique(target[target != 0])
    if class_id == 0:
        if foreground_classes.size:
            raise ValueError("normal BUSI samples must have an all-background target")
        return
    if foreground_classes.size == 0:
        raise ValueError("positive BUSI samples must have a non-empty lesion target")
    if not np.array_equal(foreground_classes, np.array([class_id])):
        raise ValueError("target foreground class disagrees with manifest class_id")


def apply_foreground_threshold(probabilities, threshold=0.5):
    """Apply the BUSI foreground threshold, then choose benign or malignant."""

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    array, single = _as_batched_probabilities(probabilities)
    foreground_probability = array[:, 1:].sum(axis=1)
    foreground_class = array[:, 1:].argmax(axis=1).astype(np.int64) + 1
    prediction = np.where(
        foreground_probability >= float(threshold),
        foreground_class,
        0,
    ).astype(np.uint8)
    return prediction[0] if single else prediction


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _precision_from_counts(true_positive, false_positive, false_negative):
    denominator = true_positive + false_positive
    if denominator == 0:
        return 0.0 if false_negative > 0 else None
    return float(true_positive) / denominator


def _dice_from_counts(true_positive, false_positive, false_negative):
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return None
    return (2.0 * true_positive) / denominator


def _iou_from_counts(true_positive, false_positive, false_negative):
    denominator = true_positive + false_positive + false_negative
    if denominator == 0:
        return None
    return float(true_positive) / denominator


def _stable_mean(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return fsum(sorted(values)) / len(values)


def _balanced_mean(*values):
    if any(value is None for value in values):
        return None
    return fsum(sorted(float(value) for value in values)) / len(values)


def _calibration_bins(confidence, correct, bin_count):
    indices = np.minimum((confidence * bin_count).astype(np.int64), bin_count - 1)
    counts = np.bincount(indices.ravel(), minlength=bin_count).astype(np.int64)
    confidence_sum = np.bincount(
        indices.ravel(), weights=confidence.ravel(), minlength=bin_count
    ).astype(np.float64)
    correct_sum = np.bincount(
        indices.ravel(),
        weights=correct.astype(np.float64).ravel(),
        minlength=bin_count,
    ).astype(np.float64)
    return counts, confidence_sum, correct_sum


def _expected_calibration_error(records, prefix, bin_count):
    total_count = sum(record[f"{prefix}_ece_count_total"] for record in records)
    if total_count == 0:
        return None
    error = 0.0
    for bin_index in range(bin_count):
        count = int(
            sum(record[f"{prefix}_ece_counts"][bin_index] for record in records)
        )
        if count == 0:
            continue
        confidence_sum = fsum(
            sorted(record[f"{prefix}_ece_confidence"][bin_index] for record in records)
        )
        correct_sum = fsum(
            sorted(record[f"{prefix}_ece_correct"][bin_index] for record in records)
        )
        error += (count / total_count) * abs(
            (correct_sum / count) - (confidence_sum / count)
        )
    return float(error)


def _surface_distance_metrics(target, prediction):
    if not _SCIPY_SURFACE_AVAILABLE:
        return None
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if not target.any() or not prediction.any():
        return None
    target_surface = np.logical_xor(target, binary_erosion(target, border_value=0))
    prediction_surface = np.logical_xor(
        prediction, binary_erosion(prediction, border_value=0)
    )
    target_to_prediction = distance_transform_edt(~prediction_surface)[target_surface]
    prediction_to_target = distance_transform_edt(~target_surface)[prediction_surface]
    distances = np.concatenate((target_to_prediction, prediction_to_target))
    return {
        "hd95": float(np.percentile(distances, 95)),
        "assd": float(distances.mean()),
    }



class BUSIMetricAccumulator:
    """Memory-bounded, sample-aware BUSI metric aggregation."""

    def __init__(self, threshold=0.5, compute_surface=False, ece_bins=15):
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be in [0,1]")
        if int(ece_bins) < 2:
            raise ValueError("ece_bins must be at least 2")
        self.threshold = float(threshold)
        self.compute_surface = bool(compute_surface)
        self.ece_bins = int(ece_bins)
        self._records = []
        self._sample_ids = set()
        self._next_sample_index = 0

    def update(
        self,
        targets,
        predictions=None,
        probabilities=None,
        class_ids=None,
        sample_ids=None,
    ):
        targets = _as_batched_labels(targets, "targets")
        batch_size, height, width = targets.shape

        probability_array = None
        if probabilities is not None:
            probability_array, _ = _as_batched_probabilities(probabilities)
            if probability_array.shape[0] != batch_size:
                raise ValueError("probabilities batch size does not match targets")
            if probability_array.shape[2:] != (height, width):
                raise ValueError("probability spatial shape does not match targets")

        if predictions is None:
            if probability_array is None:
                raise ValueError("predictions or probabilities must be provided")
            predictions = apply_foreground_threshold(
                probability_array, threshold=self.threshold
            )
        predictions = _as_batched_labels(predictions, "predictions")
        if predictions.shape != targets.shape:
            raise ValueError("prediction shape must match targets")

        resolved_class_ids = _resolve_class_ids(class_ids, targets)
        if sample_ids is None:
            resolved_sample_ids = [
                f"__sample_{self._next_sample_index + offset}"
                for offset in range(batch_size)
            ]
        else:
            if isinstance(sample_ids, str):
                sample_ids = [sample_ids]
            resolved_sample_ids = [str(value) for value in sample_ids]
            if len(resolved_sample_ids) != batch_size:
                raise ValueError("sample_ids length must equal the batch size")

        for index in range(batch_size):
            sample_id = resolved_sample_ids[index]
            if sample_id in self._sample_ids:
                raise ValueError(f"duplicate sample_id {sample_id!r}")
            self._sample_ids.add(sample_id)

            target = targets[index]
            prediction = predictions[index]
            class_id = resolved_class_ids[index]
            _validate_target_diagnosis(target, class_id)

            confusion = np.zeros((3, 3), dtype=np.int64)
            np.add.at(confusion, (target.ravel(), prediction.ravel()), 1)

            target_foreground = target != 0
            predicted_foreground = prediction != 0
            true_positive = int(
                np.logical_and(target_foreground, predicted_foreground).sum()
            )
            false_positive = int(
                np.logical_and(~target_foreground, predicted_foreground).sum()
            )
            false_negative = int(
                np.logical_and(target_foreground, ~predicted_foreground).sum()
            )
            true_negative = int(
                np.logical_and(~target_foreground, ~predicted_foreground).sum()
            )

            if predicted_foreground.any():
                foreground_counts = np.array(
                    [(prediction == 1).sum(), (prediction == 2).sum()]
                )
                predicted_diagnosis = int(foreground_counts.argmax()) + 1
            else:
                predicted_diagnosis = 0

            record = {
                "sample_id": sample_id,
                "class_id": class_id,
                "predicted_diagnosis": predicted_diagnosis,
                "confusion": confusion,
                "binary_tp": true_positive,
                "binary_fp": false_positive,
                "binary_fn": false_negative,
                "binary_tn": true_negative,
                "normal_empty": (
                    bool(not predicted_foreground.any()) if class_id == 0 else None
                ),
                "normal_foreground_fraction": (
                    float(predicted_foreground.mean()) if class_id == 0 else None
                ),
                "binary_dice": (
                    _dice_from_counts(true_positive, false_positive, false_negative)
                    if class_id != 0
                    else None
                ),
                "binary_iou": (
                    _iou_from_counts(true_positive, false_positive, false_negative)
                    if class_id != 0
                    else None
                ),
                "binary_precision": (
                    _precision_from_counts(
                        true_positive, false_positive, false_negative
                    )
                    if class_id != 0
                    else None
                ),
                "binary_recall": (
                    _safe_ratio(true_positive, true_positive + false_negative)
                    if class_id != 0
                    else None
                ),
                "class_dice": None,
                "class_precision": None,
                "class_recall": None,
                "surface": None,
            }

            if class_id != 0:
                class_true_positive = int(confusion[class_id, class_id])
                class_false_positive = int(
                    confusion[:, class_id].sum() - class_true_positive
                )
                class_false_negative = int(
                    confusion[class_id, :].sum() - class_true_positive
                )
                record["class_dice"] = _dice_from_counts(
                    class_true_positive,
                    class_false_positive,
                    class_false_negative,
                )
                record["class_precision"] = _precision_from_counts(
                    class_true_positive,
                    class_false_positive,
                    class_false_negative,
                )
                record["class_recall"] = _safe_ratio(
                    class_true_positive,
                    class_true_positive + class_false_negative,
                )
                if self.compute_surface:
                    record["surface"] = _surface_distance_metrics(
                        target_foreground, predicted_foreground
                    )

            if probability_array is not None:
                probability = probability_array[index]
                pixel_count = int(target.size)
                flat_target = target.ravel()
                flat_probability = probability.reshape(3, -1)
                true_probability = flat_probability[
                    flat_target, np.arange(pixel_count)
                ]
                multiclass_brier_sum = max(0.0, float(
                    np.square(flat_probability).sum()
                    - 2.0 * true_probability.sum()
                    + pixel_count
                ))
                foreground_probability = probability[1:].sum(axis=0)
                binary_target = target_foreground.astype(np.float64)
                binary_brier_sum = float(
                    np.square(foreground_probability - binary_target).sum()
                )

                raw_class_prediction = probability.argmax(axis=0)
                multiclass_confidence = probability.max(axis=0)
                multiclass_correct = raw_class_prediction == target
                multiclass_bins = _calibration_bins(
                    multiclass_confidence,
                    multiclass_correct,
                    self.ece_bins,
                )

                binary_confidence = np.maximum(
                    foreground_probability, 1.0 - foreground_probability
                )
                binary_correct = (
                    foreground_probability >= 0.5
                ) == target_foreground
                binary_bins = _calibration_bins(
                    binary_confidence,
                    binary_correct,
                    self.ece_bins,
                )

                record.update(
                    {
                        "probability_pixel_count": pixel_count,
                        "multiclass_brier_sum": multiclass_brier_sum,
                        "binary_brier_sum": binary_brier_sum,
                        "multiclass_ece_counts": multiclass_bins[0],
                        "multiclass_ece_confidence": multiclass_bins[1],
                        "multiclass_ece_correct": multiclass_bins[2],
                        "multiclass_ece_count_total": pixel_count,
                        "binary_ece_counts": binary_bins[0],
                        "binary_ece_confidence": binary_bins[1],
                        "binary_ece_correct": binary_bins[2],
                        "binary_ece_count_total": pixel_count,
                    }
                )

            self._records.append(record)

        self._next_sample_index += batch_size
        return self


    def compute(self):
        if not self._records:
            raise ValueError("no samples have been accumulated")

        records = self._records
        subgroup_records = {
            class_id: [
                record for record in records if record["class_id"] == class_id
            ]
            for class_id in CLASS_NAMES
        }
        normal_records = subgroup_records[0]
        benign_records = subgroup_records[1]
        malignant_records = subgroup_records[2]
        positive_records = benign_records + malignant_records

        d_normal = _stable_mean(
            float(record["normal_empty"]) for record in normal_records
        )
        d_benign_binary = _stable_mean(
            record["binary_dice"] for record in benign_records
        )
        d_malignant_binary = _stable_mean(
            record["binary_dice"] for record in malignant_records
        )
        d_benign_class = _stable_mean(
            record["class_dice"] for record in benign_records
        )
        d_malignant_class = _stable_mean(
            record["class_dice"] for record in malignant_records
        )
        d_binary_balanced = _balanced_mean(
            d_normal, d_benign_binary, d_malignant_binary
        )
        d_class_balanced = _balanced_mean(
            d_benign_class, d_malignant_class
        )
        balanced_score = _balanced_mean(d_binary_balanced, d_class_balanced)

        confusion = np.zeros((3, 3), dtype=np.int64)
        for record in records:
            confusion += record["confusion"]

        binary_tp = sum(record["binary_tp"] for record in records)
        binary_fp = sum(record["binary_fp"] for record in records)
        binary_fn = sum(record["binary_fn"] for record in records)

        per_class_metrics = {}
        foreground_class_counts = {"tp": 0, "fp": 0, "fn": 0}
        for class_id in (1, 2):
            true_positive = int(confusion[class_id, class_id])
            false_positive = int(confusion[:, class_id].sum() - true_positive)
            false_negative = int(confusion[class_id, :].sum() - true_positive)
            foreground_class_counts["tp"] += true_positive
            foreground_class_counts["fp"] += false_positive
            foreground_class_counts["fn"] += false_negative
            per_class_metrics[CLASS_NAMES[class_id]] = {
                "dice": _dice_from_counts(
                    true_positive, false_positive, false_negative
                ),
                "iou": _iou_from_counts(
                    true_positive, false_positive, false_negative
                ),
                "precision": _precision_from_counts(
                    true_positive,
                    false_positive,
                    false_negative,
                ),
                "recall": _safe_ratio(
                    true_positive, true_positive + false_negative
                ),
                "support_pixels": int(confusion[class_id, :].sum()),
            }

        diagnosis_confusion = np.zeros((3, 3), dtype=np.int64)
        for record in records:
            diagnosis_confusion[
                record["class_id"], record["predicted_diagnosis"]
            ] += 1

        normal_foreground_fractions = [
            record["normal_foreground_fraction"] for record in normal_records
        ]
        ground_truth_foreground_pixels = binary_tp + binary_fn
        predicted_foreground_pixels = binary_tp + binary_fp
        surface_values = [
            record["surface"]
            for record in positive_records
            if record["surface"] is not None
        ]
        probability_records = [
            record for record in records if "probability_pixel_count" in record
        ]
        if probability_records and len(probability_records) != len(records):
            raise ValueError(
                "probabilities must be supplied for every accumulated sample "
                "when probability metrics are requested"
            )
        total_probability_pixels = sum(
            record["probability_pixel_count"] for record in probability_records
        )

        metrics = {
            "sample_count": len(records),
            "subgroup_counts": {
                "normal": len(normal_records),
                "benign": len(benign_records),
                "malignant": len(malignant_records),
            },
            "D_N": d_normal,
            "D_B_bin": d_benign_binary,
            "D_M_bin": d_malignant_binary,
            "D_B_cls": d_benign_class,
            "D_M_cls": d_malignant_class,
            "D_bin_bal": d_binary_balanced,
            "D_cls_bal": d_class_balanced,
            "S_bal": balanced_score,
            "binary_dice_macro_positive": _stable_mean(
                record["binary_dice"] for record in positive_records
            ),
            "binary_iou_macro_positive": _stable_mean(
                record["binary_iou"] for record in positive_records
            ),
            "binary_precision_macro_positive": _stable_mean(
                record["binary_precision"] for record in positive_records
            ),
            "binary_recall_macro_positive": _stable_mean(
                record["binary_recall"] for record in positive_records
            ),
            "binary_dice_micro": _dice_from_counts(
                binary_tp, binary_fp, binary_fn
            ),
            "binary_iou_micro": _iou_from_counts(
                binary_tp, binary_fp, binary_fn
            ),
            "binary_precision_micro": _precision_from_counts(
                binary_tp, binary_fp, binary_fn
            ),
            "binary_recall_micro": _safe_ratio(
                binary_tp, binary_tp + binary_fn
            ),
            "foreground_class_dice_macro": _stable_mean(
                value["dice"] for value in per_class_metrics.values()
            ),
            "foreground_class_precision_macro_positive": _stable_mean(
                record["class_precision"] for record in positive_records
            ),
            "foreground_class_recall_macro_positive": _stable_mean(
                record["class_recall"] for record in positive_records
            ),
            "foreground_class_dice_micro": _dice_from_counts(
                foreground_class_counts["tp"],
                foreground_class_counts["fp"],
                foreground_class_counts["fn"],
            ),
            "foreground_class_precision_micro": _precision_from_counts(
                foreground_class_counts["tp"],
                foreground_class_counts["fp"],
                foreground_class_counts["fn"],
            ),
            "foreground_class_recall_micro": _safe_ratio(
                foreground_class_counts["tp"],
                foreground_class_counts["tp"] + foreground_class_counts["fn"],
            ),

            "per_class_pixel_metrics": per_class_metrics,
            "normal_specificity_image": d_normal,
            "lesion_detection_sensitivity_image": _stable_mean(
                float(record["binary_tp"] > 0) for record in positive_records
            ),
            "normal_foreground_fraction_mean": _stable_mean(
                normal_foreground_fractions
            ),
            "normal_foreground_fraction_p95": (
                float(np.percentile(normal_foreground_fractions, 95))
                if normal_foreground_fractions
                else None
            ),
            "predicted_to_gt_foreground_area_ratio": _safe_ratio(
                predicted_foreground_pixels, ground_truth_foreground_pixels
            ),
            "diagnosis_accuracy_image": _safe_ratio(
                int(np.trace(diagnosis_confusion)),
                int(diagnosis_confusion.sum()),
            ),
            "diagnosis_confusion_matrix": diagnosis_confusion.tolist(),
            "pixel_confusion_matrix": confusion.tolist(),
            "probability_metrics_available": bool(probability_records),
            "binary_brier_score": (
                fsum(
                    sorted(
                        record["binary_brier_sum"]
                        for record in probability_records
                    )
                )
                / total_probability_pixels
                if total_probability_pixels
                else None
            ),
            "multiclass_brier_score": (
                fsum(
                    sorted(
                        record["multiclass_brier_sum"]
                        for record in probability_records
                    )
                )
                / total_probability_pixels
                if total_probability_pixels
                else None
            ),
            "binary_foreground_ece": (
                _expected_calibration_error(
                    probability_records, "binary", self.ece_bins
                )
                if probability_records
                else None
            ),
            "multiclass_ece": (
                _expected_calibration_error(
                    probability_records, "multiclass", self.ece_bins
                )
                if probability_records
                else None
            ),
            "surface_metrics_requested": self.compute_surface,
            "surface_metrics_available": (
                self.compute_surface and _SCIPY_SURFACE_AVAILABLE
            ),
            "surface_valid_pair_count": len(surface_values),
            "surface_skipped_empty_prediction_count": (
                sum(
                    record["binary_tp"] + record["binary_fp"] == 0
                    for record in positive_records
                )
                if self.compute_surface
                else 0
            ),
            "hd95_macro_valid_pairs": _stable_mean(
                value["hd95"] for value in surface_values
            ),
            "assd_macro_valid_pairs": _stable_mean(
                value["assd"] for value in surface_values
            ),
        }
        return metrics



def evaluate_predictions(
    predictions,
    targets,
    class_ids=None,
    probabilities=None,
    sample_ids=None,
    compute_surface=False,
    ece_bins=15,
):
    """Evaluate fixed predictions with sample-stable BUSI aggregation."""

    accumulator = BUSIMetricAccumulator(
        compute_surface=compute_surface,
        ece_bins=ece_bins,
    )
    accumulator.update(
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        class_ids=class_ids,
        sample_ids=sample_ids,
    )
    return accumulator.compute()


def evaluate_probabilities(
    probabilities,
    targets,
    class_ids=None,
    threshold=0.5,
    sample_ids=None,
    compute_surface=False,
    ece_bins=15,
):
    """Apply a foreground threshold and evaluate raw BUSI probabilities."""

    predictions = apply_foreground_threshold(probabilities, threshold=threshold)
    metrics = evaluate_predictions(
        predictions=predictions,
        targets=targets,
        class_ids=class_ids,
        probabilities=probabilities,
        sample_ids=sample_ids,
        compute_surface=compute_surface,
        ece_bins=ece_bins,
    )
    metrics["foreground_threshold"] = float(threshold)
    return metrics


def default_threshold_grid():
    """Return the locked inclusive 0.05..0.99 grid at 0.01 steps."""

    return np.round(np.arange(5, 100, dtype=np.float64) / 100.0, 2)


def sweep_foreground_thresholds(
    probabilities,
    targets,
    class_ids=None,
    thresholds=None,
    tie_tolerance=1e-4,
    sample_ids=None,
    compute_surface=False,
    ece_bins=15,
):
    """Select a validation threshold with the locked BUSI v2 tie rules.

    S_bal is maximized. Scores within tie_tolerance are resolved by higher D_N,
    then by the higher threshold.
    """

    probability_array, _ = _as_batched_probabilities(probabilities)
    target_array = _as_batched_labels(targets, "targets")
    resolved_class_ids = _resolve_class_ids(class_ids, target_array)
    if probability_array.shape[0] != target_array.shape[0]:
        raise ValueError("probabilities batch size does not match targets")
    if probability_array.shape[2:] != target_array.shape[1:]:
        raise ValueError("probability spatial shape does not match targets")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")

    if thresholds is None:
        thresholds = default_threshold_grid()
    thresholds = sorted({float(value) for value in thresholds})
    if not thresholds:
        raise ValueError("thresholds must not be empty")
    if thresholds[0] < 0.0 or thresholds[-1] > 1.0:
        raise ValueError("all thresholds must lie in [0,1]")

    sweep_rows = []
    for threshold in thresholds:
        predictions = apply_foreground_threshold(
            probability_array, threshold=threshold
        )
        selection_metrics = evaluate_predictions(
            predictions=predictions,
            targets=target_array,
            class_ids=resolved_class_ids,
            sample_ids=sample_ids,
            compute_surface=False,
        )
        score = selection_metrics["S_bal"]
        if score is None:
            raise ValueError(
                "threshold calibration requires normal, benign, and malignant samples"
            )
        row = {
            "threshold": threshold,
            "D_N": selection_metrics["D_N"],
            "D_B_bin": selection_metrics["D_B_bin"],
            "D_M_bin": selection_metrics["D_M_bin"],
            "D_B_cls": selection_metrics["D_B_cls"],
            "D_M_cls": selection_metrics["D_M_cls"],
            "D_bin_bal": selection_metrics["D_bin_bal"],
            "D_cls_bal": selection_metrics["D_cls_bal"],
            "S_bal": score,
        }
        sweep_rows.append(row)

    maximum_score = max(row["S_bal"] for row in sweep_rows)
    tied = [
        row
        for row in sweep_rows
        if maximum_score - row["S_bal"] <= tie_tolerance
    ]
    best = max(tied, key=lambda row: (row["D_N"], row["threshold"]))

    best_metrics = evaluate_probabilities(
        probabilities=probability_array,
        targets=target_array,
        class_ids=resolved_class_ids,
        threshold=best["threshold"],
        sample_ids=sample_ids,
        compute_surface=compute_surface,
        ece_bins=ece_bins,
    )
    return {
        "threshold": best["threshold"],
        "best_threshold": best["threshold"],
        "best_metrics": best_metrics,
        "sweep": sweep_rows,
        "tie_tolerance": float(tie_tolerance),
    }
