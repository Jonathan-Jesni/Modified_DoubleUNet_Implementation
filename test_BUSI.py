"""Calibrate and test the corrected BUSI model; no arguments are required."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from BUSI_model import build_doubleunet, configure_screening_architecture
from busi_evaluation import (
    apply_foreground_threshold,
    evaluate_probabilities,
    sweep_foreground_thresholds,
)
from busi_runtime import (
    ManifestSegmentationDataset,
    atomic_json_dump,
    canonical_sha256,
    load_json,
    sha256_file,
    split_membership_hash,
    validate_dataset_contract,
    verify_generated_artifacts,
    worker_seed_init,
)
from train_BUSI import write_deploy_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/BUSI")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def _metadata_values(metadata: Mapping[str, Any], key: str) -> list[Any]:
    value = metadata[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _quartile_number(value: Any) -> int:
    text = str(value).strip().upper()
    if text in {"", "NORMAL", "NONE", "0"}:
        return 0
    if text.startswith("Q"):
        text = text[1:]
    quartile = int(text)
    if quartile not in {1, 2, 3, 4}:
        raise ValueError(f"Unexpected lesion-size quartile: {value!r}")
    return quartile


def _candidate_paths(run_dir: Path, explicit: str | None) -> list[Path]:
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            path.relative_to(run_dir)
        except ValueError as exc:
            raise RuntimeError("Checkpoint must belong to the selected run directory") from exc
        return [path]
    last_path = run_dir / "last.pt"
    if not last_path.is_file():
        raise FileNotFoundError(f"Train first; missing checkpoint: {last_path}")
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    paths = [Path(item["path"]).resolve() for item in last.get("candidate_checkpoints", [])]
    paths = [path for path in paths if path.is_file()]
    if paths:
        return paths[:3]
    best = run_dir / "best_primary.pt"
    return [best.resolve()] if best.is_file() else [last_path.resolve()]


def _load_model(checkpoint_path: Path, config: Mapping[str, Any], device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Not a full BUSI checkpoint: {checkpoint_path}")
    if checkpoint.get("config_hash") != canonical_sha256(config):
        raise RuntimeError(f"Checkpoint configuration does not match this run: {checkpoint_path}")
    if checkpoint.get("run_id") != config.get("run_id"):
        raise RuntimeError(f"Checkpoint run ID does not match this run: {checkpoint_path}")
    model = build_doubleunet(
        variant=config["variant"],
        num_classes=3,
        preprocessing_profile=config["dataset"]["preprocessing_profile"],
        input_size=config["dataset"]["input_size"],
        pretrained=False,
        bn_policy=(
            "targeted"
            if config["model"]["fine_tuning"]["targeted_bn_policy"]
            else "legacy"
        ),
    )
    configure_screening_architecture(model, config["model"]["architecture_mode"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def _collect(
    checkpoint_path: Path,
    dataset: ManifestSegmentationDataset,
    config: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    model, checkpoint = _load_model(checkpoint_path, config, device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed_init,
    )
    p1_values, p2_values, targets = [], [], []
    class_ids, sample_ids, lesion_quartiles = [], [], []
    with torch.inference_mode():
        for images, masks, metadata in loader:
            images = images.to(device, non_blocking=True)
            p1_logits, p2_logits = model(images)
            p1_values.append(torch.softmax(p1_logits, dim=1).cpu().numpy())
            p2_values.append(torch.softmax(p2_logits, dim=1).cpu().numpy())
            targets.append(masks.numpy())
            class_ids.extend(int(value) for value in _metadata_values(metadata, "class_id"))
            sample_ids.extend(str(value) for value in _metadata_values(metadata, "sample_id"))
            lesion_quartiles.extend(
                _quartile_number(value)
                for value in _metadata_values(metadata, "lesion_size_quartile")
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "p1": np.concatenate(p1_values),
        "p2": np.concatenate(p2_values),
        "targets": np.concatenate(targets),
        "class_ids": np.asarray(class_ids, dtype=np.int64),
        "sample_ids": np.asarray(sample_ids),
        "lesion_quartiles": np.asarray(lesion_quartiles, dtype=np.int64),
        "epoch": int(checkpoint.get("epoch", -1)),
    }


def _overlap_values(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target_fg = target != 0
    prediction_fg = prediction != 0
    tp = int(np.logical_and(target_fg, prediction_fg).sum())
    fp = int(np.logical_and(~target_fg, prediction_fg).sum())
    fn = int(np.logical_and(target_fg, ~prediction_fg).sum())
    binary_dice = (2.0 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    binary_iou = (tp / (tp + fp + fn)) if (tp + fp + fn) else None
    binary_precision = (tp / (tp + fp)) if (tp + fp) else (0.0 if fn else None)
    binary_recall = (tp / (tp + fn)) if (tp + fn) else None
    return {
        "binary_dice": binary_dice,
        "binary_iou": binary_iou,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
    }


def _per_image_rows(bundle: Mapping[str, Any], threshold: float, head: str):
    predictions = apply_foreground_threshold(bundle[head], threshold)
    rows = []
    for index, prediction in enumerate(predictions):
        target = bundle["targets"][index]
        class_id = int(bundle["class_ids"][index])
        foreground_counts = [int((prediction == value).sum()) for value in (1, 2)]
        predicted_diagnosis = int(np.argmax(foreground_counts)) + 1 if any(foreground_counts) else 0
        row = {
            "sample_id": str(bundle["sample_ids"][index]),
            "class_id": class_id,
            "lesion_size_quartile": int(bundle["lesion_quartiles"][index]),
            "predicted_diagnosis": predicted_diagnosis,
            "normal_empty": bool(not np.any(prediction)) if class_id == 0 else None,
            "predicted_foreground_fraction": float((prediction != 0).mean()),
            "ground_truth_foreground_fraction": float((target != 0).mean()),
            **_overlap_values(target, prediction),
        }
        if class_id:
            target_class = target == class_id
            predicted_class = prediction == class_id
            tp = int(np.logical_and(target_class, predicted_class).sum())
            fp = int(np.logical_and(~target_class, predicted_class).sum())
            fn = int(np.logical_and(target_class, ~predicted_class).sum())
            denominator = 2 * tp + fp + fn
            row["diagnosis_class_dice"] = (
                2.0 * tp / denominator if denominator else None
            )
        else:
            row["diagnosis_class_dice"] = None
        rows.append(row)
    return rows


def _quartile_metrics(bundle: Mapping[str, Any], threshold: float, head: str):
    output = {}
    for quartile in (1, 2, 3, 4):
        indices = np.flatnonzero(bundle["lesion_quartiles"] == quartile)
        if not len(indices):
            continue
        output[f"Q{quartile}"] = evaluate_probabilities(
            bundle[head][indices],
            bundle["targets"][indices],
            class_ids=bundle["class_ids"][indices],
            sample_ids=bundle["sample_ids"][indices],
            threshold=threshold,
            compute_surface=False,
        )
    return output


def _dataset(config: Mapping[str, Any], split: str) -> ManifestSegmentationDataset:
    return ManifestSegmentationDataset(
        config["dataset"]["manifest"],
        split=split,
        outer_fold=int(config["outer_fold"]),
        preprocessing_profile=config["dataset"]["preprocessing_profile"],
        input_size=config["dataset"]["input_size"],
        augmentation=None,
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "resolved_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Train first; missing configuration: {config_path}")
    config = load_json(config_path)
    manifest = Path(config["dataset"]["manifest"]).resolve()
    dataset_metadata = validate_dataset_contract(
        manifest,
        require_completed_review=True,
        expected_counts=config["dataset"].get("expected_counts"),
    )
    verify_generated_artifacts(manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibration_dataset = _dataset(config, "calibration")
    candidates = []
    for checkpoint_path in _candidate_paths(run_dir, args.checkpoint):
        bundle = _collect(
            checkpoint_path,
            calibration_dataset,
            config,
            device,
            args.batch_size,
            args.num_workers,
        )
        sweep = sweep_foreground_thresholds(
            bundle["p2"],
            bundle["targets"],
            class_ids=bundle["class_ids"],
            sample_ids=bundle["sample_ids"],
        )
        candidates.append(
            {
                "path": checkpoint_path,
                "epoch": bundle["epoch"],
                "threshold": float(sweep["threshold"]),
                "metrics": sweep["best_metrics"],
                "sweep": sweep["sweep"],
            }
        )
    best_score = max(item["metrics"]["S_bal"] for item in candidates)
    tied = [item for item in candidates if best_score - item["metrics"]["S_bal"] <= 1e-4]
    selected = max(
        tied,
        key=lambda item: (
            item["metrics"]["D_N"],
            item["threshold"],
            str(item["path"]),
        ),
    )

    outer_dataset = _dataset(config, "outer")
    outer_bundle = _collect(
        selected["path"],
        outer_dataset,
        config,
        device,
        args.batch_size,
        args.num_workers,
    )
    p1_metrics = evaluate_probabilities(
        outer_bundle["p1"],
        outer_bundle["targets"],
        class_ids=outer_bundle["class_ids"],
        sample_ids=outer_bundle["sample_ids"],
        threshold=selected["threshold"],
        compute_surface=True,
    )
    p2_metrics = evaluate_probabilities(
        outer_bundle["p2"],
        outer_bundle["targets"],
        class_ids=outer_bundle["class_ids"],
        sample_ids=outer_bundle["sample_ids"],
        threshold=selected["threshold"],
        compute_surface=True,
    )

    probability_path = run_dir / "test_probabilities.npz"
    probability_temporary = run_dir / ".test_probabilities.npz.tmp"
    outer_split_hash = split_membership_hash(outer_dataset.rows)
    checkpoint_hash = sha256_file(selected["path"])
    with probability_temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            p1=outer_bundle["p1"],
            p2=outer_bundle["p2"],
            targets=outer_bundle["targets"],
            class_ids=outer_bundle["class_ids"],
            sample_ids=outer_bundle["sample_ids"],
            lesion_quartiles=outer_bundle["lesion_quartiles"],
            dataset_fingerprint=np.asarray(dataset_metadata["dataset_fingerprint"]),
            manifest_sha256=np.asarray(sha256_file(manifest)),
            outer_split_hash=np.asarray(outer_split_hash),
            checkpoint_sha256=np.asarray(checkpoint_hash),
            threshold=np.asarray(selected["threshold"], dtype=np.float64),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(probability_temporary, probability_path)
    probability_hash = sha256_file(probability_path)
    model, selected_checkpoint = _load_model(selected["path"], config, device)
    write_deploy_checkpoint(
        run_dir,
        model,
        config,
        threshold=selected["threshold"],
        source_checkpoint=selected["path"],
        source_epoch=int(selected_checkpoint.get("epoch", selected["epoch"])),
    )
    selected_relative = selected["path"].relative_to(run_dir).as_posix()
    per_image_path = run_dir / "test_per_image.json"
    atomic_json_dump(
        {
            "threshold": float(selected["threshold"]),
            "p1": _per_image_rows(outer_bundle, selected["threshold"], "p1"),
            "p2": _per_image_rows(outer_bundle, selected["threshold"], "p2"),
        },
        per_image_path,
    )
    report = {
        "checkpoint": selected_relative,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": selected["epoch"],
        "threshold": selected["threshold"],
        "dataset_fingerprint": dataset_metadata["dataset_fingerprint"],
        "manifest_sha256": sha256_file(manifest),
        "outer_split_hash": outer_split_hash,
        "calibration": selected["metrics"],
        "test_p1": p1_metrics,
        "test_p2": p2_metrics,
        "lesion_size_quartiles_p1": _quartile_metrics(
            outer_bundle, selected["threshold"], "p1"
        ),
        "lesion_size_quartiles_p2": _quartile_metrics(
            outer_bundle, selected["threshold"], "p2"
        ),
        "per_image_metrics": per_image_path.name,
        "probabilities": probability_path.name,
        "probabilities_sha256": probability_hash,
    }
    atomic_json_dump(report, run_dir / "test_results.json")
    summary_path = run_dir / "run_summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    summary.update(
        {
            "deployment_epoch": int(selected["epoch"]),
            "selected_checkpoint": selected_relative,
            "locked_threshold": float(selected["threshold"]),
            "test_report": "test_results.json",
            "deploy_checkpoint": "deploy.pt",
        }
    )
    atomic_json_dump(summary, summary_path)
    print(
        f"Test S_bal={p2_metrics['S_bal']:.4f} "
        f"Dice benign={p2_metrics['D_B_bin']:.4f} "
        f"malignant={p2_metrics['D_M_bin']:.4f} "
        f"threshold={selected['threshold']:.2f}"
    )
    print(f"Report: {run_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
