"""Train the corrected BUSI Modified Double U-Net.

Run without arguments for the standard BUSI configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from BUSI_model import (
    WeightEMA,
    build_doubleunet,
    configure_screening_architecture,
    predict_probabilities,
)
from busi_evaluation import BUSIMetricAccumulator
from busi_runtime import (
    CLASS_MAPPING,
    ManifestSegmentationDataset,
    append_jsonl,
    atomic_json_dump,
    atomic_torch_save,
    canonical_profile,
    canonical_sha256,
    capture_rng_state,
    deep_update,
    diagnosis_balanced_sampler,
    environment_fingerprint,
    git_state,
    load_json,
    load_manifest_rows,
    manifest_fingerprint,
    preprocessed_class_pixel_counts,
    restore_rng_state,
    seed_everything,
    select_rows,
    sha256_file,
    split_membership_hash,
    validate_dataset_contract,
    verify_generated_artifacts,
    worker_seed_init,
)
from metrics import BUSIStageLoss, DeepSupervisionLoss, compute_dynamic_class_weights


CHECKPOINT_SCHEMA = 2
DEFAULT_CONFIG: dict[str, Any] = {
    "study_id": "busi_final",
    "recipe_id": "BUSI_v2_conservative",
    "dataset": {
        "manifest": "dataset_seg_BUSI_v2/manifest.csv",
        "preprocessing_profile": "padded_256_imagenet",
        "input_size": 256,
        "augmentation": "conservative_ultrasound",
        "balanced_sampler": False,
        "sampler_target_probabilities": [0.25, 0.50, 0.25],
        "require_completed_duplicate_review": True,
        "expected_counts": {"eligible": 778, "normal": 133, "benign": 436, "malignant": 209},
    },
    "model": {
        "num_classes": 3,
        "pretrained": True,
        "architecture_mode": "standard",
        "fine_tuning": {
            "discriminative_learning_rates": True,
            "staged_unfreezing": True,
            "targeted_bn_policy": True,
        },
    },
    "loss": {
        "mode": "composite",
        "class_weights": None,
        "class_weight_power": 0.65,
        "localization": "dice",
        "localization_weight": 0.7,
        "class_dice_weight": 0.3,
        "fp_weight": 0.60,
        "fn_weight": 0.40,
        "p1_weight": 0.50,
        "p2_weight": 0.50,
    },
    "training": {
        "batch_size": 8,
        "num_workers": 4,
        "max_epochs": 120,
        "minimum_epochs": 30,
        "early_stopping_patience": 12,
        "minimum_improvement": 0.002,
        "gradient_clip_norm": 5.0,
        "amp": False,
        "deterministic": True,
        "task_lr": 1e-4,
        "backbone_lr": 1e-5,
        "xception_lr": 1e-6,
        "task_weight_decay": 1e-5,
        "backbone_weight_decay": 1e-5,
        "phase_boundaries": [2, 8],
        "scheduler": "plateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 6,
        "warmup_epochs": 0,
        "ema_decay": None,
        "minimum_lr": 1e-7,
        "train_probe_size": 96,
        "equal_optimizer_steps": None,
        "all_development": False,
    },
    "evaluation": {
        "checkpoint_threshold": 0.50,
        "compute_surface_during_training": False,
        "tta": False,
        "selection_metric": "S_bal",
    },
    "metadata": {
        "study_stage": "simple",
        "task": "BUSI three-class segmentation",
        "class_mapping": {"background_or_normal": 0, "benign": 1, "malignant": 2},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional JSON override")
    parser.add_argument("--variant", default="v2", choices=("core", "v2"))
    parser.add_argument("--outer-fold", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--run-dir", default="runs/BUSI")
    parser.add_argument("--resume", default=None, help="Full checkpoint only")
    return parser.parse_args()


def _reject_unknown_overrides(supplied: Mapping[str, Any], reference: Mapping[str, Any], path: str = "") -> None:
    """Fail loudly on config keys that do not exist in DEFAULT_CONFIG.

    deep_update merges anything, so a typo ("scheduler_type" for "scheduler")
    would silently produce a second copy of the baseline. When arms differ by one
    variable, that reads as "the change had no effect" - the most expensive kind
    of quiet failure in a screening ladder.
    """

    for key, value in supplied.items():
        if key.startswith("_"):  # "_comment" and friends are documentation
            continue
        location = f"{path}.{key}" if path else key
        if key not in reference:
            known = ", ".join(sorted(k for k in reference if not k.startswith("_")))
            raise ValueError(
                f"Unknown config key {location!r}. Known keys here: {known}"
            )
        if isinstance(value, Mapping) and isinstance(reference[key], Mapping):
            _reject_unknown_overrides(value, reference[key], location)


def resolve_config(path: str | None, variant: str, outer_fold: int, seed: int) -> dict[str, Any]:
    supplied = load_json(path) if path else {}
    _reject_unknown_overrides(supplied, DEFAULT_CONFIG)
    config = deep_update(DEFAULT_CONFIG, supplied)
    config.pop("_comment", None)
    config.update({"variant": variant, "outer_fold": int(outer_fold), "seed": int(seed)})
    profile = canonical_profile(
        config["dataset"]["preprocessing_profile"], config["dataset"].get("input_size")
    )
    config["dataset"]["preprocessing"] = profile
    config["dataset"]["input_size"] = profile["input_size"]
    if config["model"]["num_classes"] != 3:
        raise ValueError("BUSI training requires exactly three pixel classes")
    if not bool(config["model"].get("pretrained")):
        raise ValueError("BUSI training requires pretrained backbone weights")
    boundaries = list(map(int, config["training"]["phase_boundaries"]))
    if len(boundaries) != 2 or not 0 < boundaries[0] < boundaries[1]:
        raise ValueError("phase_boundaries must be two increasing positive epochs")
    if config["training"]["all_development"]:
        if int(outer_fold) != -1:
            raise ValueError("all-development training requires --outer-fold -1")
        locked_threshold = config.get("evaluation", {}).get("locked_threshold")
        try:
            locked_threshold = float(locked_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "all-development training requires a pooled-OOF locked threshold"
            ) from exc
        if not math.isfinite(locked_threshold) or not 0.0 <= locked_threshold <= 1.0:
            raise ValueError("all-development locked threshold must be finite and in [0,1]")
        config["evaluation"]["locked_threshold"] = locked_threshold
        if config["training"]["minimum_epochs"]:
            config["training"]["minimum_epochs"] = 0
    return config


def _metadata_values(metadata: Mapping[str, Any], key: str) -> list[Any]:
    value = metadata[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def train_probe_indices(rows, size: int, seed: int) -> list[int]:
    """Deterministic class-stratified subsample of the fit pool.

    Held fixed across epochs so the train-side curve is comparable epoch to epoch,
    and stratified so its S_bal is computed over the same three diagnoses as the
    calibration score it is differenced against.
    """

    by_class: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        by_class.setdefault(int(row["class_id"]), []).append(index)
    total = len(rows)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for class_id in sorted(by_class):
        pool = by_class[class_id]
        take = min(len(pool), max(1, round(size * len(pool) / total)))
        chosen.extend(int(value) for value in rng.choice(pool, size=take, replace=False))
    return sorted(chosen)


def make_loaders(config: Mapping[str, Any], generator: torch.Generator):
    dataset_config = config["dataset"]
    training = config["training"]
    manifest = dataset_config["manifest"]
    fold = int(config["outer_fold"])
    all_development = bool(training["all_development"])
    fit_split = "development" if all_development else "fit"
    fit_fold = None if all_development else fold
    train_dataset = ManifestSegmentationDataset(
        manifest,
        split=fit_split,
        outer_fold=fit_fold,
        preprocessing_profile=dataset_config["preprocessing_profile"],
        input_size=dataset_config["input_size"],
        augmentation=dataset_config.get("augmentation"),
    )
    calibration_dataset = None
    if not all_development:
        calibration_dataset = ManifestSegmentationDataset(
            manifest,
            split="calibration",
            outer_fold=fold,
            preprocessing_profile=dataset_config["preprocessing_profile"],
            input_size=dataset_config["input_size"],
            augmentation=None,
        )

    sampler = None
    shuffle = True
    if dataset_config.get("balanced_sampler"):
        sampler = diagnosis_balanced_sampler(
            train_dataset.rows,
            dataset_config.get("sampler_target_probabilities", [0.25, 0.50, 0.25]),
            generator,
        )
        shuffle = False
    common = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        # Fresh workers each epoch make worker RNG replayable from the stored
        # DataLoader generator during an exact interrupted/resumed run.
        "persistent_workers": (
            int(training["num_workers"]) > 0 and not training["deterministic"]
        ),
        "worker_init_fn": worker_seed_init,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator,
        drop_last=False,
        **common,
    )
    calibration_loader = (
        DataLoader(calibration_dataset, shuffle=False, drop_last=False, **common)
        if calibration_dataset is not None
        else None
    )

    # Train-side probe: the same fit images, unaugmented and in eval mode, scored
    # with the same accumulator as calibration. Without this the log carries train
    # LOSS against validation S_bal, which cannot be differenced, so the
    # overfit/underfit gap is unreadable - the reason "no overfitting" has not been
    # a checkable claim on this pipeline.
    probe_loader = None
    probe_size = int(training.get("train_probe_size", 96))
    if probe_size > 0:
        probe_dataset = ManifestSegmentationDataset(
            manifest,
            split=fit_split,
            outer_fold=fit_fold,
            preprocessing_profile=dataset_config["preprocessing_profile"],
            input_size=dataset_config["input_size"],
            augmentation=None,
        )
        indices = train_probe_indices(
            probe_dataset.rows, probe_size, int(config["seed"])
        )
        probe_loader = DataLoader(
            Subset(probe_dataset, indices),
            shuffle=False,
            drop_last=False,
            **common,
        )
    return (
        train_dataset,
        calibration_dataset,
        train_loader,
        calibration_loader,
        probe_loader,
    )


def phase_for_epoch(epoch: int, boundaries: list[int]) -> int:
    if epoch <= boundaries[0]:
        return 1
    if epoch <= boundaries[1]:
        return 2
    return 3


def make_optimizer(model, config: Mapping[str, Any], prior=None):
    fine_tuning = config["model"]["fine_tuning"]
    training = config["training"]
    discriminative = bool(fine_tuning["discriminative_learning_rates"])
    task_lr = float(training["task_lr"])
    groups = model.optimizer_parameter_groups(
        task_lr=task_lr,
        backbone_lr=float(training["backbone_lr"]) if discriminative else task_lr,
        xception_lr=float(training["xception_lr"]) if discriminative else task_lr,
        task_weight_decay=float(training["task_weight_decay"]),
        backbone_weight_decay=(
            float(training["backbone_weight_decay"])
            if discriminative
            else float(training["task_weight_decay"])
        ),
    )
    optimizer = torch.optim.Adam(groups)
    if prior is not None:
        for parameter, state in prior.state.items():
            if parameter.requires_grad:
                optimizer.state[parameter] = state
    return optimizer


def make_scheduler(optimizer, config: Mapping[str, Any]):
    """Plateau (metric-driven) or cosine (schedule-driven) LR decay.

    Cosine exists because the plateau scheduler steps on S_bal, whose D_N term is
    a boolean over ~12 normal calibration images. That gives the score a ~0.014
    quantization step, so "is this a plateau?" is answered partly by noise. Cosine
    never consults the metric.
    """

    training = config["training"]
    kind = str(training.get("scheduler", "plateau"))
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(training["scheduler_factor"]),
            patience=int(training["scheduler_patience"]),
            min_lr=float(training["minimum_lr"]),
        )
    if kind == "cosine":
        warmup = max(0, int(training.get("warmup_epochs", 0)))
        total = max(1, int(training["max_epochs"]))
        floor = float(training["minimum_lr"])
        peaks = [group["lr"] for group in optimizer.param_groups]

        # LambdaLR multiplies each group's own initial_lr by this factor, so the
        # factor must be group-independent for discriminative LRs to keep their
        # ratio. The floor is expressed as a fraction of the largest group's peak
        # so no group decays to exactly zero.
        minimum_factor = floor / max(peaks) if peaks and max(peaks) > 0 else 0.0

        def factor(epoch_index):
            if warmup and epoch_index < warmup:
                return (epoch_index + 1) / (warmup + 1)
            span = max(1, total - warmup)
            progress = min(1.0, max(0.0, (epoch_index - warmup) / span))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_factor + (1.0 - minimum_factor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
    raise ValueError(f"Unsupported scheduler {kind!r}; expected 'plateau' or 'cosine'")


def make_loss(config: Mapping[str, Any], fit_dataset):
    loss_config = config["loss"]
    counts = preprocessed_class_pixel_counts(fit_dataset)
    if loss_config.get("class_weights") is not None:
        weights = list(map(float, loss_config["class_weights"]))
    else:
        weights = compute_dynamic_class_weights(
            counts,
            power=float(loss_config["class_weight_power"]),
            reference_class=1,
        )
    stage = BUSIStageLoss(
        num_classes=3,
        class_weights=weights,
        mode=loss_config["mode"],
        localization=loss_config["localization"],
        localization_weight=float(loss_config["localization_weight"]),
        class_dice_weight=float(loss_config["class_dice_weight"]),
        fp_weight=float(loss_config["fp_weight"]),
        fn_weight=float(loss_config["fn_weight"]),
    )
    return (
        DeepSupervisionLoss(
            stage,
            p1_weight=float(loss_config["p1_weight"]),
            p2_weight=float(loss_config["p2_weight"]),
        ),
        counts,
        weights,
    )


def training_state_diagnostics(model) -> dict[str, Any]:
    batchnorm = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    return {
        "training_phase": int(model.training_phase),
        "bn_policy": model.bn_policy,
        "batchnorm_total": len(batchnorm),
        "batchnorm_training": sum(module.training for module in batchnorm),
        "batchnorm_frozen_statistics": sum(not module.training for module in batchnorm),
        "parameter_tensors_total": len(parameters),
        "parameter_tensors_trainable": len(trainable),
        "parameters_total": sum(parameter.numel() for parameter in parameters),
        "parameters_trainable": sum(parameter.numel() for parameter in trainable),
    }


def run_epoch(model, loader, loss_fn, optimizer, scaler, device, config, ema=None):
    model.train()
    amp = bool(config["training"]["amp"] and device.type == "cuda")
    clip_limit = float(config["training"]["gradient_clip_norm"])
    totals = {"loss": 0.0, "preclip_grad_norm": 0.0, "clipped_steps": 0}
    steps = 0
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            p1, p2 = model(images)
            loss = loss_fn(p1, p2, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite BUSI loss at optimizer step {steps}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_limit)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradient norm")
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        totals["loss"] += float(loss.detach())
        totals["preclip_grad_norm"] += float(grad_norm)
        totals["clipped_steps"] += int(float(grad_norm) > clip_limit)
        steps += 1
    if not steps:
        raise RuntimeError("Training loader yielded no batches")
    return {
        "loss": totals["loss"] / steps,
        "preclip_grad_norm": totals["preclip_grad_norm"] / steps,
        "clipping_frequency": totals["clipped_steps"] / steps,
        "optimizer_steps": steps,
    }


@torch.inference_mode()
def evaluate_loader(model, loader, device, threshold=0.5, compute_surface=False, tta=False):
    model.eval()
    accumulators = [
        BUSIMetricAccumulator(threshold=threshold, compute_surface=compute_surface),
        BUSIMetricAccumulator(threshold=threshold, compute_surface=compute_surface),
    ]
    for images, masks, metadata in loader:
        images = images.to(device, non_blocking=True)
        p1, p2 = predict_probabilities(model, images, tta=tta)
        class_ids = _metadata_values(metadata, "class_id")
        sample_ids = _metadata_values(metadata, "sample_id")
        targets = masks.numpy()
        for accumulator, head in zip(accumulators, (p1, p2)):
            probabilities = head.cpu().numpy()
            accumulator.update(
                targets,
                probabilities=probabilities,
                class_ids=class_ids,
                sample_ids=sample_ids,
            )
    return {"p1": accumulators[0].compute(), "p2": accumulators[1].compute()}


def tensor_state_hash(module: torch.nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def backbone_hashes(model) -> dict[str, str]:
    return {
        "xception_stem": tensor_state_hash(model.e1.xception),
        "densenet_fragments": tensor_state_hash(
            torch.nn.ModuleList([model.e1.dense_block2, model.e1.dense_block3])
        ),
        "vgg19_fragments": tensor_state_hash(
            torch.nn.ModuleList([model.e1.vgg_block4, model.e1.vgg_block5])
        ),
    }


def checkpoint_contract(
    config,
    run_dir: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    phase,
    best_score,
    early_stopping_state,
    candidate_checkpoints,
    generator,
    provenance,
    epoch_log_row,
):
    if config["model"]["fine_tuning"]["staged_unfreezing"]:
        boundaries = config["training"]["phase_boundaries"]
        phase_start = 1 if phase == 1 else boundaries[phase - 2] + 1
    else:
        phase_start = 1
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "run_id": run_dir.name,
        "recipe_id": config["recipe_id"],
        "study_id": config["study_id"],
        "variant": config["variant"],
        "outer_fold": config["outer_fold"],
        "seed": config["seed"],
        "config": config,
        "config_hash": canonical_sha256(config),
        "model_metadata": model.model_metadata(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "amp_scaler_state_dict": scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "unfreeze_phase": int(phase),
        "phase_step": int(epoch - phase_start + 1),
        "best_score": float(best_score),
        "early_stopping_state": dict(early_stopping_state),
        "candidate_checkpoints": list(candidate_checkpoints),
        "epoch_log_row": dict(epoch_log_row),
        "rng_states": capture_rng_state(generator),
        "dataloader_state": {
            "generator_state": generator.get_state(),
            "epoch": int(epoch),
        },
        "sampler_state": {
            "epoch": int(epoch),
            "replacement": bool(config["dataset"].get("balanced_sampler")),
        },
        "git": provenance["git"],
        "environment": provenance["environment"],
        "dataset_fingerprint": provenance["dataset_fingerprint"],
        "dataset_contract_hash": provenance["dataset_contract_hash"],
        "manifest_hash": provenance["manifest_hash"],
        "artifact_inventory_hash": provenance["artifact_inventory_hash"],
        "artifact_file_count": provenance["artifact_file_count"],
        "split_membership_hashes": provenance["split_membership_hashes"],
        "backbone_weight_hashes": provenance["backbone_weight_hashes"],
        "container_digest": os.environ.get("CONTAINER_IMAGE_DIGEST", "unavailable"),
        "host": socket.gethostname(),
        "saved_unix_time": time.time(),
    }


def validate_resume(checkpoint, config, provenance):
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise RuntimeError("Checkpoint schema mismatch; use warm_start explicitly elsewhere")
    expected_run_id = str(config.get("run_id", checkpoint.get("run_id")))
    if config.get("run_id") is not None and checkpoint.get("run_id") != expected_run_id:
        raise RuntimeError("Resume refused: checkpoint belongs to a different run ID")
    required_matches = {
        "config_hash": canonical_sha256(config),
        "dataset_fingerprint": provenance["dataset_fingerprint"],
        "dataset_contract_hash": provenance["dataset_contract_hash"],
        "manifest_hash": provenance["manifest_hash"],
        "artifact_inventory_hash": provenance["artifact_inventory_hash"],
        "artifact_file_count": provenance["artifact_file_count"],
        "split_membership_hashes": provenance["split_membership_hashes"],
    }
    for key, expected in required_matches.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"Resume refused: {key} differs")
    if checkpoint.get("git") != provenance["git"]:
        raise RuntimeError("Resume refused: Git commit or dirty-state fingerprint differs")
    if checkpoint.get("environment", {}).get("digest") != provenance["environment"]["digest"]:
        raise RuntimeError("Resume refused: execution environment differs")

def validate_resume_artifacts(
    resume_path: Path,
    run_dir: Path,
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Refuse cross-run/older resumes and reconcile a checkpoint-first log write."""
    resume_path = resume_path.resolve()
    try:
        resume_path.relative_to(run_dir)
    except ValueError as exc:
        raise RuntimeError("Resume checkpoint must live inside its run directory") from exc
    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)

    resolved_path = run_dir / "resolved_config.json"
    if resolved_path.exists() and load_json(resolved_path) != dict(config):
        raise RuntimeError("Resume refused: existing resolved_config.json differs")
    provenance_path = run_dir / "provenance.json"
    if provenance_path.exists() and load_json(provenance_path) != dict(provenance):
        raise RuntimeError("Resume refused: existing provenance.json differs")

    for candidate in checkpoint.get("candidate_checkpoints", []):
        candidate_path = Path(candidate["path"]).resolve()
        try:
            candidate_path.relative_to(run_dir)
        except ValueError as exc:
            raise RuntimeError("Resume checkpoint carries a cross-run candidate path") from exc
        if not candidate_path.is_file():
            raise RuntimeError(f"Resume candidate checkpoint is missing: {candidate_path}")
        candidate_payload = torch.load(
            candidate_path, map_location="cpu", weights_only=False
        )
        # Candidates are ranked by the run's configured selection metric, so the
        # cross-check has to read that same key. Reading "S_bal" unconditionally
        # would refuse every resume of a run using selection_metric=S_bal_soft.
        candidate_row = candidate_payload.get("epoch_log_row", {})
        candidate_metric = str(
            candidate_row.get("selection_metric")
            or config.get("evaluation", {}).get("selection_metric", "S_bal")
        )
        candidate_score = (
            candidate_row
            .get("calibration_fixed_0_50", {})
            .get("p2", {})
            .get(candidate_metric)
        )
        if (
            candidate_payload.get("run_id") != run_dir.name
            or candidate_payload.get("config_hash") != canonical_sha256(config)
            or int(candidate_payload.get("epoch", -1)) != int(candidate["epoch"])
            or candidate_score is None
            or abs(float(candidate_score) - float(candidate["S_bal"])) > 1e-12
        ):
            raise RuntimeError(
                f"Resume candidate slot is inconsistent with last.pt: {candidate_path}"
            )

    log_path = run_dir / "training.jsonl"
    checkpoint_epoch = int(checkpoint["epoch"])
    if not log_path.exists():
        if checkpoint_epoch:
            raise RuntimeError("Resume refused: training log is missing")
        return
    with open(log_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, len(epochs) + 1)):
        raise RuntimeError("Resume refused: training log epochs are not contiguous")
    logged_epoch = epochs[-1] if epochs else 0
    if logged_epoch == checkpoint_epoch:
        return
    if logged_epoch == checkpoint_epoch - 1:
        missing_row = checkpoint.get("epoch_log_row")
        if not missing_row or int(missing_row.get("epoch", -1)) != checkpoint_epoch:
            raise RuntimeError("Resume refused: checkpoint cannot repair the missing log row")
        append_jsonl(log_path, missing_row)
        return
    if (
        resume_path.name == "last.prev.pt"
        and logged_epoch == checkpoint_epoch + 1
    ):
        # A corrupted newest checkpoint can leave the durable log one epoch
        # ahead of last.prev.pt. Preserve that orphaned row, then roll the log
        # back to the exact recoverable RNG/model state.
        append_jsonl(
            run_dir / "recovery_orphaned.jsonl",
            {
                "reason": "rolled_back_to_last.prev.pt",
                "checkpoint_epoch": checkpoint_epoch,
                "orphaned_epoch_row": rows[-1],
            },
        )
        temporary_log = run_dir / ".training.jsonl.recovery.tmp"
        with temporary_log.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows[:-1]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_log, log_path)
        return
    raise RuntimeError(
        "Resume refused: checkpoint epoch does not match the latest training log"
    )



def write_deploy_checkpoint(
    run_dir: Path,
    model,
    config,
    threshold=None,
    source_checkpoint: Path | None = None,
    source_epoch: int | None = None,
):
    manifest_path = Path(config["dataset"]["manifest"]).resolve()
    dataset_metadata = load_json(manifest_path.parent / "dataset_metadata.json")
    source_metadata = {}
    if source_checkpoint is not None:
        source_path = source_checkpoint.resolve()
        try:
            source_relative = source_path.relative_to(run_dir.resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError("Deploy source checkpoint must belong to its run") from exc
        source_metadata = {
            "source_checkpoint": source_relative,
            "source_checkpoint_sha256": sha256_file(source_path),
            "source_epoch": int(source_epoch) if source_epoch is not None else None,
        }
    deploy = {
        "schema_version": 1,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "variant": config["variant"],
        "model_metadata": model.model_metadata(),
        "preprocessing": config["dataset"]["preprocessing"],
        "class_mapping": CLASS_MAPPING,
        "threshold": threshold,
        "config_hash": canonical_sha256(config),
        "dataset_fingerprint": dataset_metadata["dataset_fingerprint"],
        "dataset_contract_hash": manifest_fingerprint(manifest_path),
        "manifest_hash": sha256_file(manifest_path),
        "artifact_inventory_hash": config["resolved"]["artifact_inventory_hash"],
        "artifact_file_count": config["resolved"]["artifact_file_count"],
        "split_membership_hashes": config["resolved"]["split_membership_hashes"],
        **source_metadata,
    }
    atomic_torch_save(deploy, run_dir / "deploy.pt")


def _create_run_directory(path: str, resume: str | None) -> Path:
    run_dir = Path(path).resolve()
    runs_root = (Path.cwd() / "runs").resolve()
    try:
        run_dir.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError(f"run-dir must be inside {runs_root}") from exc
    if run_dir.exists() and any(run_dir.iterdir()) and resume is None:
        raise FileExistsError(
            f"Run directory is non-empty: {run_dir}; supply --resume or a new run ID"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _rank_candidate(candidates, epoch, score, path):
    updated = [item for item in candidates if Path(item["path"]).resolve() != path.resolve()]
    updated.append({"epoch": int(epoch), "S_bal": float(score), "path": str(path)})
    updated.sort(key=lambda item: (-item["S_bal"], item["epoch"]))
    return updated[:3]


def _candidate_checkpoint_paths(items):
    return {
        Path(item["path"]).resolve()
        for item in items
        if isinstance(item, Mapping) and item.get("path")
    }


def _safe_candidate_slot(run_dir: Path, candidates) -> Path:
    """Choose a bounded slot not referenced by either recoverable last checkpoint."""

    protected = _candidate_checkpoint_paths(candidates)
    previous_path = run_dir / "last.prev.pt"
    if previous_path.is_file():
        previous = torch.load(previous_path, map_location="cpu", weights_only=False)
        if previous.get("schema_version") != CHECKPOINT_SCHEMA:
            raise RuntimeError("last.prev.pt has an unsupported checkpoint schema")
        protected.update(
            _candidate_checkpoint_paths(previous.get("candidate_checkpoints", []))
        )
    slots = [run_dir / "candidates" / f"slot_{index}.pt" for index in range(1, 6)]
    for slot in slots:
        if slot.resolve() not in protected:
            return slot
    raise RuntimeError("No crash-safe candidate checkpoint slot is available")


def main() -> None:
    args = parse_args()
    config = resolve_config(args.config, args.variant, args.outer_fold, args.seed)
    manifest_path = Path(config["dataset"]["manifest"]).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Immutable BUSI manifest not found: {manifest_path}")
    config["dataset"]["manifest"] = str(manifest_path)
    dataset_metadata = validate_dataset_contract(
        manifest_path,
        require_completed_review=bool(
            config["dataset"]["require_completed_duplicate_review"]
        ),
        expected_counts=config["dataset"].get("expected_counts"),
    )
    artifact_contract = verify_generated_artifacts(manifest_path)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU not detected. Use the ROCm PyTorch cloud image (or CUDA PyTorch "
            "on the laptop) before starting the final training run."
        )
    run_dir = _create_run_directory(args.run_dir, args.resume)
    config["run_id"] = run_dir.name
    seed_everything(config["seed"], config["training"]["deterministic"])
    generator = torch.Generator().manual_seed(config["seed"])

    (
        train_dataset,
        calibration_dataset,
        train_loader,
        calibration_loader,
        probe_loader,
    ) = make_loaders(config, generator)
    device = torch.device("cuda")
    use_amp = bool(config["training"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    resume_checkpoint = None
    resume_path = None
    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        try:
            resume_path.relative_to(run_dir)
        except ValueError as exc:
            raise RuntimeError(
                "Resume checkpoint must live inside its run directory"
            ) from exc
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    model = build_doubleunet(
        variant=config["variant"],
        num_classes=3,
        preprocessing_profile=config["dataset"]["preprocessing_profile"],
        input_size=config["dataset"]["input_size"],
        pretrained=bool(config["model"]["pretrained"] and resume_checkpoint is None),
        bn_policy=(
            "targeted"
            if config["model"]["fine_tuning"]["targeted_bn_policy"]
            else "legacy"
        ),
    )
    configure_screening_architecture(model, config["model"]["architecture_mode"])
    initial_phase = (
        int(resume_checkpoint["unfreeze_phase"]) if resume_checkpoint is not None else 1
    )
    if not config["model"]["fine_tuning"]["staged_unfreezing"]:
        initial_phase = 3
    model.set_training_phase(initial_phase)
    model.to(device)
    ema_decay = config["training"].get("ema_decay")
    ema = WeightEMA(model, decay=float(ema_decay)) if ema_decay else None
    loss_fn, pixel_counts, class_weights = make_loss(config, train_dataset)
    loss_fn.to(device)
    print(
        f"BUSI final run | variant={config['variant']} | "
        f"GPU={torch.cuda.get_device_name(0)} | "
        f"backend={'ROCm ' + str(torch.version.hip) if torch.version.hip else 'CUDA'} | "
        f"fit={len(train_dataset)} | calibration="
        f"{len(calibration_dataset) if calibration_dataset is not None else 0}",
        flush=True,
    )
    if config["training"]["all_development"]:
        split_hashes = {
            "development": split_membership_hash(train_dataset.rows),
        }
    else:
        manifest_rows = load_manifest_rows(manifest_path)
        split_hashes = {
            "fit": split_membership_hash(train_dataset.rows),
            "calibration": split_membership_hash(calibration_dataset.rows),
            "outer": split_membership_hash(
                select_rows(manifest_rows, "outer", int(config["outer_fold"]))
            ),
        }
    config["resolved"] = {
        "fit_pixel_counts": pixel_counts,
        "class_weights": class_weights,
        "device": str(device),
        "amp_enabled": use_amp,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_sample_count": len(train_dataset),
        "artifact_inventory_hash": artifact_contract["artifact_inventory_hash"],
        "artifact_file_count": artifact_contract["artifact_file_count"],
        "split_membership_hashes": split_hashes,
    }

    provenance = {
        "git": git_state(Path.cwd()),
        "environment": environment_fingerprint(),
        "dataset_fingerprint": dataset_metadata["dataset_fingerprint"],
        "dataset_contract_hash": manifest_fingerprint(manifest_path),
        "manifest_hash": sha256_file(manifest_path),
        "artifact_inventory_hash": artifact_contract["artifact_inventory_hash"],
        "artifact_file_count": artifact_contract["artifact_file_count"],
        "split_membership_hashes": split_hashes,
        "backbone_weight_hashes": (
            resume_checkpoint.get("backbone_weight_hashes", {})
            if resume_checkpoint is not None
            else backbone_hashes(model)
        ),
    }
    if resume_checkpoint is not None:
        validate_resume(resume_checkpoint, config, provenance)
        validate_resume_artifacts(
            resume_path,
            run_dir,
            resume_checkpoint,
            config,
            provenance,
        )
    atomic_json_dump(config, run_dir / "resolved_config.json")
    atomic_json_dump(provenance, run_dir / "provenance.json")

    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    start_epoch, global_step = 1, 0
    best_score = -math.inf
    early = {"bad_epochs": 0, "best_early_stop_score": -math.inf}
    candidates = []
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(resume_checkpoint["amp_scaler_state_dict"])
        restore_rng_state(resume_checkpoint["rng_states"], generator)
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        global_step = int(resume_checkpoint["global_step"])
        best_score = float(resume_checkpoint["best_score"])
        early = dict(resume_checkpoint["early_stopping_state"])
        candidates = list(resume_checkpoint.get("candidate_checkpoints", []))

    if resume_checkpoint is None:
        phase_start_payload = checkpoint_contract(
            config,
            run_dir,
            model,
            optimizer,
            scheduler,
            scaler,
            0,
            global_step,
            initial_phase,
            best_score,
            early,
            candidates,
            generator,
            provenance,
            {},
        )
        atomic_torch_save(
            phase_start_payload,
            run_dir / f"phase_{initial_phase}_start.pt",
        )

    max_epochs = int(config["training"]["max_epochs"])
    fixed_all_development = bool(config["training"]["all_development"])
    current_phase = initial_phase
    for epoch in range(start_epoch, max_epochs + 1):
        phase = (
            phase_for_epoch(epoch, config["training"]["phase_boundaries"])
            if config["model"]["fine_tuning"]["staged_unfreezing"]
            else 3
        )
        phase_boundary = phase != current_phase
        if phase_boundary:
            model.set_training_phase(phase)
            optimizer = make_optimizer(model, config, prior=optimizer)
            scheduler = make_scheduler(optimizer, config)
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Rebuilding restarts a schedule-driven scheduler at epoch 0. Left
                # alone the cosine would jump back to peak at every phase boundary
                # and never finish its decay inside max_epochs.
                for _ in range(epoch - 1):
                    scheduler.step()
            current_phase = phase
            phase_start_payload = checkpoint_contract(
                config,
                run_dir,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch - 1,
                global_step,
                phase,
                best_score,
                early,
                candidates,
                generator,
                provenance,
                {},
            )
            atomic_torch_save(
                phase_start_payload,
                run_dir / f"phase_{phase}_start.pt",
            )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, config, ema=ema
        )
        global_step += int(train_metrics["optimizer_steps"])
        post_train_state = training_state_diagnostics(model)

        validation = None
        score = None
        is_best = False
        enters_top_three = False
        selection_metric = str(config["evaluation"].get("selection_metric", "S_bal"))
        use_tta = bool(config["evaluation"].get("tta", False))
        ema_validation = None
        if calibration_loader is not None:
            validation = evaluate_loader(
                model,
                calibration_loader,
                device,
                threshold=float(config["evaluation"]["checkpoint_threshold"]),
                compute_surface=bool(
                    config["evaluation"]["compute_surface_during_training"]
                ),
                tta=use_tta,
            )
            if ema is not None:
                # Score the averaged weights alongside the raw ones so a single
                # screening run answers whether EMA helps, then restore.
                live_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
                ema.copy_to(model)
                ema_validation = evaluate_loader(
                    model,
                    calibration_loader,
                    device,
                    threshold=float(config["evaluation"]["checkpoint_threshold"]),
                    compute_surface=False,
                    tta=use_tta,
                )
                model.load_state_dict(live_state, strict=True)
            selected = validation["p2"].get(selection_metric)
            if selected is None:
                raise RuntimeError(
                    f"selection_metric {selection_metric!r} is unavailable on this "
                    "calibration split"
                )
            score = float(selected)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            is_best = score > best_score
            if is_best:
                best_score = score
            prior_top = [float(item["S_bal"]) for item in candidates]
            enters_top_three = len(prior_top) < 3 or score > min(prior_top)
            if score >= early["best_early_stop_score"] + float(
                config["training"]["minimum_improvement"]
            ):
                early = {"bad_epochs": 0, "best_early_stop_score": score}
            else:
                early["bad_epochs"] += 1

        # Same metric, same accumulator, unaugmented fit images: the difference is
        # the fit-regime signal. A large positive gap with a flat validation curve
        # is overfitting; a gap near zero with a low validation score is
        # underfitting. Prior runs on the old pipeline sat at 0.28-0.38 gap
        # unregularized and 0.11-0.16 over-regularized, with validation worse in
        # the second case - so read both numbers, never the gap alone.
        train_probe = None
        fit_regime = None
        if probe_loader is not None:
            train_probe = evaluate_loader(
                model,
                probe_loader,
                device,
                threshold=float(config["evaluation"]["checkpoint_threshold"]),
                compute_surface=False,
            )
            train_score = train_probe["p2"]["S_bal"]
            fit_regime = {
                "train_S_bal": train_score,
                "validation_S_bal": score,
                "S_bal_gap": (
                    float(train_score) - float(score)
                    if train_score is not None and score is not None
                    else None
                ),
                "train_D_bin_bal": train_probe["p2"]["D_bin_bal"],
                "validation_D_bin_bal": (
                    validation["p2"]["D_bin_bal"] if validation is not None else None
                ),
                # Comparable in spirit to the historical "val fg F1" numbers, which
                # S_bal is not: S_bal folds in a normal-image specificity term the
                # old benign/malignant-only pipeline had no equivalent for.
                "train_fg_dice": train_probe["p2"]["binary_dice_macro_positive"],
                "validation_fg_dice": (
                    validation["p2"]["binary_dice_macro_positive"]
                    if validation is not None
                    else None
                ),
                "probe_sample_count": train_probe["p2"]["sample_count"],
            }

        epoch_learning_rates = {
            group.get("name", str(index)): group["lr"]
            for index, group in enumerate(optimizer.param_groups)
        }
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # Schedule-driven: steps every epoch, including all-development runs
            # that have no calibration split to plateau on.
            scheduler.step()

        elapsed = time.perf_counter() - start
        resources = {
            "samples_per_second": len(train_dataset) / max(elapsed, 1e-12),
            "optimizer_steps_per_second": train_metrics["optimizer_steps"]
            / max(elapsed, 1e-12),
            "gpu_peak_allocated_bytes": None,
            "gpu_peak_reserved_bytes": None,
        }
        if device.type == "cuda":
            resources.update(
                {
                    "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
        epoch_row = {
            "epoch": epoch,
            "phase": phase,
            "global_step": global_step,
            "elapsed_seconds": elapsed,
            "train": train_metrics,
            "calibration_fixed_0_50": validation,
            "train_probe": train_probe,
            "ema_calibration": ema_validation,
            "fit_regime": fit_regime,
            "selection_metric": selection_metric,
            "training_state": post_train_state,
            "resources": resources,
            "learning_rates": epoch_learning_rates,
        }
        candidate_path = _safe_candidate_slot(run_dir, candidates) if enters_top_three else None
        prospective_candidates = (
            _rank_candidate(candidates, epoch, score, candidate_path)
            if enters_top_three
            else candidates
        )
        payload = checkpoint_contract(
            config,
            run_dir,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            phase,
            best_score,
            early,
            prospective_candidates,
            generator,
            provenance,
            epoch_row,
        )
        candidates = prospective_candidates
        if enters_top_three:
            atomic_torch_save(payload, candidate_path)
        if is_best:
            atomic_torch_save(payload, run_dir / "best_primary.pt")

        atomic_torch_save(payload, run_dir / "last.pt")
        append_jsonl(run_dir / "training.jsonl", epoch_row)
        score_text = f" | S_bal={score:.4f}" if score is not None else ""
        gap_text = ""
        if fit_regime is not None and fit_regime["S_bal_gap"] is not None:
            gap_text = (
                f" | train={fit_regime['train_S_bal']:.4f}"
                f" gap={fit_regime['S_bal_gap']:+.4f}"
            )
        print(
            f"Epoch {epoch:03d}/{max_epochs} | phase={phase} | "
            f"loss={train_metrics['loss']:.4f}{score_text}{gap_text} | "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        stop = (
            not fixed_all_development
            and epoch >= int(config["training"]["minimum_epochs"])
            and early["bad_epochs"]
            >= int(config["training"]["early_stopping_patience"])
        )
        if stop:
            print(
                f"Early stopping after epoch {epoch}; "
                f"best S_bal={best_score:.4f}",
                flush=True,
            )
            break

    last_checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    final_checkpoint = last_checkpoint
    deployment_source = run_dir / "last.pt"
    if not fixed_all_development and (run_dir / "best_primary.pt").exists():
        final_checkpoint = torch.load(
            run_dir / "best_primary.pt", map_location="cpu", weights_only=False
        )
        deployment_source = run_dir / "best_primary.pt"
    model.load_state_dict(final_checkpoint["model_state_dict"], strict=True)
    locked_threshold = config.get("evaluation", {}).get("locked_threshold")
    write_deploy_checkpoint(
        run_dir,
        model,
        config,
        threshold=locked_threshold,
        source_checkpoint=deployment_source,
        source_epoch=int(final_checkpoint["epoch"]),
    )
    atomic_json_dump(
        {
            "status": "complete",
            "selection_metric": str(
                config.get("evaluation", {}).get("selection_metric", "S_bal")
            ),
            "last_epoch": int(last_checkpoint["epoch"]),
            "deployment_epoch": int(final_checkpoint["epoch"]),
            # Named for the fixed threshold it was measured at; the metric itself
            # is whichever "selection_metric" above names.
            "best_fixed_threshold_score": (
                float(last_checkpoint["best_score"])
                if math.isfinite(float(last_checkpoint["best_score"]))
                else None
            ),
            "candidate_checkpoints": last_checkpoint["candidate_checkpoints"],
            "deploy_checkpoint": str(run_dir / "deploy.pt"),
        },
        run_dir / "run_summary.json",
    )
    print(f"Training complete. Outputs: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
