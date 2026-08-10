"""Build the immutable, content-grouped BUSI v2 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from busi_dataset import GenerationConfig, generate_busi_dataset


ROOT = Path(__file__).resolve().parent


def resolve_review_csv(review_csv: Path | None, prior_manifest: Path | None) -> Path | None:
    if review_csv is not None:
        return review_csv
    if prior_manifest is None:
        return None
    embedded = prior_manifest.parent / "review" / "completed_two_reviewer_review.csv"
    return embedded if embedded.is_file() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a non-destructive, content-grouped BUSI v2 dataset."
    )
    parser.add_argument("--raw-root", type=Path, default=ROOT / "BUSI_dataset")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "dataset_seg_BUSI_v2"
    )
    parser.add_argument(
        "--prior-dataset-root",
        type=Path,
        default=ROOT / "dataset_seg_BUSI",
        help="Existing prepared dataset used only to recover prior split membership.",
    )
    parser.add_argument(
        "--prior-manifest",
        type=Path,
        default=ROOT / "dataset_seg_BUSI_v2" / "manifest.csv",
        help=(
            "Canonical manifest used to recover historical split membership. "
            "It takes precedence over --prior-dataset-root when present."
        ),
    )
    parser.add_argument(
        "--include-normal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include normal all-background examples (default: true).",
    )
    parser.add_argument("--split-seed", type=int, default=20260717)
    parser.add_argument("--sealed-fraction", type=float, default=0.15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--calibration-fraction", type=float, default=0.125)
    parser.add_argument(
        "--expected-prior-test-count",
        type=int,
        default=65,
        help="Fail unless exactly this many previously inspected test images are recovered.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=None,
        help=(
            "Optional completed two-reviewer near-duplicate CSV. Pending or "
            "disputed pairs remain conservatively grouped."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    review_csv = resolve_review_csv(args.review_csv, args.prior_manifest)
    summary = generate_busi_dataset(GenerationConfig(
        raw_root=args.raw_root,
        output_root=args.output_root,
        prior_dataset_root=args.prior_dataset_root,
        prior_manifest=args.prior_manifest,
        include_normal=args.include_normal,
        split_seed=args.split_seed,
        sealed_fraction=args.sealed_fraction,
        folds=args.folds,
        calibration_fraction=args.calibration_fraction,
        expected_prior_test_count=args.expected_prior_test_count,
        review_csv=review_csv,
    ))
    print(f"Generated BUSI v2 at: {args.output_root.resolve()}")
    print(f"Eligible samples: {summary['eligible_samples']}")
    print(f"Development: {summary['development_samples']}")
    print(f"Sealed: {summary['sealed_samples']}")
    print(f"Quarantined/excluded: {summary['excluded_samples']}")
    print(f"Near-duplicate candidates: {summary['near_duplicate_candidates']}")
    print(f"Dataset fingerprint: {summary['dataset_fingerprint']}")


if __name__ == "__main__":
    main()
