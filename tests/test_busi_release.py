"""Compact release checks for the final BUSI workflow."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from BUSI_model import build_doubleunet
from busi_dataset import _prior_membership
from busi_runtime import validate_dataset_contract, verify_generated_artifacts
from metrics import BUSIStageLoss, DeepSupervisionLoss
from prepare_busi_dataset import resolve_review_csv
from train_BUSI import resolve_config


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "dataset_seg_BUSI_v2" / "manifest.csv"


class BUSIReleaseTests(unittest.TestCase):
    def test_final_default_recipe(self):
        config = resolve_config(None, "v2", 0, 42)
        self.assertEqual(config["variant"], "v2")
        self.assertEqual(
            config["dataset"]["preprocessing_profile"],
            "padded_256_imagenet",
        )
        self.assertEqual(
            config["dataset"]["augmentation"],
            "conservative_ultrasound",
        )
        self.assertEqual(config["loss"]["mode"], "composite")

    def test_v2_forward_and_loss(self):
        torch.manual_seed(7)
        model = build_doubleunet(
            variant="v2",
            num_classes=3,
            preprocessing_profile="padded_256_imagenet",
            input_size=256,
            pretrained=False,
            bn_policy="targeted",
        ).eval()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 28_442_490)
        image = torch.randn(1, 3, 256, 256)
        target = torch.zeros(1, 256, 256, dtype=torch.long)
        target[:, 80:150, 90:170] = 1
        with torch.no_grad():
            p1, p2 = model(image)
            loss = DeepSupervisionLoss(
                BUSIStageLoss(
                    class_weights=[0.1, 1.0, 1.0],
                    mode="composite",
                )
            )(p1, p2, target)
        self.assertEqual(tuple(p1.shape), (1, 3, 256, 256))
        self.assertEqual(tuple(p2.shape), (1, 3, 256, 256))
        self.assertTrue(torch.isfinite(loss).item())

    @unittest.skipUnless(MANIFEST.is_file(), "canonical BUSI dataset not present")
    def test_frozen_dataset_and_regeneration_provenance(self):
        metadata = validate_dataset_contract(MANIFEST)
        artifacts = verify_generated_artifacts(MANIFEST)
        self.assertEqual(metadata["counts"]["eligible"], 778)
        self.assertEqual(artifacts["artifact_file_count"], 1_556)
        prior = _prior_membership(None, MANIFEST)
        self.assertEqual(len(prior), 646)
        prior_test = sum(
            "test" in {split for split, _ in memberships}
            for memberships in prior.values()
        )
        self.assertEqual(prior_test, 65)
        review_csv = resolve_review_csv(None, MANIFEST)
        self.assertIsNotNone(review_csv)
        self.assertTrue(review_csv.is_file())


if __name__ == "__main__":
    unittest.main()
