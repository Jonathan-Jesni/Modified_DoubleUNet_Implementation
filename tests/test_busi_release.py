"""Compact release checks for the final BUSI workflow."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn as nn

from BUSI_model import aspp_rates_for, build_doubleunet, predict_probabilities
from CBIS_model import build_doubleunet as cbis_build_doubleunet
from busi_dataset import _prior_membership
from busi_runtime import validate_dataset_contract, verify_generated_artifacts
from metrics import BUSIStageLoss, DeepSupervisionLoss, DeterministicCrossEntropy2d
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
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 28_442_362)
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

    def test_aspp_rates_fit_the_feature_map(self):
        # A 3x3 kernel at rate r spans 2r+1 pixels and must fit the map, otherwise
        # its off-centre taps only ever see zero padding and never receive gradient.
        for feature_size in (16, 20, 32, 64):
            for rate in aspp_rates_for(feature_size):
                self.assertLessEqual(2 * rate + 1, feature_size + 2 * rate)
                self.assertLessEqual(rate, max(1, (feature_size - 1) // 2))
        self.assertEqual(aspp_rates_for(16), (2, 3, 5))
        self.assertEqual(aspp_rates_for(32), (3, 6, 9))

    def test_no_dead_aspp_weights(self):
        # Regression: at the legacy rate 18 on a 16x16 map, 8 of the 9 weights in
        # a1.c4 received exactly zero gradient - the branch was a 1x1 convolution
        # wearing a 3x3 costume.
        torch.manual_seed(0)
        model = build_doubleunet(variant="core", pretrained=False, input_size=256)
        p1, p2 = model(torch.randn(2, 3, 256, 256))
        (p1.sum() + p2.sum()).backward()
        for name, aspp in (("a1", model.a1), ("a2", model.a2)):
            for branch in ("c2", "c3", "c4"):
                gradient = getattr(aspp, branch).conv[0].weight.grad
                dead = int((gradient.abs().sum(dim=(0, 1)) == 0).sum())
                self.assertEqual(dead, 0, f"{name}.{branch} has {dead} dead taps")

    def test_single_sample_batch_trains(self):
        # Regression: fit pools of 473/474 at batch_size 8 with drop_last=False end
        # an epoch on a batch of one. A BatchNorm on the 1x1 image-pooling branch
        # would raise "Expected more than 1 value per channel when training".
        torch.manual_seed(0)
        model = build_doubleunet(variant="v2", pretrained=False).train()
        p1, p2 = model(torch.randn(1, 3, 256, 256))
        self.assertEqual(tuple(p1.shape), (1, 3, 256, 256))
        self.assertEqual(tuple(p2.shape), (1, 3, 256, 256))

    def test_cross_entropy_matches_torch_and_is_deterministic(self):
        # nn.CrossEntropyLoss on [B,C,H,W] dispatches to nll_loss2d, whose backward
        # uses atomics. At batch 2 the contention is low enough that it happens to
        # be stable, but at the batch 8 both pipelines actually train with, repeated
        # backward passes over identical inputs disagree in the last few bits.
        # seed_everything already trades away TF32 and cuDNN autotuning to buy
        # reproducibility, so one non-deterministic op meant paying that cost and
        # not receiving the guarantee.
        weights = [0.109, 1.0, 0.959]
        reference = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        candidate = DeterministicCrossEntropy2d(3, weight=weights)

        torch.manual_seed(0)
        logits = torch.randn(4, 3, 64, 64, requires_grad=True)
        targets = torch.randint(0, 3, (4, 64, 64))
        expected = reference(logits, targets)
        expected_grad, = torch.autograd.grad(expected, logits)

        other = logits.detach().clone().requires_grad_(True)
        actual = candidate(other, targets)
        actual_grad, = torch.autograd.grad(actual, other)

        self.assertAlmostEqual(expected.item(), actual.item(), places=5)
        self.assertTrue(torch.allclose(expected_grad, actual_grad, atol=1e-7))

        def gradient():
            torch.manual_seed(1)
            values = torch.randn(8, 3, 128, 128, requires_grad=True)
            labels = torch.randint(0, 3, (8, 128, 128))
            return torch.autograd.grad(candidate(values, labels), values)[0]

        runs = [gradient() for _ in range(4)]
        for run in runs[1:]:
            self.assertTrue(torch.equal(runs[0], run), "loss gradient is not bit-exact")

    def test_cbis_carries_no_dead_xception(self):
        # Regression: features_only=True instantiated the whole Xception (20,806,952
        # parameters) and ran it every step while using only features[0].
        model = cbis_build_doubleunet(input_size=512)
        total = sum(parameter.numel() for parameter in model.parameters())
        self.assertLess(total, 29_000_000, "CBIS still carries the full Xception")
        self.assertEqual(model.aspp_rates, (3, 6, 9))
        # The staged-unfreezing fix that BUSI got in July must be present here too.
        model.set_training_phase(1)
        phase_one = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.set_training_phase(3)
        phase_three = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertLess(phase_one, phase_three)

    def test_tta_averages_the_two_flip_views(self):
        torch.manual_seed(0)
        model = build_doubleunet(variant="v2", pretrained=False).eval()
        images = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            plain = predict_probabilities(model, images, tta=False)[1]
            averaged = predict_probabilities(model, images, tta=True)[1]
            # Recompute the mirrored view independently and check TTA is exactly
            # the mean of the two. Convolutions are not flip-equivariant, so the
            # two views genuinely differ - this is the property that makes the
            # averaging worth doing, and the one worth pinning down.
            mirrored = torch.flip(
                torch.softmax(model(torch.flip(images, dims=(3,)))[1].float(), dim=1),
                dims=(3,),
            )
        self.assertFalse(torch.allclose(plain, mirrored, atol=1e-4))
        self.assertTrue(torch.allclose(averaged, 0.5 * (plain + mirrored), atol=1e-6))
        # Averaging must leave the result a valid probability distribution.
        self.assertTrue(
            torch.allclose(averaged.sum(dim=1), torch.ones(1, 256, 256), atol=1e-5)
        )

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
