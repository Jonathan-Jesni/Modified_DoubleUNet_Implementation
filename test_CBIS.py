import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from glob import glob
from operator import add

import cv2
import numpy as np
import torch
from tqdm import tqdm

from CBIS_model import build_doubleunet
from utils import create_dir, seeding


NUM_CLASSES = 3
SIZE = (256, 256)
DATASET_PATH = "dataset_seg"
CHECKPOINT_PATH = "files/checkpoint.pth"
SAVE_PATH = "results_CBIS"


def load_data(path):
    def get_split_data(split_name):
        img_dir = os.path.join(path, split_name, "images")
        mask_dir = os.path.join(path, split_name, "masks")

        images = sorted(glob(os.path.join(img_dir, "*.png")))
        masks = sorted(glob(os.path.join(mask_dir, "*.png")))

        image_dict = {os.path.basename(x): x for x in images}
        mask_dict = {os.path.basename(y): y for y in masks}

        common_names = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

        images = [image_dict[name] for name in common_names]
        masks = [mask_dict[name] for name in common_names]

        return images, masks

    train_x, train_y = get_split_data("train")
    valid_x, valid_y = get_split_data("val")
    test_x, test_y = get_split_data("test")

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]


def calculate_multiclass_metrics(y_true, y_pred, num_classes=3):
    """
    Macro metrics over foreground classes only:
      class 1 = benign
      class 2 = malignant

    Background is excluded from averaged score.
    """
    y_true = y_true.detach().cpu().numpy().astype(np.uint8)
    y_pred = y_pred.detach().cpu().numpy().astype(np.uint8)

    jaccards = []
    dices = []
    recalls = []
    precisions = []

    for cls in range(1, num_classes):
        true_cls = (y_true == cls)
        pred_cls = (y_pred == cls)

        tp = np.logical_and(true_cls, pred_cls).sum()
        fp = np.logical_and(~true_cls, pred_cls).sum()
        fn = np.logical_and(true_cls, ~pred_cls).sum()

        if true_cls.sum() == 0 and pred_cls.sum() == 0:
            continue

        jaccard = tp / (tp + fp + fn + 1e-7)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)

        jaccards.append(jaccard)
        dices.append(dice)
        recalls.append(recall)
        precisions.append(precision)

    if len(jaccards) == 0:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        float(np.mean(jaccards)),
        float(np.mean(dices)),
        float(np.mean(recalls)),
        float(np.mean(precisions)),
    ]


def colorize_mask(mask_classes):
    h, w = mask_classes.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    color[mask_classes == 1] = (0, 255, 0)      # benign = green
    color[mask_classes == 2] = (0, 0, 255)      # malignant = red

    return color


def gray_mask(mask_classes):
    out = (mask_classes * 127).astype(np.uint8)
    out = np.expand_dims(out, axis=-1)
    out = np.concatenate([out, out, out], axis=2)
    return out


def make_overlay(image_rgb, mask_classes, alpha=0.45):
    mask_color = colorize_mask(mask_classes)
    overlay = image_rgb.copy()

    lesion = mask_classes > 0

    overlay[lesion] = (
        image_rgb[lesion].astype(np.float32) * (1 - alpha)
        + mask_color[lesion].astype(np.float32) * alpha
    ).astype(np.uint8)

    return overlay


def print_score(metrics_score, num_samples):
    jaccard = metrics_score[0] / num_samples
    f1 = metrics_score[1] / num_samples
    recall = metrics_score[2] / num_samples
    precision = metrics_score[3] / num_samples

    print(
        f"Jaccard: {jaccard:1.4f} - "
        f"F1/Dice: {f1:1.4f} - "
        f"Recall: {recall:1.4f} - "
        f"Precision: {precision:1.4f}"
    )


def evaluate(model, save_path, test_x, test_y, size, device):
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0]
    metrics_score_2 = [0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = os.path.basename(x)

        image_gray = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise ValueError(f"Failed to read image: {x}")

        image_gray = cv2.resize(image_gray, size, interpolation=cv2.INTER_LINEAR)
        image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        save_img = image_rgb.copy()

        image = image_rgb.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float().to(device)

        mask_raw = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise ValueError(f"Failed to read mask: {y}")

        mask_raw = cv2.resize(mask_raw, size, interpolation=cv2.INTER_NEAREST)
        final_mask = np.clip(mask_raw, 0, 2).astype(np.uint8)

        mask = torch.from_numpy(final_mask).long().to(device)

        with torch.no_grad():
            start_time = time.time()
            y_pred1, y_pred2 = model(image)
            end_time = time.time() - start_time
            time_taken.append(end_time)

            y_pred1_classes = torch.argmax(torch.softmax(y_pred1, dim=1), dim=1)[0]
            y_pred2_classes = torch.argmax(torch.softmax(y_pred2, dim=1), dim=1)[0]

            score_1 = calculate_multiclass_metrics(mask, y_pred1_classes, NUM_CLASSES)
            score_2 = calculate_multiclass_metrics(mask, y_pred2_classes, NUM_CLASSES)

            metrics_score_1 = list(map(add, metrics_score_1, score_1))
            metrics_score_2 = list(map(add, metrics_score_2, score_2))

            y_pred1_np = y_pred1_classes.detach().cpu().numpy().astype(np.uint8)
            y_pred2_np = y_pred2_classes.detach().cpu().numpy().astype(np.uint8)

        save_mask_gray = gray_mask(final_mask)
        y_pred1_gray = gray_mask(y_pred1_np)
        y_pred2_gray = gray_mask(y_pred2_np)

        save_mask_color = colorize_mask(final_mask)
        y_pred1_color = colorize_mask(y_pred1_np)
        y_pred2_color = colorize_mask(y_pred2_np)

        gt_overlay = make_overlay(save_img, final_mask)
        pred_overlay = make_overlay(save_img, y_pred2_np)

        line = np.ones((size[1], 10, 3), dtype=np.uint8) * 255

        joint_gray = np.concatenate(
            [save_img, line, save_mask_gray, line, y_pred1_gray, line, y_pred2_gray],
            axis=1
        )

        joint_color = np.concatenate(
            [save_img, line, save_mask_color, line, y_pred1_color, line, y_pred2_color],
            axis=1
        )

        joint_overlay = np.concatenate(
            [save_img, line, gt_overlay, line, pred_overlay],
            axis=1
        )

        cv2.imwrite(f"{save_path}/joint_gray/{name}", joint_gray)
        cv2.imwrite(f"{save_path}/joint_color/{name}", joint_color)
        cv2.imwrite(f"{save_path}/joint_overlay/{name}", joint_overlay)

        cv2.imwrite(f"{save_path}/gt_overlay/{name}", gt_overlay)
        cv2.imwrite(f"{save_path}/pred_overlay/{name}", pred_overlay)

        cv2.imwrite(f"{save_path}/mask1/{name}", y_pred1_gray)
        cv2.imwrite(f"{save_path}/mask2/{name}", y_pred2_gray)
        cv2.imwrite(f"{save_path}/mask1_color/{name}", y_pred1_color)
        cv2.imwrite(f"{save_path}/mask2_color/{name}", y_pred2_color)

    print("--- Output 1 Scores ---")
    print_score(metrics_score_1, len(test_x))

    print("--- Output 2 Scores / Final Output ---")
    print_score(metrics_score_2, len(test_x))

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1 / mean_time_taken

    print(f"Mean FPS: {mean_fps:1.2f}")


if __name__ == "__main__":
    seeding(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = build_doubleunet()
    model = model.to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(DATASET_PATH)

    print(f"Train images: {len(train_x)}")
    print(f"Val images  : {len(valid_x)}")
    print(f"Test images : {len(test_x)}")
    print(f"Test masks  : {len(test_y)}")

    if len(test_x) == 0:
        raise RuntimeError("No test images found. Run CBIS_prepare.py first.")

    for item in [
        "mask1",
        "mask2",
        "mask1_color",
        "mask2_color",
        "joint_gray",
        "joint_color",
        "joint_overlay",
        "gt_overlay",
        "pred_overlay",
    ]:
        create_dir(f"{SAVE_PATH}/{item}")

    evaluate(model, SAVE_PATH, test_x, test_y, SIZE, device)