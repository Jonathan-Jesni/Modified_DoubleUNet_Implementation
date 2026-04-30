import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from operator import add
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model import build_doubleunet
from utils import create_dir, seeding, calculate_metrics


def load_data(path):
    def get_split_data(split_name):
        images = sorted(glob(os.path.join(path, split_name, "images", "*.png")))
        masks = sorted(glob(os.path.join(path, split_name, "masks", "*.png")))

        if len(images) == 0:
            images = sorted(glob(os.path.join(path, split_name, "images", "*.jpg")))

        if len(images) == 0:
            images = sorted(glob(os.path.join(path, split_name, "images", "*.jpeg")))

        return images, masks

    train_x, train_y = get_split_data("train")
    valid_x, valid_y = get_split_data("val")
    test_x, test_y = get_split_data("test")

    if len(test_x) != len(test_y):
        raise ValueError(f"Test images/masks mismatch: {len(test_x)} images vs {len(test_y)} masks")

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]


def process_mask(y_pred_classes):
    """
    For visualization only:
    0 = background = black
    1 = benign     = gray
    2 = malignant  = white
    """
    y_pred = y_pred_classes[0].detach().cpu().numpy()
    y_pred = (y_pred * 127).astype(np.uint8)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)

    return y_pred


def colorize_mask(mask_classes):
    """
    Color visualization:
    0 = background = black
    1 = benign     = green
    2 = malignant  = red
    """
    h, w = mask_classes.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    color[mask_classes == 1] = (0, 255, 0)
    color[mask_classes == 2] = (0, 0, 255)

    return color


def print_score(metrics_score, num_samples):
    jaccard = metrics_score[0] / num_samples
    f1 = metrics_score[1] / num_samples
    recall = metrics_score[2] / num_samples
    precision = metrics_score[3] / num_samples

    print(
        f"Jaccard: {jaccard:1.4f} - "
        f"F1: {f1:1.4f} - "
        f"Recall: {recall:1.4f} - "
        f"Precision: {precision:1.4f}"
    )


def evaluate(model, save_path, test_x, test_y, size, device):
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0]
    metrics_score_2 = [0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = os.path.basename(x)

        # =========================
        # Image
        # =========================
        image_gray = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise ValueError(f"Failed to read image: {x}")

        image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        image_rgb = cv2.resize(image_rgb, size, interpolation=cv2.INTER_LINEAR)

        save_img = image_rgb.copy()

        image = image_rgb.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        image = torch.from_numpy(image).float().to(device)

        # =========================
        # Mask
        # CBIS masks are already class-index masks:
        # 0 = background
        # 1 = benign
        # 2 = malignant
        # =========================
        mask_raw = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise ValueError(f"Failed to read mask: {y}")

        mask_raw = cv2.resize(mask_raw, size, interpolation=cv2.INTER_NEAREST)
        final_mask = np.clip(mask_raw, 0, 2).astype(np.int64)

        save_mask_gray = (final_mask * 127).astype(np.uint8)
        save_mask_gray = np.expand_dims(save_mask_gray, axis=-1)
        save_mask_gray = np.concatenate([save_mask_gray, save_mask_gray, save_mask_gray], axis=2)

        save_mask_color = colorize_mask(final_mask)

        mask = torch.from_numpy(final_mask).long().unsqueeze(0).to(device)

        with torch.no_grad():
            start_time = time.time()

            y_pred1, y_pred2 = model(image)

            end_time = time.time() - start_time
            time_taken.append(end_time)

            y_pred1_prob = torch.softmax(y_pred1, dim=1)
            y_pred2_prob = torch.softmax(y_pred2, dim=1)

            y_pred1_classes = torch.argmax(y_pred1_prob, dim=1)
            y_pred2_classes = torch.argmax(y_pred2_prob, dim=1)

            mask_flat = mask.squeeze(0)

            score_1 = calculate_metrics(mask_flat, y_pred1_classes)
            metrics_score_1 = list(map(add, metrics_score_1, score_1))

            score_2 = calculate_metrics(mask_flat, y_pred2_classes)
            metrics_score_2 = list(map(add, metrics_score_2, score_2))

            y_pred1_img = process_mask(y_pred1_classes)
            y_pred2_img = process_mask(y_pred2_classes)

            y_pred1_color = colorize_mask(y_pred1_classes[0].detach().cpu().numpy())
            y_pred2_color = colorize_mask(y_pred2_classes[0].detach().cpu().numpy())

        line = np.ones((size[1], 10, 3), dtype=np.uint8) * 255

        joint_gray = np.concatenate(
            [save_img, line, save_mask_gray, line, y_pred1_img, line, y_pred2_img],
            axis=1
        )

        joint_color = np.concatenate(
            [save_img, line, save_mask_color, line, y_pred1_color, line, y_pred2_color],
            axis=1
        )

        cv2.imwrite(f"{save_path}/joint_gray/{name}", joint_gray)
        cv2.imwrite(f"{save_path}/joint_color/{name}", joint_color)
        cv2.imwrite(f"{save_path}/mask1/{name}", y_pred1_img)
        cv2.imwrite(f"{save_path}/mask2/{name}", y_pred2_img)
        cv2.imwrite(f"{save_path}/mask1_color/{name}", y_pred1_color)
        cv2.imwrite(f"{save_path}/mask2_color/{name}", y_pred2_color)

    print("--- Output 1 Scores ---")
    print_score(metrics_score_1, len(test_x))

    print("--- Output 2 Scores (Final Output) ---")
    print_score(metrics_score_2, len(test_x))

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1 / mean_time_taken

    print("Mean FPS:", mean_fps)


if __name__ == "__main__":
    seeding(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = build_doubleunet()
    model = model.to(device)

    checkpoint_path = "files/checkpoint.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    path = "dataset_seg"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    print(f"Test images: {len(test_x)}")
    print(f"Test masks : {len(test_y)}")

    save_path = "results_CBIS"

    for item in [
        "mask1",
        "mask2",
        "mask1_color",
        "mask2_color",
        "joint_gray",
        "joint_color",
    ]:
        create_dir(f"{save_path}/{item}")

    size = (256, 256)

    evaluate(model, save_path, test_x, test_y, size, device)