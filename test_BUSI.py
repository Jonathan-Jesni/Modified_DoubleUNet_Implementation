import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from operator import add
import numpy as np
import cv2
from tqdm import tqdm
import torch

from BUSI_model import build_doubleunet
from utils import create_dir, seeding, calculate_metrics
from train_BUSI import load_data


COLOR_MAP = {
    0: (0, 0, 0),       # background = black
    1: (0, 255, 0),     # benign = green
    2: (0, 0, 255),     # malignant = red
}


def class_to_gray(mask):
    mask = (mask * 127).astype(np.uint8)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=2)
    return mask


def class_to_color(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id, color in COLOR_MAP.items():
        color_mask[mask == class_id] = color

    return color_mask


def print_score(metrics_score, num_samples):
    jaccard = metrics_score[0] / num_samples
    f1 = metrics_score[1] / num_samples
    recall = metrics_score[2] / num_samples
    precision = metrics_score[3] / num_samples

    print(
        f"Jaccard/IoU: {jaccard:1.4f} - "
        f"Dice/F1: {f1:1.4f} - "
        f"Recall: {recall:1.4f} - "
        f"Precision: {precision:1.4f}"
    )


def dice_per_class(pred, target, class_id, eps=1e-7):
    pred_c = pred == class_id
    target_c = target == class_id

    intersection = np.logical_and(pred_c, target_c).sum()
    total = pred_c.sum() + target_c.sum()

    if total == 0:
        return np.nan

    return (2 * intersection + eps) / (total + eps)


def iou_per_class(pred, target, class_id, eps=1e-7):
    pred_c = pred == class_id
    target_c = target == class_id

    intersection = np.logical_and(pred_c, target_c).sum()
    union = np.logical_or(pred_c, target_c).sum()

    if union == 0:
        return np.nan

    return (intersection + eps) / (union + eps)


def evaluate(model, save_path, test_x, test_y, size, device):
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0]
    metrics_score_2 = [0.0, 0.0, 0.0, 0.0]

    class_dice = {0: [], 1: [], 2: []}
    class_iou = {0: [], 1: [], 2: []}

    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = os.path.basename(x)

        # ---------------- Image ----------------
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)

        save_img = image.copy()

        image_input = np.transpose(image, (2, 0, 1))
        image_input = np.expand_dims(image_input, axis=0)
        image_input = image_input / 255.0
        image_input = image_input.astype(np.float32)

        image_tensor = torch.from_numpy(image_input).to(device)

        # ---------------- Mask ----------------
        mask_raw = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.resize(mask_raw, size, interpolation=cv2.INTER_NEAREST)

        final_mask = np.zeros(mask_raw.shape, dtype=np.int64)
        filename = os.path.basename(y).lower()

        tumor_pixels = mask_raw > 127

        if "benign" in filename:
            final_mask[tumor_pixels] = 1
        elif "malignant" in filename:
            final_mask[tumor_pixels] = 2
        else:
            final_mask[tumor_pixels] = 0

        save_mask_gray = class_to_gray(final_mask)
        save_mask_color = class_to_color(final_mask)

        mask_tensor = torch.from_numpy(final_mask).long().unsqueeze(0).to(device)

        with torch.no_grad():
            start_time = time.time()

            y_pred1, y_pred2 = model(image_tensor)

            end_time = time.time() - start_time
            time_taken.append(end_time)

            y_pred1 = torch.softmax(y_pred1, dim=1)
            y_pred2 = torch.softmax(y_pred2, dim=1)

            y_pred1_classes = torch.argmax(y_pred1, dim=1)
            y_pred2_classes = torch.argmax(y_pred2, dim=1)

            mask_flat = mask_tensor.squeeze(0)

            score_1 = calculate_metrics(mask_flat, y_pred1_classes)
            score_2 = calculate_metrics(mask_flat, y_pred2_classes)

            metrics_score_1 = list(map(add, metrics_score_1, score_1))
            metrics_score_2 = list(map(add, metrics_score_2, score_2))

            pred1_np = y_pred1_classes[0].cpu().numpy().astype(np.uint8)
            pred2_np = y_pred2_classes[0].cpu().numpy().astype(np.uint8)

            for class_id in [0, 1, 2]:
                d = dice_per_class(pred2_np, final_mask, class_id)
                j = iou_per_class(pred2_np, final_mask, class_id)

                if not np.isnan(d):
                    class_dice[class_id].append(d)
                if not np.isnan(j):
                    class_iou[class_id].append(j)

            pred1_gray = class_to_gray(pred1_np)
            pred2_gray = class_to_gray(pred2_np)

            pred1_color = class_to_color(pred1_np)
            pred2_color = class_to_color(pred2_np)

        # ---------------- Overlay ----------------
        overlay = cv2.addWeighted(save_img, 0.65, pred2_color, 0.35, 0)

        # ---------------- Save outputs ----------------
        line = np.ones((size[1], 10, 3), dtype=np.uint8) * 255

        joint = np.concatenate(
            [
                save_img,
                line,
                save_mask_color,
                line,
                pred1_color,
                line,
                pred2_color,
                line,
                overlay,
            ],
            axis=1,
        )

        cv2.imwrite(f"{save_path}/joint/{name}", joint)
        cv2.imwrite(f"{save_path}/mask1_gray/{name}", pred1_gray)
        cv2.imwrite(f"{save_path}/mask2_gray/{name}", pred2_gray)
        cv2.imwrite(f"{save_path}/mask1_color/{name}", pred1_color)
        cv2.imwrite(f"{save_path}/mask2_color/{name}", pred2_color)
        cv2.imwrite(f"{save_path}/overlays/{name}", overlay)

    print("\n--- Output 1 Scores ---")
    print_score(metrics_score_1, len(test_x))

    print("\n--- Output 2 Scores Final Output ---")
    print_score(metrics_score_2, len(test_x))

    print("\n--- Per-Class Final Output Scores ---")

    class_names = {
        0: "Background",
        1: "Benign lesion",
        2: "Malignant lesion",
    }

    for class_id, class_name in class_names.items():
        mean_dice = np.mean(class_dice[class_id]) if class_dice[class_id] else 0.0
        mean_iou = np.mean(class_iou[class_id]) if class_iou[class_id] else 0.0

        print(f"{class_name}: Dice = {mean_dice:.4f}, IoU = {mean_iou:.4f}")

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1 / mean_time_taken

    print(f"\nMean FPS: {mean_fps:.2f}")
    print(f"Results saved at: {save_path}")


if __name__ == "__main__":
    seeding(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_doubleunet()
    model = model.to(device)

    checkpoint_path = "files/BUSI_checkpoint.pth.zip"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    path = "dataset_seg_BUSI"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    save_path = "results_BUSI"

    for item in [
        "mask1_gray",
        "mask2_gray",
        "mask1_color",
        "mask2_color",
        "overlays",
        "joint",
    ]:
        create_dir(f"{save_path}/{item}")

    size = (256, 256)

    evaluate(model, save_path, test_x, test_y, size, device)