import os
import time
import datetime
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import (
    seeding,
    create_dir,
    print_and_save,
    shuffling,
    epoch_time,
    calculate_metrics,
)
from CBIS_model import build_doubleunet
from metrics import CombinedLoss


DEBUG_VIS_DIR = "files/debug_train_visuals"
SAVE_DEBUG_EVERY_N_SAMPLES = 100

NUM_CLASSES = 3


def colorize_mask(mask):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color[mask == 1] = (0, 255, 0)      # benign = green
    color[mask == 2] = (0, 0, 255)      # malignant = red
    return color


def save_debug_visual(image_rgb, mask, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    image_vis = image_rgb.copy()
    if image_vis.max() <= 1:
        image_vis = (image_vis * 255).astype(np.uint8)

    mask_color = colorize_mask(mask)

    overlay = image_vis.copy()
    lesion = mask > 0

    overlay[lesion] = (
        image_vis[lesion].astype(np.float32) * 0.55
        + mask_color[lesion].astype(np.float32) * 0.45
    ).astype(np.uint8)

    panel = np.concatenate([image_vis, mask_color, overlay], axis=1)
    cv2.imwrite(out_path, panel)


def load_data(path):
    def get_split_data(split_name):
        img_dir = os.path.join(path, split_name, "images")
        mask_dir = os.path.join(path, split_name, "masks")

        images = sorted(glob(os.path.join(img_dir, "*.png")))
        masks = sorted(glob(os.path.join(mask_dir, "*.png")))

        image_dict = {os.path.basename(x): x for x in images}
        mask_dict = {os.path.basename(y): y for y in masks}

        common_names = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

        paired_images = [image_dict[name] for name in common_names]
        paired_masks = [mask_dict[name] for name in common_names]

        missing_masks = sorted(set(image_dict.keys()) - set(mask_dict.keys()))
        missing_images = sorted(set(mask_dict.keys()) - set(image_dict.keys()))

        if missing_masks:
            print(f"[WARN] {split_name}: {len(missing_masks)} images have no matching mask.")
        if missing_images:
            print(f"[WARN] {split_name}: {len(missing_images)} masks have no matching image.")

        return paired_images, paired_masks

    train_x, train_y = get_split_data("train")
    valid_x, valid_y = get_split_data("val")
    test_x, test_y = get_split_data("test")

    if len(train_x) == 0:
        raise FileNotFoundError(f"No training image/mask pairs found in: {os.path.join(path, 'train')}")
    if len(valid_x) == 0:
        raise FileNotFoundError(f"No validation image/mask pairs found in: {os.path.join(path, 'val')}")
    if len(test_x) == 0:
        print("[INFO] No test image/mask pairs found. Test split will remain empty for now.")

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        image_path = self.images_path[index]
        mask_path = self.masks_path[index]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        # keep class IDs exactly as prepared:
        # 0 = background, 1 = benign, 2 = malignant
        mask = np.clip(mask, 0, NUM_CLASSES - 1).astype(np.uint8)

        if index % SAVE_DEBUG_EVERY_N_SAMPLES == 0:
            out_name = os.path.basename(image_path)
            save_debug_visual(
                image.copy(),
                mask.copy(),
                os.path.join(DEBUG_VIS_DIR, out_name)
            )

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

            mask = np.clip(mask, 0, NUM_CLASSES - 1).astype(np.uint8)

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).float()

        mask = mask.astype(np.int64)
        mask = torch.from_numpy(mask).long()

        return image, mask

    def __len__(self):
        return self.n_samples


def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    use_cuda_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_cuda_amp):
            p1, p2 = model(x)
            loss = loss_fn(p1, y) + loss_fn(p2, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        p2_classes = torch.argmax(p2, dim=1)
        score = calculate_metrics(y, p2_classes)

        epoch_jac += score[0]
        epoch_f1 += score[1]
        epoch_recall += score[2]
        epoch_precision += score[3]

    return (
        epoch_loss / len(loader),
        [
            epoch_jac / len(loader),
            epoch_f1 / len(loader),
            epoch_recall / len(loader),
            epoch_precision / len(loader),
        ],
    )


def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    use_cuda_amp = device.type == "cuda"

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                p1, p2 = model(x)
                loss = loss_fn(p1, y) + loss_fn(p2, y)

            epoch_loss += loss.item()

            p2_classes = torch.argmax(p2, dim=1)
            score = calculate_metrics(y, p2_classes)

            epoch_jac += score[0]
            epoch_f1 += score[1]
            epoch_recall += score[2]
            epoch_precision += score[3]

    return (
        epoch_loss / len(loader),
        [
            epoch_jac / len(loader),
            epoch_f1 / len(loader),
            epoch_recall / len(loader),
            epoch_precision / len(loader),
        ],
    )


if __name__ == "__main__":
    seeding(42)

    create_dir("files")
    create_dir(DEBUG_VIS_DIR)

    train_log_path = "files/train_log.txt"
    if not os.path.exists(train_log_path):
        with open(train_log_path, "w") as f:
            f.write("\n")

    print_and_save(train_log_path, str(datetime.datetime.now()))
    print("")

    image_size = 256
    size = (image_size, image_size)

    batch_size = 8
    num_epochs = 300
    lr = 1e-4
    early_stopping_patience = 50

    checkpoint_path = "files/CBIS_checkpoint.pth"
    path = "dataset_seg"

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    data_str += f"Dataset Path: {path}\n"
    data_str += f"Classes: {NUM_CLASSES} -> 0 background, 1 benign, 2 malignant\n"
    print_and_save(train_log_path, data_str)

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    train_x, train_y = shuffling(train_x, train_y)

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}\n"
    print_and_save(train_log_path, data_str)

    transform = A.Compose([
        A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_REFLECT_101),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(1, 20),
            hole_width_range=(1, 20),
            p=0.15
        ),
    ])

    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    num_workers = 4
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print_and_save(train_log_path, f"Device: {device}\n")

    model = build_doubleunet()
    model = model.to(device)

    if os.path.exists(checkpoint_path):
        print(f"--- Found existing checkpoint. Resuming from {checkpoint_path} ---")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5
    )

    loss_fn = CombinedLoss(num_classes=NUM_CLASSES)
    loss_name = "CrossEntropy + Multi-Class Dice Loss"

    print_and_save(train_log_path, f"Optimizer: Adam\nLoss: {loss_name}\n")

    best_valid_f1 = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)

        scheduler.step(valid_loss)

        if valid_metrics[1] > best_valid_f1:
            data_str = (
                f"Valid F1 improved from {best_valid_f1:2.4f} "
                f"to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            )
            print_and_save(train_log_path, data_str)

            best_valid_f1 = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        epoch_mins, epoch_secs = epoch_time(start_time, time.time())

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += (
            f"\tTrain Loss: {train_loss:.4f} - "
            f"Jaccard: {train_metrics[0]:.4f} - "
            f"F1: {train_metrics[1]:.4f} - "
            f"Recall: {train_metrics[2]:.4f} - "
            f"Precision: {train_metrics[3]:.4f}\n"
        )
        data_str += (
            f"\t Val. Loss: {valid_loss:.4f} - "
            f"Jaccard: {valid_metrics[0]:.4f} - "
            f"F1: {valid_metrics[1]:.4f} - "
            f"Recall: {valid_metrics[2]:.4f} - "
            f"Precision: {valid_metrics[3]:.4f}\n"
        )

        print_and_save(train_log_path, data_str)

        if early_stopping_count >= early_stopping_patience:
            print_and_save(
                train_log_path,
                f"Early stopping: validation F1 did not improve for {early_stopping_patience} consecutive epochs.\n"
            )
            break