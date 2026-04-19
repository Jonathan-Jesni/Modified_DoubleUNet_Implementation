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
from model import build_doubleunet
from metrics import DiceBCELoss


def load_data(path):
    """
    Expected dataset structure:
    dataset_seg/
        train/
            images/*.png
            masks/*.png
        val/
            images/*.png
            masks/*.png
        test/
            images/*.png
            masks/*.png
    """

    def get_split_data(split_name):
        images = sorted(glob(os.path.join(path, split_name, "images", "*.png")))
        masks = sorted(glob(os.path.join(path, split_name, "masks", "*.png")))
        return images, masks

    train_x, train_y = get_split_data("train")
    valid_x, valid_y = get_split_data("val")
    test_x, test_y = get_split_data("test")

    if len(train_x) == 0:
        raise FileNotFoundError(f"No training images found in: {os.path.join(path, 'train', 'images')}")
    if len(valid_x) == 0:
        raise FileNotFoundError(f"No validation images found in: {os.path.join(path, 'val', 'images')}")
    if len(test_x) == 0:
        print("[INFO] No test images found. Test split will remain empty for now.")

    if not (len(train_x) == len(train_y)):
        raise ValueError(f"Train images/masks mismatch: {len(train_x)} images vs {len(train_y)} masks")
    if not (len(valid_x) == len(valid_y)):
        raise ValueError(f"Val images/masks mismatch: {len(valid_x)} images vs {len(valid_y)} masks")
    if len(test_x) != len(test_y):
        raise ValueError(f"Test images/masks mismatch: {len(test_x)} images vs {len(test_y)} masks")

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

        # Read grayscale ultrasound image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Repeat grayscale into 3 identical channels for pretrained encoder
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Read mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        # Resize first
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        # Augment
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        # Binarize mask and shape to [1, H, W]
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

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

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        p1, p2 = model(x)
        loss = loss_fn(p1, y) + loss_fn(p2, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y, p2):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss / len(loader)
    epoch_jac = epoch_jac / len(loader)
    epoch_f1 = epoch_f1 / len(loader)
    epoch_recall = epoch_recall / len(loader)
    epoch_precision = epoch_precision / len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            p1, p2 = model(x)
            loss = loss_fn(p1, y) + loss_fn(p2, y)
            epoch_loss += loss.item()

            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y, p2):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss / len(loader)
    epoch_jac = epoch_jac / len(loader)
    epoch_f1 = epoch_f1 / len(loader)
    epoch_recall = epoch_recall / len(loader)
    epoch_precision = epoch_precision / len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


if __name__ == "__main__":
    # Seeding
    seeding(42)

    # Directories
    create_dir("files")

    # Training logfile
    train_log_path = "files/train_log.txt"
    if not os.path.exists(train_log_path):
        with open(train_log_path, "w") as f:
            f.write("\n")

    # Record date & time
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    # Hyperparameters
    image_size = 256
    size = (image_size, image_size)
    batch_size = 8
    num_epochs = 300
    lr = 1e-4
    early_stopping_patience = 50
    checkpoint_path = "files/checkpoint.pth"
    path = "dataset_seg"

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    data_str += f"Dataset Path: {path}\n"
    print_and_save(train_log_path, data_str)

    # Dataset
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    train_x, train_y = shuffling(train_x, train_y)

    data_str = (
        f"Dataset Size:\n"
        f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}\n"
    )
    print_and_save(train_log_path, data_str)

    # Data augmentation
    transform = A.Compose([
        A.Rotate(limit=20, p=0.3, border_mode=cv2.BORDER_REFLECT_101),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.CoarseDropout(
            max_holes=8,
            max_height=24,
            max_width=24,
            p=0.2
        ),
    ])

    # Dataset and loader
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_and_save(train_log_path, f"Device: {device}\n")

    model = build_doubleunet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, verbose=True
    )
    loss_fn = DiceBCELoss()
    loss_name = "BCE + Dice Loss"

    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    # Training
    best_valid_f1 = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[1] > best_valid_f1:
            data_str = (
                f"Valid F1 improved from {best_valid_f1:2.4f} to {valid_metrics[1]:2.4f}. "
                f"Saving checkpoint: {checkpoint_path}"
            )
            print_and_save(train_log_path, data_str)

            best_valid_f1 = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

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
            data_str = (
                f"Early stopping: validation F1 did not improve for "
                f"{early_stopping_patience} consecutive epochs.\n"
            )
            print_and_save(train_log_path, data_str)
            break