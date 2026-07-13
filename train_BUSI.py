import os
import time
import datetime
from glob import glob

# NOTE: import torch BEFORE albumentations. On Windows, importing albumentations
# first loads a runtime DLL that breaks torch's c10.dll init (OSError WinError 1114).
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
import cv2
import numpy as np

from utils import (
    seeding,
    create_dir,
    print_and_save,
    shuffling,
    epoch_time,
    calculate_metrics,
    calculate_foreground_metrics,
)
from BUSI_model import build_doubleunet
from metrics import DiceBCELoss, MultiClassDiceLoss, CombinedLoss, BUSI_CLASS_WEIGHTS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def load_data(path):
    """
    Expected dataset structure:
    dataset_seg_BUSI/
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

            # --- PYTORCH IMAGE FORMATTING ---
            image = np.transpose(image, (2, 0, 1))  # Convert from HWC to CHW
            image = image / 255.0                   # Scale pixels to 0-1
            image = (image - IMAGENET_MEAN) / IMAGENET_STD
            image = torch.from_numpy(image).float() # Convert to PyTorch Tensor

            # --- MULTI-CLASS PIXEL MAPPING ---
            final_mask = np.zeros(mask.shape, dtype=np.int64)
            filename = os.path.basename(mask_path).lower()
            tumor_pixels = (mask > 127)

            if "benign" in filename:
                final_mask[tumor_pixels] = 1 
            elif "malignant" in filename:
                final_mask[tumor_pixels] = 2
            mask = torch.from_numpy(final_mask).long()
            return image, mask

    def __len__(self):
        return self.n_samples


def train(model, loader, optimizer, loss_fn, device, scaler, frozen_backbone_modules=None):
    model.train()

    # Keep the frozen backbone submodules pinned in eval mode for BatchNorm
    # purposes, even though model.train() (above) just recursively set
    # everything - including them - back to train mode. requires_grad=False
    # only stops weight *gradient* updates; it does NOT stop BatchNorm's
    # running_mean/running_var from being recalculated off live batch
    # statistics every epoch. Left unpinned, the "frozen" backbones drift away
    # from their pretrained calibration using small, noisy batch=8 statistics
    # that the (frozen) conv weights can never adapt to compensate for. This
    # must be re-applied every epoch since train() is called once per epoch.
    if frozen_backbone_modules:
        for module in frozen_backbone_modules:
            module.eval()

    epoch_loss = 0.0
    metrics_bg = [0.0, 0.0, 0.0, 0.0]   # background-inclusive (logging only)
    metrics_fg = [0.0, 0.0, 0.0, 0.0]   # foreground-only (drives checkpoint/early-stop)

    # SPEED PATCH 1: Use provided AMP Scaler

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)

        # SPEED PATCH 2: set_to_none=True clears gradients faster
        optimizer.zero_grad(set_to_none=True)

        # SPEED PATCH 3: autocast for 16-bit math
        with torch.amp.autocast('cuda'):
            p1, p2 = model(x)
            loss = 0.4 * loss_fn(p1, y) + 1.0 * loss_fn(p2, y)

        # SPEED PATCH 4: Scaled backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        p2_classes = torch.argmax(p2, dim=1)
        score_bg = calculate_metrics(y, p2_classes)
        score_fg = calculate_foreground_metrics(y, p2_classes)
        metrics_bg = [a + b for a, b in zip(metrics_bg, score_bg)]
        metrics_fg = [a + b for a, b in zip(metrics_fg, score_fg)]

    n = len(loader)
    return epoch_loss / n, [m / n for m in metrics_bg], [m / n for m in metrics_fg]


def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    metrics_bg = [0.0, 0.0, 0.0, 0.0]   # background-inclusive (logging only)
    metrics_fg = [0.0, 0.0, 0.0, 0.0]   # foreground-only (drives checkpoint/early-stop)

    with torch.no_grad():
        for x, y in loader:
            with torch.amp.autocast('cuda'):
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.long)
                p1, p2 = model(x)
                loss = 0.4 * loss_fn(p1, y) + 1.0 * loss_fn(p2, y)

            epoch_loss += loss.item()

            p2_classes = torch.argmax(p2, dim=1)
            score_bg = calculate_metrics(y, p2_classes)
            score_fg = calculate_foreground_metrics(y, p2_classes)
            metrics_bg = [a + b for a, b in zip(metrics_bg, score_bg)]
            metrics_fg = [a + b for a, b in zip(metrics_fg, score_fg)]

    n = len(loader)
    return epoch_loss / n, [m / n for m in metrics_bg], [m / n for m in metrics_fg]


if __name__ == "__main__":
    # Seeding
    seeding(42)

    # Directories
    create_dir("files")

    # Training logfile
    train_log_path = "files/BUSI_train_log.txt"
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
    batch_size = 8 # SPEED PATCH 6: Increased batch size
    num_epochs = int(os.environ.get("MAX_EPOCHS", "300"))
    lr = 1e-4
    weight_decay = 1e-4
    early_stopping_patience = 50
    checkpoint_path = "files/BUSI_checkpoint.pth"
    path = "dataset_seg_BUSI"

    data_str = (
        f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\n"
        f"Weight Decay: {weight_decay}\nEpochs: {num_epochs}\n"
    )
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
    # CoarseDropout's kwargs changed in albumentations >=1.4 (range tuples) vs the
    # older scalar API (<1.4). Build it compatibly so the same script runs on both
    # the local venv (1.3.x) and the cloud env (2.x).
    try:
        coarse_dropout = A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(1, 24),
            hole_width_range=(1, 24),
            p=0.2,
        )
    except TypeError:
        coarse_dropout = A.CoarseDropout(
            min_holes=1, max_holes=8,
            min_height=1, max_height=24,
            min_width=1, max_width=24,
            p=0.2,
        )

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=25,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.ElasticTransform(alpha=120, sigma=120*0.05,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        coarse_dropout,
    ])

    # Dataset and loader
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    # SPEED PATCH 7: Set num_workers=4 and persistent_workers=True
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True 
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True 
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_cuda_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    
    print_and_save(train_log_path, f"Device: {device}\n")

    model = build_doubleunet()
    backbone_prefixes = (
        "e1.xception",
        "e1.dense_block2",
        "e1.dense_block3",
        "e1.vgg_block4",
        "e1.vgg_block5",
    )
    for name, param in model.named_parameters():
        if name.startswith(backbone_prefixes):
            param.requires_grad = False

    # Same submodules as the freeze loop above (exact name match against
    # backbone_prefixes), kept as module references so their BatchNorm layers
    # can be pinned to eval mode every epoch - see the comment in train().
    frozen_backbone_modules = [
        module for name, module in model.named_modules()
        if name in backbone_prefixes
    ]

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen_params = total_params - trainable_params
    parameter_summary = (
        f"Model parameters - total: {total_params:,}, "
        f"trainable: {trainable_params:,}, frozen: {frozen_params:,}"
    )
    print_and_save(train_log_path, parameter_summary)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        (param for param in model.parameters() if param.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    # Step on validation FOREGROUND F1 (the metric that still has headroom), not
    # val_loss. val_loss plateaus early and noisily, which with the old
    # (mode="min", patience=5, factor=0.1) config collapsed the LR to ~0 mid-run
    # and froze training. mode="max" + higher patience + gentler factor keeps the
    # LR alive while genuinely reducing it only on a real fg-F1 plateau.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=12, factor=0.5
    )
    
    loss_fn = CombinedLoss(num_classes=3, class_weights=BUSI_CLASS_WEIGHTS)
    loss_fn = loss_fn.to(device)
    loss_name = "Weighted CrossEntropy + Foreground Multi-Class Dice Loss"

    data_str = f"Optimizer: Adam (fresh start)\nLoss: {loss_name}\n"
    data_str += f"CE class weights [bg, benign, malignant]: {BUSI_CLASS_WEIGHTS}\n"
    print_and_save(train_log_path, data_str)

    # Training
    best_valid_f1 = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_bg, train_fg = train(
            model, train_loader, optimizer, loss_fn, device, scaler, frozen_backbone_modules
        )
        valid_loss, valid_bg, valid_fg = evaluate(model, valid_loader, loss_fn, device)
        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(valid_fg[1])

        # Checkpoint / early-stopping decision is driven by FOREGROUND-only F1
        # (lesions), not the background-inclusive number.
        if valid_fg[1] > best_valid_f1:
            data_str = (
                f"Valid foreground F1 improved from {best_valid_f1:2.4f} to {valid_fg[1]:2.4f}. "
                f"Saving checkpoint: {checkpoint_path}"
            )
            print_and_save(train_log_path, data_str)

            best_valid_f1 = valid_fg[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | LR: {lr_before:.2e}\n"
        data_str += (
            f"\tTrain Loss: {train_loss:.4f}\n"
            f"\t  [fg]  Jaccard: {train_fg[0]:.4f} - F1: {train_fg[1]:.4f} - "
            f"Recall: {train_fg[2]:.4f} - Precision: {train_fg[3]:.4f}\n"
            f"\t  [all] Jaccard: {train_bg[0]:.4f} - F1: {train_bg[1]:.4f} - "
            f"Recall: {train_bg[2]:.4f} - Precision: {train_bg[3]:.4f}\n"
        )
        data_str += (
            f"\t Val. Loss: {valid_loss:.4f}\n"
            f"\t  [fg]  Jaccard: {valid_fg[0]:.4f} - F1: {valid_fg[1]:.4f} - "
            f"Recall: {valid_fg[2]:.4f} - Precision: {valid_fg[3]:.4f}\n"
            f"\t  [all] Jaccard: {valid_bg[0]:.4f} - F1: {valid_bg[1]:.4f} - "
            f"Recall: {valid_bg[2]:.4f} - Precision: {valid_bg[3]:.4f}\n"
        )
        print_and_save(train_log_path, data_str)

        if early_stopping_count >= early_stopping_patience:
            data_str = (
                f"Early stopping: validation foreground F1 did not improve for "
                f"{early_stopping_patience} consecutive epochs.\n"
            )
            print_and_save(train_log_path, data_str)
            break