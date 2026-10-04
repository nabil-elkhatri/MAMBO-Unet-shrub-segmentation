# =============================================================================
# TRAIN.PY — Mambo Shrub U-Net Training Script (final model)
# Run cell-by-cell in VS Code (Shift+Enter on each # %% block), or from the
# terminal with: python train.py
# This code does training. Preprocessing, inference, and visualization stay
# in their own scripts.
# =============================================================================

# %%
# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import os
import sys
import json
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)

import rasterio
from rasterio.windows import Window
from rasterio.windows import from_bounds
from rasterio.coords import BoundingBox
from rasterio.features import rasterize
from rasterio.errors import RasterioIOError

import geopandas as gpd
from shapely.geometry import box

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage

import random
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

import albumentations as A

from tqdm import tqdm
import gc

from torch.utils.tensorboard import SummaryWriter

# Import the main U-net model and its building blocks from model.py
from model import conv_block, up_conv, Attention_block, AttentionUNet

# %%
# =============================================================================
# SECTION 2: PATHS
# =============================================================================
BASE_DIR = Path(r"C:\Users\nabkha\Desktop\MAMBO\Strawberry hills")

train_images_dir = BASE_DIR / "mambo_output/train/images"
train_labels_dir = BASE_DIR / "mambo_output/train/labels"
test_images_dir = BASE_DIR / "mambo_output/test/images"
test_labels_dir = BASE_DIR / "mambo_output/test/labels"

# Everything this run produces saves straight here.
MODEL_DIR = BASE_DIR / "model_states"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RUNS_DIR = BASE_DIR / "runs"

# %%
# =============================================================================
# SECTION 3: DEVICE (GPU check)
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    logging.warning("No GPU detected — running on CPU. Training will be much slower.")
print("Using device:", device)

# %%
# =============================================================================
# SECTION 4: DATASET CLASS (RSDataset)
# =============================================================================
class RSDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, augment=False,
                 repeat_augmentations=0):
        self.images_dir = Path(images_dir)
        self.images = os.listdir(images_dir)
        self.labels_dir = Path(labels_dir)
        self.labels = os.listdir(labels_dir)
        self.transform = transform

        if not self.transform:
            self.transform = transforms.ToTensor()

        self.augment = augment
        self.repeat_augmentations = repeat_augmentations

        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".tif"))]
        self.image_files.sort()

        if self.augment:
            self.aug = A.Compose([
                A.Rotate(limit=5, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
                A.GaussNoise(p=0.2),
                A.RandomScale(scale_limit=0.15, p=0.3),
                A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, p=1.0),
                A.RandomCrop(height=512, width=512, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.image_files) * (1 + self.repeat_augmentations)

    def __getitem__(self, idx):
        if self.aug:
            base_image_idx = idx // (1 + self.repeat_augmentations)
            is_augmented = (idx % (1 + self.repeat_augmentations)) > 0
        else:
            is_augmented = False
            base_image_idx = idx

        image_path = str(self.images_dir / self.image_files[base_image_idx])
        label_path = image_path.replace("images", "labels")

        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        label = np.array(Image.open(label_path).convert("L")) / 255.0

        if is_augmented and self.aug:
            augmented = self.aug(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        image = self.transform(image)
        label = np.expand_dims(label, axis=0)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label

# %%
# =============================================================================
# SECTION 5: METRICS
# =============================================================================
def calculate_accuracy(outputs, labels, threshold=0.45):
    preds = (outputs > threshold).float()
    correct = (preds == labels).float().sum()
    total = torch.numel(labels)
    return (correct / total).item()


def calculate_metrics(outputs, labels, threshold=0.45):
    preds = (outputs > threshold).float()
    tp = ((preds == 1) & (labels == 1)).float().sum()
    fp = ((preds == 1) & (labels == 0)).float().sum()
    fn = ((preds == 0) & (labels == 1)).float().sum()

    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def compute_iou(outputs, labels, threshold=0.45):
    preds = (outputs > threshold).float()
    intersection = (preds * labels).sum()
    union = ((preds + labels) >= 1).float().sum()
    return (intersection / (union + 1e-8)).item()

# %%
# =============================================================================
# SECTION 6: LOSS FUNCTIONS (Dice + BCE combined)
# =============================================================================
def dice_loss(pred_logits, target, smooth=1.0):
    pred = torch.sigmoid(pred_logits)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred_logits, target, bce_weight=0.5):
    bce = nn.functional.binary_cross_entropy_with_logits(pred_logits, target)
    dice = dice_loss(pred_logits, target)
    return bce_weight * bce + (1 - bce_weight) * dice

# %%
# =============================================================================
# SECTION 7: TRAINING LOOP
# =============================================================================
def train_model(model, train_dataset, val_dataset, epochs=50, batch_size=16, lr=0.0001,
                 accumulation_steps=4, device="cpu", model_dir="model_states", patience=5,
                 bce_weight=0.5, train_summary_writer=None, val_summary_writer=None):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    criterion = lambda logits, target: combined_loss(logits, target, bce_weight=bce_weight)
    val_criterion = lambda logits, target: combined_loss(logits, target, bce_weight=bce_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    scaler = GradScaler()

    model.to(device)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if not Path(model_dir).exists():
        Path(model_dir).mkdir(parents=True, exist_ok=True)
    best_model_path = Path(model_dir) / "best_model.pth"

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.float().to(device), labels.float().to(device)
            labels = torch.clamp(labels, 0, 1)

            if labels.ndim == 3:
                labels = labels.unsqueeze(1)

            if labels.max() > 1 or labels.min() < 0:
                raise ValueError(f"Labels out of bounds: min={labels.min()}, max={labels.max()}")

            with autocast():
                raw_outputs = model(images)
                loss = criterion(raw_outputs, labels)

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                probs = torch.sigmoid(raw_outputs)
                acc = calculate_accuracy(probs, labels)

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item(), "accuracy": acc})

            if train_summary_writer:
                train_summary_writer.add_scalar("Loss", loss.item(), epoch * len(train_loader) + i)
                train_summary_writer.add_scalar("Accuracy", acc, epoch * len(train_loader) + i)

        scheduler.step()
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(train_loader):.4f}")

        val_loss, val_acc, val_precision, val_recall, val_f1, val_iou = validate_model(
            model, val_loader, val_criterion, device
        )
        print(f"Validation: Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}, "
              f"Precision = {val_precision:.4f}, Recall = {val_recall:.4f}, "
              f"F1 = {val_f1:.4f}, IoU = {val_iou:.4f}")

        if val_summary_writer:
            val_summary_writer.add_scalar("Loss", val_loss, epoch)
            val_summary_writer.add_scalar("Accuracy", val_acc, epoch)
            val_summary_writer.add_scalar("Precision", val_precision, epoch)
            val_summary_writer.add_scalar("Recall", val_recall, epoch)
            val_summary_writer.add_scalar("F1", val_f1, epoch)
            val_summary_writer.add_scalar("IoU", val_iou, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

            # Save this epoch's TRAIN metrics alongside the model, so report.py
            # can pull them in later without re-running training.
            train_summary = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss / len(train_loader),
                "train_accuracy": acc,
            }
            with open(Path(model_dir) / "last_train_summary.json", "w") as f:
                json.dump(train_summary, f, indent=2)

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"No improvement for {patience} epochs - stopping early at epoch {epoch + 1}")
                break

    return model


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss, val_acc, val_precision, val_recall, val_f1, val_iou = 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                raw_outputs = model(images)
                loss = criterion(raw_outputs, labels)
            probs = torch.sigmoid(raw_outputs)

            val_loss += loss.item()
            acc = calculate_accuracy(probs, labels)
            precision, recall, f1 = calculate_metrics(probs, labels)
            iou = compute_iou(probs, labels)

            val_acc += acc
            val_precision += precision
            val_recall += recall
            val_f1 += f1
            val_iou += iou

    n = len(val_loader)
    return val_loss / n, val_acc / n, val_precision / n, val_recall / n, val_f1 / n, val_iou / n

# %%
# =============================================================================
# SECTION 8: BUILD DATASETS + MODEL
# NOTE: kept under `if __name__ == "__main__":` on purpose — this file gets
# imported by other scripts (e.g. `from train import RSDataset` in
# threshold_sweep.py), and without the guard that import would kick off a
# full training run automatically.
# =============================================================================
if __name__ == "__main__":
    train_dataset = RSDataset(train_images_dir, train_labels_dir, augment=True, repeat_augmentations=2)
    val_dataset = RSDataset(test_images_dir, test_labels_dir)

    print("Train examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))

    torch.cuda.empty_cache()
    gc.collect()

    model = AttentionUNet(img_ch=3, output_ch=1)

    train_writer = SummaryWriter(log_dir=str(RUNS_DIR / "train"))
    val_writer = SummaryWriter(log_dir=str(RUNS_DIR / "val"))

# %%
# =============================================================================
# SECTION 9: RUN TRAINING — this is the long-running cell. Re-run it to
# start a fresh training run (re-run Section 8 first if you want a fresh,
# untrained model instead of continuing from whatever `model` currently is).
# =============================================================================
if __name__ == "__main__":
    trained_model = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=50,
        batch_size=16,
        lr=0.0001,
        device=device,
        model_dir=MODEL_DIR,
        patience=10,
        train_summary_writer=train_writer,
        val_summary_writer=val_writer,
    )

    print("Training complete.")
