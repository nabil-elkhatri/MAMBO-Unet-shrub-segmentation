# =============================================================================
# REPORT.PY — Build one comparable record for a training run: train metrics,
# same-site test metrics, new-site (generalization) metrics, and a saved
# overlay image. Appends one row to experiment_log.csv so every run can be
# compared side by side.
#
# Run this AFTER train.py (and ideally after inference.py, so prediction_map.tif
# is fresh) — cell-by-cell in VS Code, or: python report.py
#
# Standalone on purpose (duplicates RSDataset/metrics like validate_new_site.py
# does) — does NOT import from train.py, so running this never risks
# accidentally triggering a training run.
# =============================================================================

# %%
# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import os
import csv
import json
import datetime
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import rasterio
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torchvision import transforms

# Import the main U-net model and its building blocks from model.py —
# whichever architecture is currently active there is what gets evaluated.
from model import conv_block, up_conv, Attention_block, AttentionUNet

# %%
# =============================================================================
# SECTION 2: PATHS + RUN LABEL — set RUN_LABEL for every run before executing
# =============================================================================
BASE_DIR = Path(r"C:\Users\nabkha\Desktop\MAMBO\Strawberry hills")

# Just a label used to name the CSV row / PDF / overlay for this run.
RUN_LABEL = "v1_baseline_4maxpool"

MODEL_DIR = BASE_DIR / "model_states"
MODEL_PATH = MODEL_DIR / "best_model.pth"
TRAIN_SUMMARY_PATH = MODEL_DIR / "last_train_summary.json"

test_images_dir = BASE_DIR / "mambo_output/test/images"
test_labels_dir = BASE_DIR / "mambo_output/test/labels"

VALID_DIR = BASE_DIR / "Validation_output"
new_site_images_dir = VALID_DIR / "train/images"
new_site_labels_dir = VALID_DIR / "train/labels"

PREDICTION_TIF = BASE_DIR / "prediction_map.tif"            # from inference_full_tile.py
INPUT_TIF = BASE_DIR / "Strawberry_hills_sample_tif.tif"    # same image inference_full_tile.py used

LOG_CSV = BASE_DIR / "experiment_log.csv"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.45

# %%
# =============================================================================
# SECTION 3: DEVICE
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
# =============================================================================
# SECTION 4: DATASET CLASS (evaluation only — no augmentation)
# =============================================================================
class RSDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform or transforms.ToTensor()
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".tif")]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = str(self.images_dir / self.image_files[idx])
        label_path = image_path.replace("images", "labels")

        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        label = np.array(Image.open(label_path).convert("L")) / 255.0

        image = self.transform(image)
        label = np.expand_dims(label, axis=0)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

# %%
# =============================================================================
# SECTION 5: METRICS + LOSS (same definitions as train.py / validate_new_site.py)
# =============================================================================
def calculate_accuracy(outputs, labels, threshold=THRESHOLD):
    preds = (outputs > threshold).float()
    correct = (preds == labels).float().sum()
    total = torch.numel(labels)
    return (correct / total).item()


def calculate_metrics(outputs, labels, threshold=THRESHOLD):
    preds = (outputs > threshold).float()
    tp = ((preds == 1) & (labels == 1)).float().sum()
    fp = ((preds == 1) & (labels == 0)).float().sum()
    fn = ((preds == 0) & (labels == 1)).float().sum()
    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def compute_iou(outputs, labels, threshold=THRESHOLD):
    preds = (outputs > threshold).float()
    intersection = (preds * labels).sum()
    union = ((preds + labels) >= 1).float().sum()
    return (intersection / (union + 1e-8)).item()


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


def evaluate(model, loader, device):
    model.eval()
    totals = {"loss": 0, "acc": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                raw_outputs = model(images)
                loss = combined_loss(raw_outputs, labels)
            probs = torch.sigmoid(raw_outputs)

            precision, recall, f1 = calculate_metrics(probs, labels)
            totals["loss"] += loss.item()
            totals["acc"] += calculate_accuracy(probs, labels)
            totals["precision"] += precision
            totals["recall"] += recall
            totals["f1"] += f1
            totals["iou"] += compute_iou(probs, labels)

    n = len(loader)
    return {k: v / n for k, v in totals.items()}

# %%
# =============================================================================
# SECTION 6: LOAD MODEL
# =============================================================================
model = AttentionUNet(img_ch=3, output_ch=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
print("Model loaded from:", MODEL_PATH)

# %%
# =============================================================================
# SECTION 7: SAME-SITE TEST METRICS
# =============================================================================
test_dataset = RSDataset(test_images_dir, test_labels_dir)
print("Same-site test patches:", len(test_dataset))

same_site_metrics = evaluate(
    model, DataLoader(test_dataset, batch_size=16, shuffle=False), device
)
print("Same-site test:", same_site_metrics)

# %%
# =============================================================================
# SECTION 8: NEW-SITE (GENERALIZATION) METRICS
# =============================================================================
if new_site_images_dir.exists():
    new_site_dataset = RSDataset(new_site_images_dir, new_site_labels_dir)
    print("New-site patches:", len(new_site_dataset))

    new_site_metrics = evaluate(
        model, DataLoader(new_site_dataset, batch_size=16, shuffle=False), device
    )
    print("New-site:", new_site_metrics)
else:
    print(f"New-site patches not found at {new_site_images_dir} — run validate_new_site.py first.")
    new_site_metrics = {"loss": None, "acc": None, "precision": None, "recall": None, "f1": None, "iou": None}

# %%
# =============================================================================
# SECTION 9: TRAIN METRICS (from the JSON train.py saves alongside best_model.pth)
# =============================================================================
if TRAIN_SUMMARY_PATH.exists():
    with open(TRAIN_SUMMARY_PATH) as f:
        train_summary = json.load(f)
    print("Train summary:", train_summary)
else:
    print(f"No train summary found at {TRAIN_SUMMARY_PATH} — train.py needs the json-saving edit, "
          f"or hasn't been run since adding it.")
    train_summary = {"epoch": None, "train_loss": None, "train_accuracy": None}

# %%
# =============================================================================
# SECTION 10: APPEND ONE ROW TO experiment_log.csv
# =============================================================================
row = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "run_label": RUN_LABEL,
    "train_epoch": train_summary.get("epoch"),
    "train_loss": train_summary.get("train_loss"),
    "train_accuracy": train_summary.get("train_accuracy"),
    "test_loss": same_site_metrics["loss"],
    "test_acc": same_site_metrics["acc"],
    "test_precision": same_site_metrics["precision"],
    "test_recall": same_site_metrics["recall"],
    "test_f1": same_site_metrics["f1"],
    "test_iou": same_site_metrics["iou"],
    "newsite_loss": new_site_metrics["loss"],
    "newsite_acc": new_site_metrics["acc"],
    "newsite_precision": new_site_metrics["precision"],
    "newsite_recall": new_site_metrics["recall"],
    "newsite_f1": new_site_metrics["f1"],
    "newsite_iou": new_site_metrics["iou"],
}

file_exists = LOG_CSV.exists()
with open(LOG_CSV, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)

print(f"Row appended to {LOG_CSV}")

# %%
# =============================================================================
# SECTION 11: SAVE A VISUAL OVERLAY FOR THIS RUN (needs prediction_map.tif —
# run inference.py first with this same model for the overlay to be current)
# =============================================================================
overlay_img, overlay_pred_mask = None, None   # kept around for the PDF in Section 13

if PREDICTION_TIF.exists():
    SCALE = 10

    with rasterio.open(INPUT_TIF) as src_img:
        out_shape = (3, src_img.height // SCALE, src_img.width // SCALE)
        overlay_img = src_img.read([1, 2, 3], out_shape=out_shape).transpose(1, 2, 0)
        if overlay_img.max() > 1.0:
            overlay_img = overlay_img / 255.0

    with rasterio.open(PREDICTION_TIF) as src_pred:
        overlay_pred_mask = src_pred.read(1, out_shape=(src_pred.height // SCALE, src_pred.width // SCALE))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(overlay_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(overlay_img)
    axes[1].imshow(overlay_pred_mask, cmap="jet", alpha=0.5)
    axes[1].set_title(f"Prediction Overlay — {RUN_LABEL}")
    axes[1].axis("off")

    plt.tight_layout()
    overlay_path = REPORTS_DIR / f"{RUN_LABEL}_overlay.png"
    plt.savefig(overlay_path, dpi=150)
    plt.show()
    print(f"Overlay saved to {overlay_path}")
else:
    print(f"{PREDICTION_TIF} not found — run inference.py first to get an overlay image for this run.")

# %%
# =============================================================================
# SECTION 12: PRINT A QUICK SUMMARY TABLE FOR THIS RUN
# =============================================================================
print(f"\n=== {RUN_LABEL} ===")
print(f"{'Metric':<15}{'Train':<10}{'Test':<10}{'New-site':<10}")
for key, label in [("loss", "Loss"), ("acc", "Accuracy"), ("f1", "F1"), ("iou", "IoU")]:
    train_val = train_summary.get("train_loss") if key == "loss" else (
        train_summary.get("train_accuracy") if key == "acc" else None)
    test_val = same_site_metrics[key]
    new_val = new_site_metrics[key]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "-"
    print(f"{label:<15}{fmt(train_val):<10}{fmt(test_val):<10}{fmt(new_val):<10}")

# %%
# =============================================================================
# SECTION 13: BUILD A ONE-FILE PDF REPORT FOR THIS RUN — a metrics table page
# + the overlay image page (if available), saved to
# reports/{RUN_LABEL}_report.pdf. Shareable/printable, unlike the raw CSV row.
# =============================================================================
pdf_path = REPORTS_DIR / f"{RUN_LABEL}_report.pdf"
fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "-"

with PdfPages(pdf_path) as pdf:

    # --- Page 1: title + metrics table ---
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
    ax.axis("off")

    ax.text(0.5, 0.95, "MAMBO Shrub U-Net — Run Report", ha="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.91, RUN_LABEL, ha="center", fontsize=12, style="italic", color="dimgray")
    ax.text(0.5, 0.87, row["timestamp"], ha="center", fontsize=9, color="gray")

    table_rows = [
        ("Metric", "Train", "Same-site test", "New-site"),
        ("Loss",
         fmt(train_summary.get("train_loss")), fmt(same_site_metrics["loss"]), fmt(new_site_metrics["loss"])),
        ("Accuracy",
         fmt(train_summary.get("train_accuracy")), fmt(same_site_metrics["acc"]), fmt(new_site_metrics["acc"])),
        ("Precision",
         "-", fmt(same_site_metrics["precision"]), fmt(new_site_metrics["precision"])),
        ("Recall",
         "-", fmt(same_site_metrics["recall"]), fmt(new_site_metrics["recall"])),
        ("F1",
         "-", fmt(same_site_metrics["f1"]), fmt(new_site_metrics["f1"])),
        ("IoU",
         "-", fmt(same_site_metrics["iou"]), fmt(new_site_metrics["iou"])),
    ]

    tbl = ax.table(cellText=table_rows, loc="center", cellLoc="center", bbox=[0.1, 0.45, 0.8, 0.35])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e0e0e0")

    ax.text(0.1, 0.38, f"Best epoch: {train_summary.get('epoch', '-')}", fontsize=9)
    ax.text(0.1, 0.35, f"Threshold used: {THRESHOLD}", fontsize=9)
    ax.text(0.1, 0.32, f"Same-site test patches: {len(test_dataset)}", fontsize=9)
    if new_site_images_dir.exists():
        ax.text(0.1, 0.29, f"New-site patches: {len(new_site_dataset)}", fontsize=9)

    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 2: overlay image, if one was generated this run ---
    if overlay_img is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))  # A4 landscape
        axes[0].imshow(overlay_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(overlay_img)
        axes[1].imshow(overlay_pred_mask, cmap="jet", alpha=0.5)
        axes[1].set_title(f"Prediction Overlay — {RUN_LABEL}")
        axes[1].axis("off")

        fig.suptitle(f"{RUN_LABEL} — prediction overlay", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

print(f"PDF report saved to {pdf_path}")
