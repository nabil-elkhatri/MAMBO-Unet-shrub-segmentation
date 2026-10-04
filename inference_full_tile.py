# =============================================================================
# INFERENCE_FULL_TILE.PY — Apply the trained model to the FULL Strawberry Hills
# orthomosaic (Strawberry_all.tif, ~70 GB).
#
# v2: the first version read each window from disk one at a time in the same
# thread that runs the GPU, so the GPU sat idle during every read and disk sat
# idle during every forward pass — fully serial, ~4 windows/s, ~11 hour ETA.
# This version reads windows with several background worker processes
# (a torch DataLoader) while the GPU works on the previous batch, so I/O and
# compute overlap instead of alternating. Everything else (overlap-and-crop
# writing, BigTIFF output, full-extent coverage) is unchanged from v1.
#
# Run from the terminal (NOT cell-by-cell — this is a single long job):
#   python inference_full_tile.py
# Needs a trained model already saved (model_states/best_model.pth).
#
# TUNING if it's still slow — check Task Manager -> Performance while it runs:
#   - GPU usage pinned near 100%, disk usage low  -> you're compute-bound.
#     Raise BATCH_SIZE until you're close to an out-of-memory error.
#   - GPU usage low, disk usage high               -> you're I/O-bound.
#     Raise NUM_WORKERS (try your CPU core count), or move the .tif to an SSD
#     if it's currently on an HDD / network drive.
#   - Both low                                     -> something else is the
#     bottleneck (antivirus scanning each read, a slow network drive, etc.)
# =============================================================================

import logging
import time
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioIOError

import torch
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader

from model import conv_block, up_conv, Attention_block, AttentionUNet

# =============================================================================
# SECTION 1: PATHS + RUN SELECTION
# =============================================================================
BASE_DIR = Path(r"C:\Users\nabkha\Desktop\MAMBO\Strawberry hills")

MODEL_DIR = BASE_DIR / "model_states"
MODEL_PATH = MODEL_DIR / "best_model.pth"

INPUT_TIF = BASE_DIR / "Strawberry_all.tif"                    # the full ~70GB orthomosaic
OUTPUT_TIF = BASE_DIR / "Strawberry_all_prediction.tif"

# =============================================================================
# SECTION 2: TILING + PERFORMANCE PARAMETERS
# =============================================================================
WINDOW = 512           # must match the model's training patch size
MARGIN = 64            # pixels trimmed off each side of a prediction before writing
STRIDE = WINDOW - 2 * MARGIN    # = 384: how far the window advances each step
BATCH_SIZE = 32         # windows per forward pass — raise if GPU has headroom, lower on OOM
NUM_WORKERS = 8         # parallel background processes reading windows from disk —
                         # try setting this near your CPU's core count
PREFETCH_FACTOR = 4     # batches each worker stages ahead of time
THRESHOLD = 0.45        # same threshold used in inference.py / report.py

# Device + model are set up inside run_full_tile_inference(), NOT here at
# module level. On Windows, each DataLoader worker process re-imports this
# whole script when it spawns — code left at module level (outside any
# function) runs again in every one of the NUM_WORKERS processes. Module-level
# device/model setup previously caused the model to be loaded onto the GPU
# once per worker (8 redundant copies, visible as repeated "Model loaded"
# log lines at startup) even though workers never run the model — they only
# read windows from disk. Keeping this inside the function means it only runs
# once, in the main process.

# =============================================================================
# SECTION 4: WINDOW ORIGINS (covers the full raster, including the last
# partial row/column — no area is skipped) and the dataset that reads them
# =============================================================================
def window_origins(width, height, stride):
    xs = list(range(0, width, stride))
    ys = list(range(0, height, stride))
    return [(x, y) for y in ys for x in xs]


class WindowDataset(Dataset):
    """Reads one WINDOW x WINDOW tile per item. Each worker process opens its
    own rasterio handle on first use (file handles can't be shared safely
    across multiprocessing workers), so __getitem__ lazily opens it."""

    def __init__(self, input_path, origins, window, margin):
        self.input_path = str(input_path)
        self.origins = origins
        self.window = window
        self.margin = margin
        self._src = None

    def _ensure_open(self):
        if self._src is None:
            self._src = rasterio.open(self.input_path)

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, idx):
        self._ensure_open()
        x, y = self.origins[idx]
        read_window = Window(x - self.margin, y - self.margin, self.window, self.window)
        # boundless=True + fill_value=0: always returns a full WINDOW x WINDOW
        # read, even past the raster edge or with a negative offset (the
        # margin) — no special-casing needed for edge tiles.
        tile = self._src.read(window=read_window, boundless=True, fill_value=0) / 255.0
        return torch.from_numpy(tile).float(), x, y


# =============================================================================
# SECTION 5: RUN — background workers prefetch windows while the GPU runs the
# previous batch; the main process only writes finished predictions to disk.
# Device setup and model loading live here (not at module level) so they run
# exactly once, in the main process — see the note above SECTION 3.
# =============================================================================
def run_full_tile_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        log.warning("No GPU detected — running on CPU. This WILL be very slow on a 70GB tile.")
    log.info(f"Using device: {device}")

    model = AttentionUNet(img_ch=3, output_ch=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    log.info(f"Model loaded from: {MODEL_PATH}")
    torch.cuda.empty_cache()

    try:
        with rasterio.open(INPUT_TIF) as src:
            width, height = src.width, src.height
            dtype = src.dtypes[0]
            out_meta = src.meta.copy()

        log.info(f"Input raster: {width} x {height} px, dtype {dtype}")

        out_meta.update({
            "count": 1,
            "dtype": "uint8",
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "YES",   # required: output can exceed 4GB at this extent
        })

        origins = window_origins(width, height, STRIDE)
        total = len(origins)
        log.info(
            f"{total} windows to process (window={WINDOW}, margin={MARGIN}, stride={STRIDE}, "
            f"batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS})"
        )

        dataset = WindowDataset(INPUT_TIF, origins, WINDOW, MARGIN)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
            persistent_workers=(NUM_WORKERS > 0),
        )

        Path(OUTPUT_TIF).parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        done = 0

        with rasterio.open(OUTPUT_TIF, "w", **out_meta) as dst:
            with torch.no_grad():
                for batch_tensor, xs, ys in loader:
                    batch_input = batch_tensor.to(device, non_blocking=True)
                    with autocast():
                        raw_outputs = model(batch_input)
                    probs = torch.sigmoid(raw_outputs).squeeze(1).cpu().numpy()
                    if probs.ndim == 2:
                        probs = np.expand_dims(probs, axis=0)
                    preds = (probs > THRESHOLD).astype(np.uint8)

                    xs = xs.tolist()
                    ys = ys.tolist()
                    for idx, (x, y) in enumerate(zip(xs, ys)):
                        # Keep only the centre MARGIN:-MARGIN crop of this window's
                        # prediction (least affected by convolution edge effects)
                        # and write it at its true (x, y) position. Every output
                        # pixel is written exactly once, so no accumulation
                        # buffer is needed.
                        centre = preds[idx][MARGIN:MARGIN + STRIDE, MARGIN:MARGIN + STRIDE]

                        # Clip the write window at the true raster edge (the last
                        # row/column of windows overshoots the real extent).
                        w = min(STRIDE, width - x)
                        h = min(STRIDE, height - y)
                        if w <= 0 or h <= 0:
                            continue
                        dst.write(centre[:h, :w], 1, window=Window(x, y, w, h))

                    done += len(xs)
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta_min = (total - done) / rate / 60 if rate > 0 else float("inf")
                    log.info(
                        f"{done}/{total} windows ({done/total*100:.1f}%) — "
                        f"{rate:.1f} windows/s — ETA {eta_min:.1f} min"
                    )

        log.info(f"Done. Prediction written to {OUTPUT_TIF}")

    except RasterioIOError as err:
        log.error(err)
        raise


if __name__ == "__main__":
    run_full_tile_inference()
