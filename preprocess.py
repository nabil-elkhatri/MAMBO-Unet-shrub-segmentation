# =============================================================================
# PREPROCESS.PY — Turn raw drone image + shrub polygons into training patches
# Run cell-by-cell in VS Code (Shift+Enter on each # %% block), or from the
# terminal with: python preprocess.py
# Only needed if you're generating NEW patches (new site, new data). If your
# mambo_output/train and /test folders already have patches, skip this file
# entirely and go straight to train.py.
#
# Includes targeted negative sampling: patches centered on your hand-picked
# grass/other-species points, labeled as background (no shrub). These are
# generated BEFORE the train/test split, so everything — shrubs, random
# background, and targeted negatives — gets split together in ONE pass.
#
# NOTE: this file is also imported by validate_new_site.py
# (`from preprocess import process_data`) — the # %% markers below are just
# comments and don't affect that import.
#
# FIX (this version): shrub patches are now skipped if their window falls
# even partly outside the real image extent, instead of being silently
# saved. Previously, big shrubs (which get split into 4 overlapping windows
# by shrub_overlaps()) had NO bounds check at all — only background_samples()
# had one. A window that fell outside the image still got its label drawn
# correctly (rasterize() only needs a transform, it doesn't care whether the
# window is within the real image), but the image read for that same window
# came back black. Result: patches with a correct shrub label pointing at a
# black image — actively wrong training signal. This version skips those
# windows instead, for both the huge-shrub and normal-shrub code paths.
# =============================================================================

# %%
# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import os
import shutil
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)

import rasterio
from rasterio.windows import Window
from rasterio.windows import from_bounds
from rasterio.coords import BoundingBox
from rasterio.features import rasterize

import geopandas as gpd
from shapely.geometry import box

import numpy as np
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# %%
# =============================================================================
# SECTION 2: PATHS — edit these for whichever site you're processing
# =============================================================================
BASE_DIR = Path(r"C:\Users\nabkha\Desktop\MAMBO\Strawberry hills")

# Training site (Strawberry Hills)
IMG = BASE_DIR / "Strawberry_hills_sample_tif.tif"
PLGN = BASE_DIR / "Strawberry_sample_training.gpkg"
NEGATIVE_PTS = BASE_DIR / "negative_training.gpkg"   # your hand-picked grass/other-species points
OUTPUT_DIR = BASE_DIR / "mambo_output"

# %%
# =============================================================================
# SECTION 3: SAVING INDIVIDUAL PATCHES TO DISK
# =============================================================================
def save_image_patch(window, image, index, directory="images"):
    image_patch = image.read(window=window)
    transform = rasterio.windows.transform(window, image.transform)
    meta = image.meta.copy()
    meta.update({"height": window.height, "width": window.width, "transform": transform})
    path = os.path.join(directory, f"shrubs_{index}.tif")
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(image_patch)


def save_label_patch(data, window, image, index, directory="labels"):
    transform = rasterio.windows.transform(window, image.transform)
    meta = image.meta.copy()
    meta.update({"height": window.height, "width": window.width, "transform": transform, "count": 1})
    path = os.path.join(directory, f"shrubs_{index}.tif")
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data, 1)


def label_patch_with_window_instance_ids(geoms, window, image):
    """
    Like label_patch_with_window, but instead of one shared value (255) for
    every shrub pixel, each GEOMETRY gets its OWN integer id (1, 2, 3, ...),
    local to this window. Background stays 0.

    This is what lets train_v3_unet_id.py know the TRUE number of distinct
    shrubs inside a touching/overlapping cluster, instead of only being able
    to tell "there is shrub here" from the plain binary label — the ID U-Net
    (Method C) needs that true count to know how hard it needs to erode to
    actually separate a specific cluster, rather than guessing from pixel
    connectivity after the fact.

    Saved as a SEPARATE file (labels_instance/) — the existing binary
    labels/ output is untouched, so methods A and B (plain binary
    segmentation) are not affected by this at all.

    NOTE: if two shrub polygons' pixels genuinely overlap (not just touch),
    rasterize() keeps whichever geometry is drawn last at the shared pixels
    — an acceptable approximation; touching-but-not-overlapping crowns (the
    normal case) get fully correct, separate ids.
    """
    transform = rasterio.windows.transform(window, image.transform)
    shapes = [(geom, local_id) for local_id, geom in enumerate(geoms.geometry, start=1)]
    if not shapes:
        return np.zeros((int(window.height), int(window.width)), dtype=np.int32)
    arr = rasterize(
        shapes,
        out_shape=(int(window.height), int(window.width)),
        transform=transform,
        fill=0,
        dtype="int32",
    )
    return arr


def save_instance_label_patch(data, window, image, index, directory="labels_instance"):
    transform = rasterio.windows.transform(window, image.transform)
    meta = image.meta.copy()
    meta.update({"height": window.height, "width": window.width, "transform": transform,
                 "count": 1, "dtype": "int32"})
    path = os.path.join(directory, f"shrubs_{index}.tif")
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data, 1)

# %%
# =============================================================================
# SECTION 4: WINDOW GEOMETRY HELPERS
# =============================================================================
def patch_window(geom, image, patch_size=512):
    # Works for both polygons and points — geom.centroid of a point is itself.
    half_patch = patch_size // 2
    center_x, center_y = geom.centroid.x, geom.centroid.y
    row, col = image.index(center_x, center_y)
    window = Window(col - half_patch, row - half_patch, patch_size, patch_size)
    return window


def shrub_window(shrub, image):
    bounds = shrub.geometry.bounds
    win = from_bounds(*bounds, image.transform)
    return win


def is_shrub_huge(shrub_px, size=512):
    return shrub_px.height > size or shrub_px.width > size


def shrub_overlaps(shrub, image, patch_size=512):
    shift = patch_size * 0.75
    center_x, center_y = shrub.geometry.centroid.x, shrub.geometry.centroid.y
    row, col = image.index(center_x, center_y)
    windows = []
    windows.append(Window(col - shift, row - shift, patch_size, patch_size))
    windows.append(Window(col, row - shift, patch_size, patch_size))
    windows.append(Window(col - shift, row, patch_size, patch_size))
    windows.append(Window(col, row, patch_size, patch_size))
    return windows


def shrub_labels_in_window(geometries, window, image):
    bounds = rasterio.windows.bounds(window, image.transform)
    bbox = box(*bounds)
    s = geometries.intersection(bbox)
    out_series = s[~(s.is_empty)]
    return out_series


def label_patch_with_window(geoms, window, image):
    transform = rasterio.windows.transform(window, image.transform)
    arr = rasterize(
        geoms.geometry,
        out_shape=(int(window.height), int(window.width)),
        transform=transform,
        default_value=255,
    )
    return arr


def window_is_within_image(window, image):
    """FIX: shared bounds check, used for every patch type. A window that
    falls even partly outside the real image would previously be saved as a
    black image paired with a correctly-drawn label — this stops that."""
    window_bounds = rasterio.windows.bounds(window, image.transform)
    return box(*image.bounds).contains(box(*window_bounds))

# %%
# =============================================================================
# SECTION 5: RANDOM BACKGROUND (NEGATIVE) SAMPLING
# =============================================================================
def background_samples(image, shrubs, window_size=512, within_df=False, max_samples=50):
    img_bounds = image.bounds
    if within_df:
        img_bounds = BoundingBox(*shrubs.total_bounds.tolist())

    shrub_buffer = shrubs.copy()
    shrub_buffer["geometry"] = shrub_buffer.geometry.buffer(5)
    shrub_union = shrub_buffer.union_all()

    num_negative_samples = min(len(shrubs) * 2, max_samples)
    negative_windows = []
    attempts = 0
    max_attempts = num_negative_samples * 10

    while len(negative_windows) < num_negative_samples and attempts < max_attempts:
        rand_x = random.uniform(img_bounds.left, img_bounds.right)
        rand_y = random.uniform(img_bounds.bottom, img_bounds.top)
        row, col = image.index(rand_x, rand_y)

        half_patch = window_size // 2
        potential_window = Window(col - half_patch, row - half_patch, window_size, window_size)

        if not window_is_within_image(potential_window, image):
            attempts += 1
            continue

        window_bounds = rasterio.windows.bounds(potential_window, image.transform)
        window_bbox = box(*window_bounds)

        overlap = window_bbox.intersects(shrub_union)
        if overlap:
            attempts += 1
            continue

        window_data = image.read(window=potential_window)
        if len(np.unique(window_data)) > 1:
            negative_windows.append(potential_window)
        attempts += 1

    return negative_windows


def background_label(size=512):
    return np.zeros((size, size), dtype=np.uint8)

# %%
# =============================================================================
# SECTION 6: TRAIN/TEST SPLIT
# =============================================================================
def test_train_split(output_dir, label="shrubs", also_move=("labels_instance",)):
    image_files = [os.path.join(output_dir, "images", f)
                   for f in os.listdir(os.path.join(output_dir, "images"))]
    label_files = [os.path.join(output_dir, "labels", f)
                   for f in os.listdir(os.path.join(output_dir, "labels"))]

    def get_index(filename):
        image = os.path.basename(filename).split("_")[-1]
        image = image.replace(".tif", "")
        return image

    image_indices = [get_index(f) for f in image_files]
    label_indices = [get_index(f) for f in label_files]

    if sorted(image_indices) != sorted(label_indices):
        raise ValueError("Indices of image and label files do not match.")

    all_indices = sorted(list(set(image_indices)))

    # --- LEAKAGE FIX ---
    # A big shrub that got split into overlapping sub-patches has indices like
    # "12.0", "12.1", "12.2", "12.3" — these all share real, physical ground
    # with each other. If they land on different sides of the split, the
    # "test" set secretly contains ground the model already trained on.
    # Fix: group by the shrub's base id (the part before the dot) and split
    # by GROUP, not by individual patch — so all sub-patches of one shrub
    # always end up entirely in train, or entirely in test, never mixed.
    def base_id(index):
        return index.split(".")[0]

    groups = {}
    for idx in all_indices:
        groups.setdefault(base_id(idx), []).append(idx)

    group_keys = sorted(groups.keys())
    train_groups, test_groups = train_test_split(group_keys, test_size=0.2, random_state=42)

    train_indices = [idx for g in train_groups for idx in groups[g]]
    test_indices = [idx for g in test_groups for idx in groups[g]]
    # --- END LEAKAGE FIX ---

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)
    # FIX: also create + move labels_instance/ (per-shrub unique-id labels,
    # used only by train_v3_unet_id.py) alongside images/ and labels/, so it
    # stays in sync with the same train/test split instead of being left
    # behind in the top-level output_dir.
    for extra_dir in also_move:
        os.makedirs(os.path.join(train_dir, extra_dir), exist_ok=True)
        os.makedirs(os.path.join(test_dir, extra_dir), exist_ok=True)

    def move_files(indices, source_dir, dest_dir):
        for index in indices:
            filename = f"{label}_{index}.tif"
            folders_to_move = ("images", "labels") + also_move
            for folder in folders_to_move:
                source_path = os.path.join(source_dir, folder, filename)
                dest_path = os.path.join(dest_dir, folder, filename)
                if os.path.exists(source_path):
                    shutil.move(source_path, dest_path)
                else:
                    logging.info(f"Warning: {folder} file not found for index {index}")

    move_files(train_indices, output_dir, train_dir)
    move_files(test_indices, output_dir, test_dir)

    logging.info(f"Data split into train ({len(train_indices)} samples) "
                 f"and test ({len(test_indices)} samples).")

# %%
# =============================================================================
# SECTION 7: process_data — THE ORCHESTRATOR
# =============================================================================
def process_data(raster_path, shapefile_path, output_dir, label, window_size=512,
                  negative_shapefile_path=None):
    labels_dir = os.path.join(output_dir, "labels")
    images_dir = os.path.join(output_dir, "images")
    # Instance-id labels — only needed for train_v3_unet_id.py (Method C).
    # Kept fully separate from labels_dir so A/B's binary pipeline is
    # unaffected; see label_patch_with_window_instance_ids above.
    labels_instance_dir = os.path.join(output_dir, "labels_instance")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_instance_dir, exist_ok=True)

    shrubs = gpd.read_file(shapefile_path)
    total_shrubs = len(shrubs)

    skipped_out_of_bounds = 0

    with rasterio.open(raster_path) as image:
        # --- shrub patches ---
        for index, shrub in tqdm(shrubs.iterrows(), total=len(shrubs), desc="Shrub patches"):
            shrub_px = shrub_window(shrub, image)
            if is_shrub_huge(shrub_px, window_size):
                windows = shrub_overlaps(shrub, image, window_size)
            else:
                windows = [patch_window(shrub.geometry, image, patch_size=window_size)]

            for i, window in enumerate(windows):
                use_index = f"{index}.{i}"

                # FIX: skip windows that fall even partly outside the real
                # image instead of silently saving a black image paired
                # with a correctly-drawn label.
                if not window_is_within_image(window, image):
                    skipped_out_of_bounds += 1
                    continue

                labels = shrub_labels_in_window(shrubs, window, image)
                arr = label_patch_with_window(labels, window, image)
                instance_arr = label_patch_with_window_instance_ids(labels, window, image)
                save_image_patch(window, image, use_index, directory=images_dir)
                save_label_patch(arr, window, image, use_index, directory=labels_dir)
                save_instance_label_patch(instance_arr, window, image, use_index, directory=labels_instance_dir)

        # --- random background patches ---
        print("Selecting background examples")
        negative_windows = background_samples(image, shrubs, window_size=window_size, within_df=True)

        for idx, neg_window in enumerate(negative_windows):
            bg_index = total_shrubs + idx
            save_image_patch(neg_window, image, bg_index, directory=images_dir)
            save_label_patch(background_label(window_size), neg_window, image, bg_index, directory=labels_dir)
            # All-background negative -> instance-id label is all zeros too,
            # same shape, so train_v3_unet_id.py's labels/labels_instance
            # file pairing (one file per image, by name) stays intact.
            save_instance_label_patch(
                np.zeros((window_size, window_size), dtype=np.int32),
                neg_window, image, bg_index, directory=labels_instance_dir,
            )

        # --- targeted negative patches: saved for use AFTER the split (see below) ---
        negative_windows_for_train = []
        if negative_shapefile_path is not None:
            negatives = gpd.read_file(negative_shapefile_path)
            print(f"Number of targeted negative points: {len(negatives)}")

            for index, neg in tqdm(negatives.iterrows(), total=len(negatives), desc="Reading negative points"):
                window = patch_window(neg.geometry, image, patch_size=window_size)
                if not window_is_within_image(window, image):
                    continue
                negative_windows_for_train.append((window, index))

    print(f"Shrub sub-patches skipped (window outside real image bounds): {skipped_out_of_bounds}")

    # Split shrubs + random background ONLY (same as before — reproducible,
    # not affected by negatives at all).
    test_train_split(output_dir, label=label)

    # Now write ALL targeted negatives directly into train/ — guaranteed,
    # not subject to the random 80/20 split, since they exist to teach the
    # model what to reject, not to be held out for evaluation.
    if negative_shapefile_path is not None and negative_windows_for_train:
        train_images_dir = os.path.join(output_dir, "train", "images")
        train_labels_dir = os.path.join(output_dir, "train", "labels")
        train_labels_instance_dir = os.path.join(output_dir, "train", "labels_instance")
        os.makedirs(train_labels_instance_dir, exist_ok=True)

        with rasterio.open(raster_path) as image:
            saved_negatives = 0
            for window, index in tqdm(negative_windows_for_train, desc="Saving negatives into train/"):
                neg_index = f"neg{index}"
                save_image_patch(window, image, neg_index, directory=train_images_dir)
                save_label_patch(background_label(window_size), window, image, neg_index, directory=train_labels_dir)
                save_instance_label_patch(
                    np.zeros((window_size, window_size), dtype=np.int32),
                    window, image, neg_index, directory=train_labels_instance_dir,
                )
                saved_negatives += 1

        print(f"Targeted negative patches saved directly into train/: {saved_negatives} of {len(negative_windows_for_train)}")

    print("process_data complete.")

# %%
# =============================================================================
# SECTION 8: SANITY CHECK — confirm image/shapefiles actually overlap before
# running the (potentially long) real pipeline below.
# NOTE: kept under `if __name__ == "__main__":` on purpose — this file is
# imported by validate_new_site.py, and without the guard that import would
# trigger these checks (and the full pipeline below) automatically.
# =============================================================================
if __name__ == "__main__":
    with rasterio.open(IMG) as _src:
        print("Image bounds:", _src.bounds)
        print("Image CRS:", _src.crs)

    _shrubs_check = gpd.read_file(PLGN)
    print("Shrubs bounds:", _shrubs_check.total_bounds)
    print("Shrubs CRS:", _shrubs_check.crs)
    print("Number of shrub polygons:", len(_shrubs_check))

    _negatives_check = gpd.read_file(NEGATIVE_PTS)
    print("Negative points bounds:", _negatives_check.total_bounds)
    print("Negative points CRS:", _negatives_check.crs)
    print("Number of negative points:", len(_negatives_check))

# %%
# =============================================================================
# SECTION 9: RUN THE PIPELINE
# =============================================================================
if __name__ == "__main__":
    process_data(
        raster_path=IMG,
        shapefile_path=PLGN,
        output_dir=OUTPUT_DIR,
        label="shrubs",
        window_size=512,
        negative_shapefile_path=NEGATIVE_PTS,
    )
