# MAMBO Shrub U-Net

Shrub delineation from drone imagery of Strawberry Hills, built with an Attention U-Net trained on manually digitized shrub polygons. Part of the MAMBO project (drone-based biomass estimation).

Given RGB drone orthomosaic tiles (~10mm/pixel), the model predicts a binary shrub / no-shrub mask over the whole site. This repo holds the final, chosen model and the full pipeline that produces and evaluates it — no other experimental variants.

Full methodology, dataset description, results and known limitations: [`docs/MAMBO_Shrub_UNet_Report.pdf`](docs/MAMBO_Shrub_UNet_Report.pdf).

## Model

- Architecture: Attention U-Net, 4 max-pooling encoder stages (reduced from the usual 5-stage design for better spatial detail on small shrubs — see `model.py`).
- Loss: combined BCE + Dice, no border-weighting.
- Threshold: 0.45.
- Input: 512×512 RGB patches at ~10mm resolution.

## Setup requirements

- Python 3.9+
- GPU recommended (CUDA) — training/inference will run on CPU but much slower.
- Packages: `torch`, `torchvision`, `rasterio`, `geopandas`, `shapely`, `opencv-python`, `pillow`, `scipy`, `scikit-learn`, `albumentations`, `tqdm`, `matplotlib`, `numpy`

  ```bash
  pip install torch torchvision rasterio geopandas shapely opencv-python pillow scipy scikit-learn albumentations tqdm matplotlib numpy
  ```

- Data needed before running anything: the drone orthomosaic(s) (`.tif`) and the hand-digitized shrub polygons (shapefile/geopackage) for the training site.
- Each script has a `BASE_DIR` (and related path variables) near the top — point these at your own local data folder before running. They currently hold the paths used for this project and will need editing for a new machine.

## Pipeline — files in the order you run them

### 1. `model.py` — architecture definition
Defines the Attention U-Net (`AttentionUNet`) and its building blocks (`conv_block`, `up_conv`, `Attention_block`). Not run on its own — every other script imports from it. Nothing to configure here.

### 2. `preprocess.py` — build the training/test patches
Takes the full labelled orthomosaic + the digitized shrub polygons and cuts them into 512×512 patches:
- Patches centred on shrub polygons (positives).
- Randomly placed patches that don't overlap any labelled shrub (negatives / background).
- Splits everything into `train/` and `test/` folders (grouping patches from the same large/split shrub together, so one shrub's sub-patches can't end up on both sides of the split — avoids train/test leakage).

Output: `mambo_output/train/images`, `mambo_output/train/labels`, `mambo_output/test/images`, `mambo_output/test/labels` (image/label pairs as georeferenced `.tif` patches).

### 3. `train.py` — train the model
Loads the patches `preprocess.py` produced, trains the Attention U-Net with combined BCE+Dice loss, and saves the best checkpoint (by validation loss, with early stopping) to `model_states/best_model.pth`, plus `model_states/last_train_summary.json` (final train loss/accuracy for that checkpoint). Can be run cell-by-cell in VS Code or as a plain script (`python train.py`).

### 4. New-site validation data (not scripted here)
To check the model generalizes beyond the training site, run `preprocess.py`-style patch extraction on a *second*, independently labelled site, and place the result at `Validation_output/train/images` + `Validation_output/train/labels` (same image/label `.tif` patch format). `report.py` picks this up automatically if present — if it isn't, new-site metrics are simply skipped.

### 5. `inference_full_tile.py` — run the trained model on the full orthomosaic
Takes the trained `best_model.pth` and the full, much larger orthomosaic (e.g. the entire Strawberry Hills site, not just the labelled training area) and predicts over it tile-by-tile, stitching the results back into one seamless output raster (`Strawberry_all_prediction.tif`). Uses an overlap-and-crop tiling strategy to avoid seams/edge artifacts at tile boundaries, and a background-worker `DataLoader` so disk reads and GPU inference overlap (keeps a ~70GB raster from taking all day). Run as a standalone script (`python inference_full_tile.py`) — this is a long job, not meant for cell-by-cell use.

### 6. `report.py` — evaluate and summarize
Loads the trained model and computes:
- **Train metrics** (from `last_train_summary.json`, saved during training).
- **Same-site test metrics** (on the held-out `test/` patches from step 2).
- **New-site metrics** (on the `Validation_output` patches from step 4, if present).

Then appends one row to `experiment_log.csv`, saves a prediction-overlay PNG (original image vs. predicted mask side-by-side, needs `inference_full_tile.py`'s output), and builds a one-file PDF report (metrics table + overlay) under `reports/`.

## Run order, summarized

```
preprocess.py  →  train.py  →  (optional: new-site validation patches)  →  inference_full_tile.py  →  report.py
```

## Known limitation

This model predicts a binary shrub / no-shrub mask — it does not separate individual, touching or overlapping shrub crowns into distinct instances. See the limitations section of the PDF report for details.

## Author

Nabil El Khatri, PhD candidate at UM6P — ecological rehabilitation of phosphate-mined landscapes in semi-arid Morocco.
