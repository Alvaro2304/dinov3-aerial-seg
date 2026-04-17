# dinov3-aerial-seg

First-pass inference with Meta's **CHMv2** canopy-height model
(`facebook/dinov3-vitl16-chmv2-dpt-head`, DINOv3 Sat-L backbone + DPT head)
on aerial RGB JPGs from a SONY ILX-LR1. Predicts per-pixel canopy height in
meters; a binary canopy mask can be obtained later by thresholding.

---

## 0. Prerequisites (Windows PC)

- Windows 10/11 with an NVIDIA GPU (tested target: RTX 4080 Super, 16 GB VRAM).
- Up-to-date NVIDIA driver (R550+ is enough for CUDA 12.4).
  Check: `nvidia-smi` in a terminal should print the GPU name and a CUDA version.
- Python **3.10 or 3.11** installed (add to PATH). Grab from
  https://www.python.org/downloads/windows/ — *not* the Microsoft Store build.
- Git for Windows: https://git-scm.com/download/win
- ~10 GB free disk (model weights are ~1.2 GB; torch CUDA wheels are large).

No Anaconda is needed — `venv` + `pip` is enough. If you prefer conda, install
**Miniconda** (not full Anaconda): https://www.anaconda.com/docs/getting-started/miniconda/install

---

## 1. Get the code

```bat
cd C:\Users\<you>\Documents
git clone <your-remote-url> dinov3-aerial-seg
cd dinov3-aerial-seg
```

Folder layout after cloning:

```
dinov3-aerial-seg\
├── chmv2_infer.py        main inference script
├── requirements.txt
├── data\                 put your input JPGs here
└── outputs\              results go here
```

---

## 2. Create a Python environment

### Option A — venv (recommended)

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### Option B — Miniconda

```bat
conda create -n chmv2 python=3.11 -y
conda activate chmv2
python -m pip install --upgrade pip
```

Your shell prompt should now start with `(.venv)` or `(chmv2)`.

---

## 3. Install PyTorch with CUDA

Pick the command that matches your installed CUDA driver
(`nvidia-smi` → top right corner shows "CUDA Version: 12.x").

```bat
:: CUDA 12.4 (works on any driver >= R550)
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

:: CUDA 12.6
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```

Verify CUDA is visible to PyTorch:

```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `True NVIDIA GeForce RTX 4080 SUPER` (or similar).
If it prints `False`, the driver/CUDA/torch versions don't match — reinstall
torch with the correct `cuXXX` index.

---

## 4. Install the rest of the dependencies

```bat
pip install -r requirements.txt
```

That pulls `transformers`, `Pillow`, `numpy`, `matplotlib`. CHMv2 support
requires `transformers >= 4.53` (merged March 2026). If you get an
`ImportError: cannot import name 'CHMv2ForDepthEstimation'`:

```bat
pip install -U transformers
```

---

## 5. Put your images in place

```
dinov3-aerial-seg\data\
    IMG_0001.JPG
    IMG_0002.JPG
    ...
```

Default source GSD is **3.52 cm/px** (SONY ILX-LR1 at 150 m AGL, 16 mm focal
length). If you fly at a different altitude or focal length, either:

- Pass `--source-gsd <meters_per_pixel>` on the command line, or
- Edit the default in `chmv2_infer.py` (argparse section).

Formula: `GSD_m_per_px = altitude_m * pixel_pitch_m / focal_length_m`.
Pixel pitch for ILX-LR1 = 3.76 µm = 3.76e-6 m.

---

## 6. Run inference

All commands below assume the activated env and a cwd of `dinov3-aerial-seg\`.

### 6a. Quick first test (recommended)

Single forward pass on a resized copy, longest side 2048 px:

```bat
python chmv2_infer.py data\IMG_0001.JPG
```

### 6b. Scale-matched test (closer to what CHMv2 was trained on)

Resize so each pixel = 20 cm on the ground (~5× finer than training, still
viable). Good middle ground for drone imagery:

```bat
python chmv2_infer.py data\IMG_0001.JPG --target-gsd 0.2
```

For the "fair" 1 m/px comparison (image will be tiny, model will upsize
internally):

```bat
python chmv2_infer.py data\IMG_0001.JPG --target-gsd 1.0
```

### 6c. Full-resolution tile mode

Slower; uses overlapping 1024×1024 tiles blended with a Hann window:

```bat
python chmv2_infer.py data\IMG_0001.JPG --mode tile --tile 1024 --overlap 128
```

You can combine `--target-gsd` with `--mode tile` to pre-resize then tile:

```bat
python chmv2_infer.py data\IMG_0001.JPG --mode tile --target-gsd 0.2
```

### 6d. Whole folder

```bat
python chmv2_infer.py data\ -o outputs\
```

### 6e. Compare runs side-by-side

Send each test to its own subfolder:

```bat
python chmv2_infer.py data\IMG_0001.JPG --target-gsd 1.0 -o outputs\gsd_1m\
python chmv2_infer.py data\IMG_0001.JPG --target-gsd 0.2 -o outputs\gsd_20cm\
python chmv2_infer.py data\IMG_0001.JPG --mode tile       -o outputs\tile_native\
```

---

## 7. Outputs

For each input `foo.jpg`, three files land in `outputs\`:

| File                 | Type                | Meaning                                                              |
|----------------------|---------------------|----------------------------------------------------------------------|
| `foo_height.npy`     | float32, H×W        | Canopy height in **meters**, full 9504×6336 resolution               |
| `foo_height_cm.tif`  | uint16, H×W         | Same data in **centimeters**, GIS-friendly, clipped at 655.35 m      |
| `foo_preview.png`    | colormapped PNG     | Visual check with colorbar (viridis, 0 → 99th-percentile height)     |

Load the raw heights in Python with `np.load("outputs/foo_height.npy")`.
Open the `.tif` in QGIS / ArcGIS / any image viewer.

---

## 8. Troubleshooting

- **`ImportError: CHMv2ForDepthEstimation`** — `pip install -U transformers`.
- **`torch.cuda.is_available()` is False** — wrong CUDA wheel. Uninstall and
  reinstall torch with the `cu124`/`cu126` index that matches `nvidia-smi`.
- **`CUDA out of memory`** in tile mode — lower `--tile` (e.g. `--tile 768`)
  or use `--target-gsd 0.2` to shrink the image first.
- **PIL "DecompressionBombError"** — already disabled in the script
  (`Image.MAX_IMAGE_PIXELS = None`).
- **Preview looks like flat noise / constant value** — this is expected when
  feeding CHMv2 imagery that's far from its training scale. Try
  `--target-gsd 1.0` (matches training) or build an orthomosaic first.
- **First run is slow / hangs** — the model is being downloaded from Hugging
  Face (~1.2 GB). It caches in `%USERPROFILE%\.cache\huggingface\hub\` for
  subsequent runs.

---

## 9. What next

Once the previews look reasonable, binary segmentation is one line:

```python
import numpy as np
height = np.load("outputs/IMG_0001_height.npy")
canopy_mask = height >= 2.0   # tweak threshold based on your scene
```

A `--threshold` flag can be added to `chmv2_infer.py` once we pick a sensible
cutoff from real outputs.
