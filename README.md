# dinov3-aerial-seg

First-pass inference with Meta's **CHMv2** canopy-height model
(`facebook/dinov3-vitl16-chmv2-dpt-head`, DINOv3 Sat-L backbone + DPT head)
on aerial RGB JPGs from a SONY ILX-LR1. Predicts per-pixel canopy height in
meters; a binary canopy mask can be obtained later by thresholding.

> **Shell used in this guide:** Git Bash (MINGW64) on Windows. All commands
> below assume `bash`. If you use `cmd.exe` or PowerShell instead, swap
> forward slashes for backslashes in paths and activate the venv with
> `.venv\Scripts\activate` (no `source`). See the note in section 2.

---

## 0. Prerequisites (Windows PC)

- Windows 10/11 with an NVIDIA GPU (tested target: RTX 4080 Super, 16 GB VRAM).
- Up-to-date NVIDIA driver (R550+ is enough for CUDA 12.4).
  Check: `nvidia-smi` should print the GPU name and a CUDA version.
- Python **3.10 or 3.11** installed and on PATH. Grab from
  https://www.python.org/downloads/windows/ — *not* the Microsoft Store build.
- Git for Windows (ships with Git Bash): https://git-scm.com/download/win
- ~10 GB free disk (model weights are ~1.2 GB; torch CUDA wheels are large).

No Anaconda is needed — `venv` + `pip` is enough. If you prefer conda, install
**Miniconda** (not full Anaconda): https://www.anaconda.com/docs/getting-started/miniconda/install

---

## 1. Get the code

```bash
cd /c/Users/<you>/Documents          # Git Bash path style; cmd: cd C:\Users\<you>\Documents
git clone <your-remote-url> dinov3-aerial-seg
cd dinov3-aerial-seg
```

Folder layout after cloning:

```
dinov3-aerial-seg/
├── chmv2_infer.py        main inference script
├── requirements.txt
├── data/                 put your input JPGs here
└── outputs/              results go here
```

---

## 2. Create a Python environment

### Option A — venv (recommended)

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash
# cmd.exe:    .venv\Scripts\activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Option B — Miniconda

```bash
conda create -n chmv2 python=3.11 -y
conda activate chmv2
python -m pip install --upgrade pip
```

> If `conda activate` fails in Git Bash, run `conda init bash` once, then
> close and reopen the terminal.

Your shell prompt should now start with `(.venv)` or `(chmv2)`.

---

## 3. Install PyTorch with CUDA

The `CUDA Version: 13.1` that `nvidia-smi` prints is the **maximum** CUDA the
driver supports — not an installed toolkit. Any PyTorch CUDA wheel whose
version is ≤ that number will work. Current stable PyTorch ships wheels for
**CUDA 11.8 / 12.6 / 12.8**. Recommended: **cu128**.

```bash
# Recommended for RTX 4080 Super, driver >= R550
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# Older alternative if cu128 has issues
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```

Verify CUDA is visible to PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True NVIDIA GeForce RTX 4080 SUPER` (or similar).
If it prints `False`, the driver/CUDA/torch versions don't match — reinstall
torch with the correct `cuXXX` index.

---

## 4. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

That pulls `transformers`, `Pillow`, `numpy`, `matplotlib`. CHMv2 support
requires `transformers >= 4.53` (merged March 2026). If you get an
`ImportError: cannot import name 'CHMv2ForDepthEstimation'`:

```bash
pip install -U transformers
```

---

## 4b. Authenticate with Hugging Face (one-time, required)

The CHMv2 weights are **gated** under Meta's DINOv3 license. You must accept
the terms in your HF account before the model can be downloaded.

1. Sign in / sign up at https://huggingface.co
2. Open https://huggingface.co/facebook/dinov3-vitl16-chmv2-dpt-head and
   click **Agree and access** (you'll get an email confirmation).
3. Create a read token: https://huggingface.co/settings/tokens → *New token*
   → select *Read* role → copy the token.
4. Log in on this machine:

   ```bash
   pip install huggingface_hub
   hf auth login
   # paste the token; answer "n" to the git-credential prompt
   ```

   (Older versions: `huggingface-cli login` — same thing.) Token is cached
   at `~/.cache/huggingface/token`.

From then on, `from_pretrained(...)` downloads the weights once into
`~/.cache/huggingface/hub/` and every subsequent run is offline-fast.

---

## 5. Put your images in place

```
dinov3-aerial-seg/data/
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

All commands below assume the activated env and a cwd of `dinov3-aerial-seg/`.

### 6a. Quick first test (recommended)

Single forward pass on a resized copy, longest side 2048 px:

```bash
python chmv2_infer.py data/IMG_0001.JPG
```

### 6b. Scale-matched test (closer to what CHMv2 was trained on)

Resize so each pixel = 20 cm on the ground (~5× finer than training, still
viable). Good middle ground for drone imagery:

```bash
python chmv2_infer.py data/IMG_0001.JPG --target-gsd 0.2
```

For the "fair" 1 m/px comparison (image will be tiny, model upsizes
internally):

```bash
python chmv2_infer.py data/IMG_0001.JPG --target-gsd 1.0
```

### 6c. Full-resolution tile mode

Slower; uses overlapping 1024×1024 tiles blended with a Hann window:

```bash
python chmv2_infer.py data/IMG_0001.JPG --mode tile --tile 1024 --overlap 128
```

Combine `--target-gsd` with `--mode tile` to pre-resize then tile:

```bash
python chmv2_infer.py data/IMG_0001.JPG --mode tile --target-gsd 0.2
```

### 6d. Whole folder

```bash
python chmv2_infer.py data/ -o outputs/
```

### 6e. Compare runs side-by-side

Send each test to its own subfolder:

```bash
python chmv2_infer.py data/IMG_0001.JPG --target-gsd 1.0 -o outputs/gsd_1m/
python chmv2_infer.py data/IMG_0001.JPG --target-gsd 0.2 -o outputs/gsd_20cm/
python chmv2_infer.py data/IMG_0001.JPG --mode tile       -o outputs/tile_native/
```

---

## 7. Outputs

For each input `foo.jpg`, three files land in `outputs/`:

| File                 | Type                | Meaning                                                              |
|----------------------|---------------------|----------------------------------------------------------------------|
| `foo_height.npy`     | float32, H×W        | Canopy height in **meters**, full 9504×6336 resolution               |
| `foo_height_cm.tif`  | uint16, H×W         | Same data in **centimeters**, GIS-friendly, clipped at 655.35 m      |
| `foo_preview.png`    | colormapped PNG     | Visual check with colorbar (viridis, 0 → 99th-percentile height)     |

Load the raw heights in Python with `np.load("outputs/foo_height.npy")`.
Open the `.tif` in QGIS / ArcGIS / any image viewer.

---

## 8. Troubleshooting

- **`bash: .venvScriptsactivate: command not found`** — Git Bash eats the
  backslashes. Use `source .venv/Scripts/activate` instead.
- **`401 Unauthorized` / `403 Forbidden` when loading the model** — you
  haven't accepted the gated terms or aren't logged in. Redo section 4b.
- **`ImportError: CHMv2ForDepthEstimation`** — `pip install -U transformers`.
- **`torch.cuda.is_available()` is False** — wrong CUDA wheel. Uninstall and
  reinstall torch with a `cu126`/`cu128` index supported by your driver.
- **`CUDA out of memory`** in tile mode — lower `--tile` (e.g. `--tile 768`)
  or use `--target-gsd 0.2` to shrink the image first.
- **PIL "DecompressionBombError"** — already disabled in the script
  (`Image.MAX_IMAGE_PIXELS = None`).
- **Preview looks like flat noise / constant value** — expected when feeding
  CHMv2 imagery that's far from its training scale. Try `--target-gsd 1.0`
  (matches training) or build an orthomosaic first.
- **First run is slow / hangs** — the model is being downloaded from Hugging
  Face (~1.2 GB). It caches in `~/.cache/huggingface/hub/` (Git Bash) or
  `%USERPROFILE%\.cache\huggingface\hub\` (cmd) for subsequent runs.

---

## 9. Optional: rectify tilted frames before inference

The VTOL has no gimbal, so each JPG inherits the airframe's roll/pitch.
CHMv2 was trained on near-nadir satellite imagery, so a 10°+ roll degrades
absolute heights significantly. `rectify.py` applies a pure-rotation
homography that warps each tilted frame to a virtual nadir view, using
per-image attitude from your flight CSV.

Conventions match `geo_inference/extra/docs/geo_projection.html`
(R_mount = Rz(−90°), ZYX Euler, +X right / +Y down / +Z forward).

```bash
# rectify all JPGs in data/ → data_rectified/
python rectify.py data/ flight.csv -o data_rectified/

# only rectify mildly tilted frames (skip extreme cases)
python rectify.py data/ flight.csv --max-tilt-deg 20

# then run CHMv2 on the rectified images
python chmv2_infer.py data_rectified/IMG_0001.JPG --target-gsd 0.2
```

The flight CSV must be semicolon-separated with columns including
`name`, `drone_roll`, `drone_pitch`, `drone_yaw`, `current_height` (matches
the format produced by your VTOL's metadata exporter).

What rectification fixes:
- Tilt-induced perspective drift in CHMv2 absolute heights.

What it does NOT fix:
- Parallax (tall trees still lean outward at frame edges — needs multi-view).
- Variable AGL between frames (±10 m altitude → ~7 % GSD variation).
- Fundamental scale mismatch with CHMv2's ~1 m/px training data.

For trustworthy absolute heights, build an orthomosaic (WebODM/Metashape)
from the survey and run inference on the mosaic at `--target-gsd 1.0`.

---

## 10. Build a KMZ for Google Earth

After running CHMv2 across a flight, package the per‑frame canopy masks
into a single KMZ for visual QA along the corridor.

```bash
# mask only (smallest file)
python mask_to_kmz.py outputs/ data/flight.txt -o canopy.kmz --threshold 2.0

# add the original photos as a reference layer (recommended)
python mask_to_kmz.py outputs/ data/flight.txt -o canopy.kmz \
    --threshold 2.0 --photos-dir data_rectified/

# everything: photos + height heatmap + mask
python mask_to_kmz.py outputs/ data/flight.txt -o canopy.kmz \
    --threshold 2.0 --photos-dir data_rectified/ --max-edge 1024
```

The KMZ contains up to three toggleable layers (Google Earth sidebar):

- **Photos (reference)** — visible by default. Source JPGs from `--photos-dir`.
- **CHMv2 height** — hidden by default. Colormapped preview from `*_preview.png`.
- **Canopy mask (>= T m)** — visible by default, drawn on top of photos.

What it does, per `*_height.npy` in `outputs/`:

1. Threshold to a binary canopy mask (default `>= 2.0 m`).
2. Render as a transparent PNG (canopy = green semi‑transparent, else clear).
3. Project the four image corners to lat/lon using `R_mount = Rz(-90°)`,
   ZYX Euler attitude, and the metadata GPS + AGL.
4. Add a `<GroundOverlay>` with `<gx:LatLonQuad>` (handles rotated frames).
5. Zip everything into `canopy.kmz`.

Flags worth knowing:

- `--threshold M` — height ≥ this in meters is canopy. Default 2.0.
- `--color R,G,B,A` — overlay color. Default `0,200,0,160` (semi‑green).
- `--max-edge N` — downsample overlay images to this max side. Default 1024.
- `--height-vmax M` — upper bound (meters) for the height heatmap colormap.
  Default 15.0. Pixels above this saturate yellow; pixels below
  `--height-alpha-threshold` (default 1.0) are transparent.
- `--dark-threshold N` — photo pixels with `max(R,G,B) <= N` become fully
  transparent (handles black borders left by `rectify.py`). Default 5.
  Photos without dark pixels are saved as JPEG (smaller); photos with
  dark borders are saved as PNG with an alpha channel.
- `--photo-quality Q` — JPEG quality 0–100. Default 85.
- `--unrectified` — pass this if your height rasters came from raw tilted
  frames; otherwise the script assumes you ran inference on rectified
  outputs (default workflow).

**Where to view:**

- **Google Earth (desktop / web / mobile):** drag `canopy.kmz` into Earth.
  All overlays render correctly, including rotated quads.
- **Google Maps consumer site:** does not load KMZ directly anymore.
- **Google My Maps (`mymaps.google.com`):** can import KMZ as a custom
  layer, but limited to **5 MB / 2000 features / 10 layers per map**.
  For a 200 km corridor with ~1000 frames you will exceed those limits —
  use Google Earth as the primary viewer.

---

## 11. What next

Once the previews look reasonable, binary segmentation is one line:

```python
import numpy as np
height = np.load("outputs/IMG_0001_height.npy")
canopy_mask = height >= 2.0   # tweak threshold based on your scene
```

A `--threshold` flag can be added to `chmv2_infer.py` once we pick a sensible
cutoff from real outputs.
