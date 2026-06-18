"""CHMv2 canopy-height inference for aerial RGB JPGs.

Runs Meta's `facebook/dinov3-vitl16-chmv2-dpt-head` (DINOv3 Sat-L + DPT head)
on one image or a folder. Two modes and two ways to control input size:

  --mode downsample   single forward pass on a resized copy (fast; default)
  --mode tile         overlapping tiles blended with a Hann window (slower)

Resize controls (--target-gsd wins, and it is the default):

  --target-gsd M      physical control (DEFAULT 0.6): resample so each pixel
                      covers M m on the ground. 0.6 = CHMv2's native operating
                      GSD (Maxar Vivid2 ~0.597 m/px, paper sec 2.2/5) — match it
                      or the canopy-height predictions degrade. Uses
                      --source-gsd as the input's native GSD.
  --longest N         raw-pixel fallback (only if --target-gsd is None): resize
                      so longest side = N px. Downsample mode only. Default 2048.

--source-gsd defaults to a SONY ILX-LR1 @ 150 m AGL, 16 mm lens (3.52 cm/px) —
set it to the native GSD of YOUR input images.

Outputs per input `foo.jpg` into `--output`:
  foo_height.npy        float32 canopy height, meters, full image resolution
  foo_height_cm.tif     uint16 centimeters (GIS-friendly; clipped at 655.35 m)
  foo_preview.png       colormapped preview with colorbar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CHMv2ForDepthEstimation, CHMv2ImageProcessorFast

# 61 MP JPGs trip PIL's decompression-bomb check.
Image.MAX_IMAGE_PIXELS = None

MODEL_ID = "facebook/dinov3-vitl16-chmv2-dpt-head"


def load_model(device: str):
    processor = CHMv2ImageProcessorFast.from_pretrained(MODEL_ID)
    model = CHMv2ForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
    return model, processor


@torch.inference_mode()
def _predict(model, processor, img: Image.Image, device: str) -> np.ndarray:
    inputs = processor(images=img, return_tensors="pt").to(device)
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=dev_type == "cuda"):
        outputs = model(**inputs)
    depth = processor.post_process_depth_estimation(
        outputs, target_sizes=[(img.height, img.width)]
    )[0]["predicted_depth"]
    return depth.float().cpu().numpy()


def _resize_for_inference(
    img: Image.Image,
    source_gsd: float,
    target_gsd: float | None,
    longest: int | None,
) -> tuple[Image.Image, float]:
    w, h = img.size
    if target_gsd is not None:
        scale = source_gsd / target_gsd
    elif longest is not None:
        scale = longest / max(w, h)
    else:
        return img, 1.0
    if scale >= 1.0:
        return img, 1.0
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR), scale


def _upscale(pred: np.ndarray, h: int, w: int) -> np.ndarray:
    if pred.shape == (h, w):
        return pred
    t = torch.from_numpy(pred)[None, None]
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def _hann2d(h: int, w: int) -> np.ndarray:
    return np.clip(np.outer(np.hanning(h), np.hanning(w)), 1e-3, None).astype(np.float32)


def infer_tiled(
    model,
    processor,
    img: Image.Image,
    tile: int,
    overlap: int,
    device: str,
) -> np.ndarray:
    w, h = img.size
    stride = max(tile - overlap, 1)

    def _starts(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        xs = list(range(0, extent - tile + 1, stride))
        if xs[-1] != extent - tile:
            xs.append(extent - tile)
        return xs

    xs, ys = _starts(w), _starts(h)
    acc = np.zeros((h, w), dtype=np.float32)
    wacc = np.zeros((h, w), dtype=np.float32)
    total = len(xs) * len(ys)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            box = (x, y, min(x + tile, w), min(y + tile, h))
            crop = img.crop(box)
            pred = _predict(model, processor, crop, device)
            win = _hann2d(crop.height, crop.width)
            acc[y:y + crop.height, x:x + crop.width] += pred * win
            wacc[y:y + crop.height, x:x + crop.width] += win
            print(f"    tile {i * len(xs) + j + 1}/{total}")
    return acc / np.maximum(wacc, 1e-6)


def save_outputs(height: np.ndarray, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}_height.npy", height.astype(np.float32))

    tif = np.clip(height * 100.0, 0, 65535).astype(np.uint16)
    Image.fromarray(tif, mode="I;16").save(out_dir / f"{stem}_height_cm.tif")

    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    vmax = float(max(1.0, np.nanpercentile(height, 99)))
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(height, cmap=colormaps["viridis"], vmin=0, vmax=vmax)
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="canopy height (m)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_preview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _gather(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="image file or folder")
    p.add_argument("-o", "--output", type=Path, default=Path("outputs"))
    p.add_argument("--mode", choices=["downsample", "tile"], default="downsample")
    p.add_argument("--longest", type=int, default=2048,
                   help="[downsample] longest side in px. Ignored if --target-gsd is set.")
    p.add_argument("--source-gsd", type=float, default=0.0352,
                   help="native GSD of input images in meters/pixel (default: 0.0352 = ILX-LR1 @ 150m AGL, 16mm)")
    p.add_argument("--target-gsd", type=float, default=0.6,
                   help="resample input to this GSD (m/px) before inference. Default 0.6 = CHMv2's native "
                        "operating GSD (Maxar Vivid2 ~0.597 m/px, paper sec 2.2/5); feeding a different scale "
                        "degrades the canopy-height predictions. Set None to fall back to --longest.")
    p.add_argument("--tile", type=int, default=1024, help="[tile] tile size in px")
    p.add_argument("--overlap", type=int, default=128, help="[tile] overlap in px")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    paths = _gather(args.input)
    if not paths:
        raise SystemExit(f"no images found in {args.input}")

    longest = args.longest if args.target_gsd is None and args.mode == "downsample" else None
    resize_note = (
        f"target-gsd={args.target_gsd} m/px (source={args.source_gsd} m/px)"
        if args.target_gsd is not None
        else (f"longest={args.longest} px" if args.mode == "downsample" else "no pre-resize (tile at native)")
    )
    print(f"device={args.device}  mode={args.mode}  {resize_note}  images={len(paths)}")
    model, processor = load_model(args.device)

    for path in paths:
        print(f"-> {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size

        img_in, scale = _resize_for_inference(img, args.source_gsd, args.target_gsd, longest)
        if scale < 1.0:
            eff_gsd = args.source_gsd / scale
            print(f"   resized {orig_w}x{orig_h} -> {img_in.size[0]}x{img_in.size[1]}  "
                  f"(effective GSD ~ {eff_gsd * 100:.2f} cm/px)")

        if args.mode == "downsample":
            pred = _predict(model, processor, img_in, args.device)
        else:
            pred = infer_tiled(model, processor, img_in, args.tile, args.overlap, args.device)

        height = _upscale(pred, orig_h, orig_w)
        save_outputs(height, args.output, path.stem)
        print(f"   saved -> {args.output}/{path.stem}_*  (min={height.min():.2f}m  max={height.max():.2f}m)")


if __name__ == "__main__":
    main()
