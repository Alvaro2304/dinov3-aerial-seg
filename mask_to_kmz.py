"""Build a KMZ of canopy-mask overlays from CHMv2 height rasters.

Inputs:
  * folder of *_height.npy files (produced by chmv2_infer.py)
  * flight metadata file with columns:
        name, latitude, longitude, drone_roll, drone_pitch, drone_yaw,
        current_height
    (semicolon-separated, with `<deg>° S/N/E/W` style coordinates)

For each frame:
  1. Threshold the height raster to a binary canopy mask
  2. Render the mask as a transparent RGBA PNG (canopy = colored, else clear)
  3. Project the four image corners to lat/lon using the project's geo math
     (R_mount = Rz(-90°), ZYX Euler, +X right / +Y down / +Z forward)
  4. Add a <GroundOverlay> with <gx:LatLonQuad> to the KML

By default assumes the height rasters came from RECTIFIED frames (so roll
and pitch are zero in the corner projection; only yaw is applied). Use
--unrectified if the inputs were inferred from raw tilted frames.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageOps

PHOTO_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG")

SENSOR_W_MM = 35.7
SENSOR_H_MM = 23.8
DEFAULT_FOCAL_MM = 16.0
EARTH_M_PER_DEG_LAT = 111_320.0

R_MOUNT = np.array([[0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]])


def parse_coord(s: str) -> float:
    """`'13.343707° S'` -> -13.343707. Robust to any non-ASCII garbage
    around the number and to ° encoding mishaps."""
    m = re.search(r"[+-]?\d+\.?\d*", s)
    if not m:
        raise ValueError(f"no number in {s!r}")
    val = float(m.group())
    last = s.strip()[-1].upper() if s.strip() else ""
    if last in ("S", "W"):
        val = -abs(val)
    elif last in ("N", "E"):
        val = abs(val)
    return val


def _rot(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project_pixel_to_latlon(
    px: float, py: float, w: int, h: int, focal_mm: float,
    drone_lat: float, drone_lon: float, agl_m: float,
    roll_deg: float, pitch_deg: float, yaw_deg: float,
) -> tuple[float, float] | None:
    u_mm = (px - w / 2) * SENSOR_W_MM / w
    v_mm = (py - h / 2) * SENSOR_H_MM / h
    ray_cam = np.array([u_mm, v_mm, focal_mm])
    ray_body = R_MOUNT @ ray_cam
    R_att = _rot(np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg))
    ray_ned = R_att @ ray_body
    if ray_ned[2] < 1e-3:
        return None
    t = agl_m / ray_ned[2]
    delta_n = t * ray_ned[0]
    delta_e = t * ray_ned[1]
    d_lat = delta_n / EARTH_M_PER_DEG_LAT
    d_lon = delta_e / (EARTH_M_PER_DEG_LAT * np.cos(np.radians(drone_lat)))
    return drone_lat + d_lat, drone_lon + d_lon


def height_to_rgba(height: np.ndarray, threshold: float,
                   color: tuple[int, int, int, int]) -> Image.Image:
    mask = height >= threshold
    rgba = np.zeros((*height.shape, 4), dtype=np.uint8)
    rgba[mask] = color
    return Image.fromarray(rgba, mode="RGBA")


def height_to_heatmap(height: np.ndarray, vmax: float,
                      alpha_threshold: float, alpha: int = 200,
                      cmap_name: str = "viridis") -> Image.Image:
    norm = np.clip(np.nan_to_num(height) / max(vmax, 1e-3), 0.0, 1.0)
    cmap = colormaps[cmap_name]
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(height < alpha_threshold, 0, alpha).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def photo_to_overlay_bytes(path: Path, max_edge: int, dark_threshold: int,
                           jpeg_quality: int) -> tuple[bytes, str]:
    """Load a JPG, drop near-black borders to alpha=0, return (bytes, suffix).

    Falls back to JPEG when the image has no dark borders (smaller file)."""
    img = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    img = downsample(img, max_edge)
    arr = np.array(img)
    transparent = arr.max(axis=2) <= dark_threshold
    if not transparent.any():
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=jpeg_quality, optimize=True)
        return buf.getvalue(), "jpg"
    alpha = np.where(transparent, 0, 255).astype(np.uint8)
    rgba = np.dstack([arr, alpha])
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue(), "png"


def downsample(img: Image.Image, max_edge: int) -> Image.Image:
    if max(img.size) <= max_edge:
        return img
    scale = max_edge / max(img.size)
    new_size = (max(1, round(img.size[0] * scale)),
                max(1, round(img.size[1] * scale)))
    return img.resize(new_size, Image.NEAREST)


def read_metadata(path: Path) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            meta[row["name"]] = {
                "lat": parse_coord(row["latitude"]),
                "lon": parse_coord(row["longitude"]),
                "roll": float(row["drone_roll"]),
                "pitch": float(row["drone_pitch"]),
                "yaw": float(row["drone_yaw"]),
                "h": float(row["current_height"]),
            }
    return meta


def find_match(stem: str, meta: dict[str, dict]) -> str | None:
    for name in meta:
        if name.split(".")[0].lower() == stem.lower():
            return name
    return None


def find_photo(stem: str, photos_dir: Path) -> Path | None:
    for ext in PHOTO_EXTS:
        p = photos_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _overlay_xml(name: str, href: str, coords: str, draw_order: int) -> str:
    return (
        "    <GroundOverlay>\n"
        f"      <name>{escape(name)}</name>\n"
        f"      <drawOrder>{draw_order}</drawOrder>\n"
        f"      <Icon><href>{href}</href></Icon>\n"
        "      <gx:LatLonQuad>\n"
        f"        <coordinates>{coords}</coordinates>\n"
        "      </gx:LatLonQuad>\n"
        "    </GroundOverlay>"
    )


def _folder_xml(name: str, visible: bool, overlays: list[str]) -> list[str]:
    return [
        "  <Folder>",
        f"    <name>{escape(name)}</name>",
        f"    <visibility>{1 if visible else 0}</visibility>",
        "    <open>0</open>",
        *overlays,
        "  </Folder>",
    ]


def build_kmz(
    inputs_dir: Path, metadata_path: Path, out_path: Path,
    threshold: float, max_edge: int, color: tuple[int, int, int, int],
    rectified: bool, focal_mm: float,
    photos_dir: Path | None = None,
    include_heights: bool = True,
    include_mask: bool = True,
    photo_quality: int = 85,
    dark_threshold: int = 5,
    height_vmax: float = 15.0,
    height_alpha_threshold: float = 1.0,
) -> None:
    meta = read_metadata(metadata_path)
    height_files = sorted(inputs_dir.glob("*_height.npy"))
    if not height_files:
        raise SystemExit(f"no *_height.npy in {inputs_dir}")

    print(f"frames: {len(height_files)}  threshold: {threshold} m  rectified: {rectified}")
    print(f"layers: photos={'yes (' + str(photos_dir) + ')' if photos_dir else 'no'}  "
          f"heights={include_heights}  mask={include_mask}")

    photo_overlays: list[str] = []
    height_overlays: list[str] = []
    mask_overlays: list[str] = []
    n_skip = 0

    with ZipFile(out_path, "w", ZIP_DEFLATED) as kmz:
        for hpath in height_files:
            stem = hpath.stem[:-len("_height")]
            jpg_name = find_match(stem, meta)
            if jpg_name is None:
                print(f"   skip {stem}: not in metadata")
                n_skip += 1
                continue
            m = meta[jpg_name]

            height = np.load(hpath)
            H, W = height.shape

            roll = 0.0 if rectified else m["roll"]
            pitch = 0.0 if rectified else m["pitch"]

            corner_px = [(0, H), (W, H), (W, 0), (0, 0)]  # LL, LR, UR, UL
            corners_ll = []
            for cx, cy in corner_px:
                pt = project_pixel_to_latlon(
                    cx, cy, W, H, focal_mm,
                    m["lat"], m["lon"], m["h"],
                    roll, pitch, m["yaw"],
                )
                if pt is None:
                    break
                corners_ll.append(pt)
            if len(corners_ll) != 4:
                print(f"   skip {stem}: ray misses ground")
                n_skip += 1
                continue
            coords = " ".join(f"{lon:.7f},{lat:.7f},0" for (lat, lon) in corners_ll)

            if photos_dir is not None:
                photo_path = find_photo(stem, photos_dir)
                if photo_path is not None:
                    blob, ext = photo_to_overlay_bytes(
                        photo_path, max_edge, dark_threshold, photo_quality,
                    )
                    href = f"photos/{stem}.{ext}"
                    kmz.writestr(href, blob)
                    photo_overlays.append(_overlay_xml(stem, href, coords, draw_order=0))

            if include_heights:
                heatmap = downsample(
                    height_to_heatmap(height, height_vmax, height_alpha_threshold),
                    max_edge,
                )
                buf = io.BytesIO()
                heatmap.save(buf, "PNG", optimize=True)
                href = f"heights/{stem}.png"
                kmz.writestr(href, buf.getvalue())
                height_overlays.append(_overlay_xml(stem, href, coords, draw_order=1))

            if include_mask:
                mask = downsample(height_to_rgba(height, threshold, color), max_edge)
                buf = io.BytesIO()
                mask.save(buf, "PNG", optimize=True)
                href = f"masks/{stem}.png"
                kmz.writestr(href, buf.getvalue())
                mask_overlays.append(_overlay_xml(stem, href, coords, draw_order=2))

            n_added_so_far = max(len(photo_overlays), len(height_overlays), len(mask_overlays))
            if n_added_so_far and n_added_so_far % 50 == 0:
                print(f"   ... {n_added_so_far} frames")

        kml = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2" '
            'xmlns:gx="http://www.google.com/kml/ext/2.2">',
            "<Document>",
            f"  <name>Canopy survey (threshold {threshold} m)</name>",
        ]
        if photo_overlays:
            kml.extend(_folder_xml("Photos (reference)", True, photo_overlays))
        if height_overlays:
            kml.extend(_folder_xml("CHMv2 height", False, height_overlays))
        if mask_overlays:
            kml.extend(_folder_xml(f"Canopy mask (>= {threshold} m)", True, mask_overlays))
        kml.extend(["</Document>", "</kml>"])
        kmz.writestr("doc.kml", "\n".join(kml))

    size_mb = out_path.stat().st_size / 1e6
    counts = []
    if photo_overlays:
        counts.append(f"{len(photo_overlays)} photos")
    if height_overlays:
        counts.append(f"{len(height_overlays)} heights")
    if mask_overlays:
        counts.append(f"{len(mask_overlays)} masks")
    print(f"wrote {out_path}  ({', '.join(counts) or 'nothing'}, "
          f"{n_skip} skipped, {size_mb:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", type=Path,
                   help="folder containing *_height.npy from chmv2_infer.py")
    p.add_argument("metadata", type=Path,
                   help="flight metadata file (CSV/TSV/TXT, semicolon-separated)")
    p.add_argument("-o", "--output", type=Path, default=Path("canopy.kmz"))
    p.add_argument("--threshold", type=float, default=2.0,
                   help="canopy if height >= this (meters); default 2.0")
    p.add_argument("--max-edge", type=int, default=1024,
                   help="downsample mask PNGs to this max side length")
    p.add_argument("--color", default="0,200,0,160",
                   help="canopy RGBA color, comma-separated (default '0,200,0,160')")
    p.add_argument("--focal-mm", type=float, default=DEFAULT_FOCAL_MM)
    p.add_argument("--photos-dir", type=Path, default=None,
                   help="folder with reference JPGs to include as a layer "
                        "(usually data_rectified/ if you ran inference on rectified frames)")
    p.add_argument("--no-heights", dest="include_heights", action="store_false",
                   help="omit the CHMv2 colormapped height layer")
    p.add_argument("--no-mask", dest="include_mask", action="store_false",
                   help="omit the canopy mask layer")
    p.add_argument("--photo-quality", type=int, default=85,
                   help="JPEG quality for photo overlays (default 85)")
    p.add_argument("--dark-threshold", type=int, default=5,
                   help="photo pixels with max(R,G,B) <= this become transparent "
                        "(handles rectified-frame black borders); default 5")
    p.add_argument("--height-vmax", type=float, default=15.0,
                   help="upper bound for the height heatmap colormap, in meters; default 15")
    p.add_argument("--height-alpha-threshold", type=float, default=1.0,
                   help="height heatmap pixels < this (m) are transparent; default 1.0")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--rectified", dest="rectified", action="store_true", default=True,
                   help="height rasters come from rectified frames (default)")
    g.add_argument("--unrectified", dest="rectified", action="store_false",
                   help="height rasters come from raw tilted frames")
    args = p.parse_args()

    color = tuple(int(x) for x in args.color.split(","))
    if len(color) != 4:
        raise SystemExit("--color must be 4 ints (R,G,B,A)")

    build_kmz(
        args.inputs, args.metadata, args.output,
        args.threshold, args.max_edge, color,
        args.rectified, args.focal_mm,
        photos_dir=args.photos_dir,
        include_heights=args.include_heights,
        include_mask=args.include_mask,
        photo_quality=args.photo_quality,
        dark_threshold=args.dark_threshold,
        height_vmax=args.height_vmax,
        height_alpha_threshold=args.height_alpha_threshold,
    )


if __name__ == "__main__":
    main()
