"""Telemetry-driven height mosaic for canopy mapping.

For each chunk of N kilometers of flight:
  1. Allocate a local lat/lon raster at the chosen GSD (default 1 m/px).
  2. Warp every frame's CHMv2 height map onto that raster using the
     4-corner homography (cv2.warpPerspective), weighted by a Hann window
     centered on the frame so parallax-prone edges contribute less.
  3. Accumulate sum(h * w) and sum(w); take their ratio for the mean.
  4. Threshold once on the merged raster -> single binary canopy mask.

Outputs:
  <out>/chunk_NNN.kmz   one self-contained KMZ per chunk
  <out>/all_chunks.kmz  combined KMZ holding every chunk overlay

The combined KMZ has two toggleable layers (Heights, Mask) the same way
mask_to_kmz.py does. No more per-frame stacking — overlap zones are
averaged into a single coherent surface.
"""
from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
from PIL import Image

from mask_to_kmz import (
    DEFAULT_FOCAL_MM,
    EARTH_M_PER_DEG_LAT,
    height_to_heatmap,
    project_pixel_to_latlon,
    read_metadata,
)

DEFAULT_CHUNK_KM = 10.0
DEFAULT_GSD_M = 1.0
HANN_FLOOR = 1e-2  # avoid hard zeros at frame edges


# ─────────────────────────────────────────────────────────────────────────────
# Geo helpers
# ─────────────────────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def latlon_to_local_xy(lat: float, lon: float, lat0: float, lon0: float, gsd_m: float
                       ) -> tuple[float, float]:
    """Local NE plane -> raster pixel coords (x = east, y = south, image-style)."""
    dn = (lat - lat0) * EARTH_M_PER_DEG_LAT
    de = (lon - lon0) * EARTH_M_PER_DEG_LAT * math.cos(math.radians(lat0))
    return de / gsd_m, -dn / gsd_m


def raster_bounds_to_latlon(x_min: int, x_max: int, y_min: int, y_max: int,
                            lat0: float, lon0: float, gsd_m: float
                            ) -> tuple[float, float, float, float]:
    """Convert raster pixel bounds back to (north, south, east, west)."""
    coslat = math.cos(math.radians(lat0))
    north = lat0 - y_min * gsd_m / EARTH_M_PER_DEG_LAT
    south = lat0 - y_max * gsd_m / EARTH_M_PER_DEG_LAT
    west = lon0 + x_min * gsd_m / (EARTH_M_PER_DEG_LAT * coslat)
    east = lon0 + x_max * gsd_m / (EARTH_M_PER_DEG_LAT * coslat)
    return north, south, east, west


# ─────────────────────────────────────────────────────────────────────────────
# Frame footprint and chunking
# ─────────────────────────────────────────────────────────────────────────────

def frame_corners_latlon(W: int, H: int, focal_mm: float, m: dict, rectified: bool
                         ) -> list[tuple[float, float]] | None:
    roll = 0.0 if rectified else m["roll"]
    pitch = 0.0 if rectified else m["pitch"]
    out = []
    for cx, cy in [(0, H), (W, H), (W, 0), (0, 0)]:  # LL, LR, UR, UL
        pt = project_pixel_to_latlon(
            cx, cy, W, H, focal_mm, m["lat"], m["lon"], m["h"],
            roll, pitch, m["yaw"],
        )
        if pt is None:
            return None
        out.append(pt)
    return out


def chunk_by_distance(frame_meta: list[tuple[Path, dict]], chunk_m: float
                      ) -> list[list[tuple[Path, dict]]]:
    chunks: list[list[tuple[Path, dict]]] = [[]]
    cumulative = 0.0
    prev = None
    for entry in frame_meta:
        _, m = entry
        if prev is not None:
            cumulative += haversine_m(prev["lat"], prev["lon"], m["lat"], m["lon"])
            if cumulative >= chunk_m:
                chunks.append([])
                cumulative = 0.0
        chunks[-1].append(entry)
        prev = m
    return [c for c in chunks if c]


# ─────────────────────────────────────────────────────────────────────────────
# Mosaicking
# ─────────────────────────────────────────────────────────────────────────────

def hann2d(h: int, w: int) -> np.ndarray:
    return np.clip(np.outer(np.hanning(h), np.hanning(w)),
                   HANN_FLOOR, None).astype(np.float32)


def mosaic_chunk(chunk: list[tuple[Path, dict]], inputs_dir: Path,
                 gsd_m: float, focal_mm: float, rectified: bool,
                 max_pixels: int) -> dict | None:
    """Return {height, weight, north, south, east, west, n_frames} or None."""
    frames = []
    for hpath, m in chunk:
        height = np.load(hpath)
        H, W = height.shape
        corners = frame_corners_latlon(W, H, focal_mm, m, rectified)
        if corners is None:
            continue
        frames.append((height, corners))
    if not frames:
        return None

    all_lats = [lat for _, cs in frames for lat, _ in cs]
    all_lons = [lon for _, cs in frames for _, lon in cs]
    lat0 = (min(all_lats) + max(all_lats)) / 2
    lon0 = (min(all_lons) + max(all_lons)) / 2

    pts_xy = [latlon_to_local_xy(lat, lon, lat0, lon0, gsd_m)
              for _, cs in frames for lat, lon in cs]
    xs, ys = zip(*pts_xy)
    x_min, x_max = math.floor(min(xs)) - 1, math.ceil(max(xs)) + 1
    y_min, y_max = math.floor(min(ys)) - 1, math.ceil(max(ys)) + 1
    world_w = x_max - x_min
    world_h = y_max - y_min

    if world_w * world_h > max_pixels:
        print(f"   skip chunk: bbox {world_w}x{world_h}={world_w * world_h} px "
              f"> --max-pixels {max_pixels}; reduce --chunk-km or raise --max-pixels")
        return None

    sum_h = np.zeros((world_h, world_w), dtype=np.float32)
    sum_w = np.zeros((world_h, world_w), dtype=np.float32)

    for height, corners in frames:
        Hi, Wi = height.shape
        src = np.array([[0, Hi], [Wi, Hi], [Wi, 0], [0, 0]], dtype=np.float32)
        dst_xy = [latlon_to_local_xy(lat, lon, lat0, lon0, gsd_m) for lat, lon in corners]
        dst = np.array([(x - x_min, y - y_min) for x, y in dst_xy], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped_h = cv2.warpPerspective(height.astype(np.float32), M, (world_w, world_h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        warped_w = cv2.warpPerspective(hann2d(Hi, Wi), M, (world_w, world_h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        sum_h += warped_h * warped_w
        sum_w += warped_w

    mean_h = np.where(sum_w > 1e-3, sum_h / np.maximum(sum_w, 1e-3), 0.0).astype(np.float32)
    n, s, e, w = raster_bounds_to_latlon(x_min, x_max, y_min, y_max, lat0, lon0, gsd_m)
    return {
        "height": mean_h, "weight": sum_w,
        "north": n, "south": s, "east": e, "west": w,
        "n_frames": len(frames), "size": (world_w, world_h),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_mask(mosaic: dict, threshold: float, color: tuple[int, int, int, int]
                ) -> Image.Image:
    h = mosaic["height"]
    canopy = h >= threshold
    no_data = mosaic["weight"] < 1e-3
    rgba = np.zeros((*h.shape, 4), dtype=np.uint8)
    rgba[canopy] = color
    rgba[no_data, 3] = 0
    return Image.fromarray(rgba, mode="RGBA")


def render_height(mosaic: dict, vmax: float, alpha_threshold: float,
                  alpha: int = 200) -> Image.Image:
    img = height_to_heatmap(mosaic["height"], vmax, alpha_threshold, alpha)
    arr = np.array(img)
    arr[mosaic["weight"] < 1e-3, 3] = 0
    return Image.fromarray(arr, mode="RGBA")


# ─────────────────────────────────────────────────────────────────────────────
# KMZ output
# ─────────────────────────────────────────────────────────────────────────────

def _ground_overlay_xml(href: str, draw_order: int, n: float, s: float, e: float, w: float
                        ) -> list[str]:
    return [
        "    <GroundOverlay>",
        f"      <drawOrder>{draw_order}</drawOrder>",
        f"      <Icon><href>{href}</href></Icon>",
        "      <LatLonBox>",
        f"        <north>{n:.7f}</north><south>{s:.7f}</south>",
        f"        <east>{e:.7f}</east><west>{w:.7f}</west>",
        "      </LatLonBox>",
        "    </GroundOverlay>",
    ]


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def write_chunk_kmz(mosaic: dict, out_path: Path, threshold: float,
                    color: tuple[int, int, int, int], vmax: float,
                    alpha_threshold: float, chunk_id: str,
                    include_mask: bool, include_heights: bool) -> None:
    n, s, e, w = mosaic["north"], mosaic["south"], mosaic["east"], mosaic["west"]
    with ZipFile(out_path, "w", ZIP_DEFLATED) as kmz:
        kml = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            "<Document>",
            f"  <name>{escape(f'Canopy mosaic chunk {chunk_id}')}</name>",
        ]
        if include_heights:
            kmz.writestr("height.png", _png_bytes(render_height(mosaic, vmax, alpha_threshold)))
            kml.extend([
                "  <Folder>",
                "    <name>CHMv2 height</name>",
                "    <visibility>0</visibility>",
                *_ground_overlay_xml("height.png", 1, n, s, e, w),
                "  </Folder>",
            ])
        if include_mask:
            kmz.writestr("mask.png", _png_bytes(render_mask(mosaic, threshold, color)))
            kml.extend([
                "  <Folder>",
                f"    <name>{escape(f'Canopy mask (>= {threshold} m)')}</name>",
                "    <visibility>1</visibility>",
                *_ground_overlay_xml("mask.png", 2, n, s, e, w),
                "  </Folder>",
            ])
        kml.extend(["</Document>", "</kml>"])
        kmz.writestr("doc.kml", "\n".join(kml))


def write_combined_kmz(mosaics: list[dict], out_path: Path, threshold: float,
                       color: tuple[int, int, int, int], vmax: float,
                       alpha_threshold: float,
                       include_mask: bool, include_heights: bool) -> None:
    with ZipFile(out_path, "w", ZIP_DEFLATED) as kmz:
        kml = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            "<Document>",
            f"  <name>{escape(f'Canopy mosaic ({len(mosaics)} chunks)')}</name>",
        ]
        if include_heights:
            entries = []
            for i, mos in enumerate(mosaics, 1):
                href = f"heights/chunk_{i:03d}.png"
                kmz.writestr(href, _png_bytes(render_height(mos, vmax, alpha_threshold)))
                entries.extend(_ground_overlay_xml(
                    href, 1, mos["north"], mos["south"], mos["east"], mos["west"]))
            kml.extend([
                "  <Folder>",
                "    <name>CHMv2 height</name>",
                "    <visibility>0</visibility>",
                *entries,
                "  </Folder>",
            ])
        if include_mask:
            entries = []
            for i, mos in enumerate(mosaics, 1):
                href = f"masks/chunk_{i:03d}.png"
                kmz.writestr(href, _png_bytes(render_mask(mos, threshold, color)))
                entries.extend(_ground_overlay_xml(
                    href, 2, mos["north"], mos["south"], mos["east"], mos["west"]))
            kml.extend([
                "  <Folder>",
                f"    <name>{escape(f'Canopy mask (>= {threshold} m)')}</name>",
                "    <visibility>1</visibility>",
                *entries,
                "  </Folder>",
            ])
        kml.extend(["</Document>", "</kml>"])
        kmz.writestr("doc.kml", "\n".join(kml))


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def find_match(stem: str, meta: dict) -> str | None:
    for name in meta:
        if name.split(".")[0].lower() == stem.lower():
            return name
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", type=Path,
                   help="folder containing *_height.npy from chmv2_infer.py")
    p.add_argument("metadata", type=Path,
                   help="flight metadata file (semicolon-separated)")
    p.add_argument("-o", "--output", type=Path, default=Path("mosaic"),
                   help="output folder (default: mosaic/)")
    p.add_argument("--gsd-m", type=float, default=DEFAULT_GSD_M,
                   help="output mosaic GSD in meters per pixel (default 1.0)")
    p.add_argument("--chunk-km", type=float, default=DEFAULT_CHUNK_KM,
                   help="approximate cumulative GPS distance per chunk in km (default 10)")
    p.add_argument("--threshold", type=float, default=2.0,
                   help="canopy if mean height >= this (m); default 2.0")
    p.add_argument("--color", default="0,200,0,200",
                   help="RGBA mask color (default '0,200,0,200', fully visible)")
    p.add_argument("--height-vmax", type=float, default=15.0)
    p.add_argument("--height-alpha-threshold", type=float, default=1.0)
    p.add_argument("--focal-mm", type=float, default=DEFAULT_FOCAL_MM)
    p.add_argument("--max-pixels", type=int, default=200_000_000,
                   help="abort a chunk if its bbox exceeds this many pixels (default 2e8)")
    p.add_argument("--no-heights", dest="include_heights", action="store_false")
    p.add_argument("--no-mask", dest="include_mask", action="store_false")
    p.add_argument("--no-combined", dest="combined", action="store_false",
                   help="skip the combined all_chunks.kmz output")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--rectified", dest="rectified", action="store_true", default=True)
    g.add_argument("--unrectified", dest="rectified", action="store_false")
    args = p.parse_args()

    color = tuple(int(x) for x in args.color.split(","))
    if len(color) != 4:
        raise SystemExit("--color must be 4 ints (R,G,B,A)")

    meta = read_metadata(args.metadata)
    height_files = sorted(args.inputs.glob("*_height.npy"))
    if not height_files:
        raise SystemExit(f"no *_height.npy in {args.inputs}")

    frame_meta: list[tuple[Path, dict]] = []
    skipped = 0
    for hpath in height_files:
        stem = hpath.stem[:-len("_height")]
        name = find_match(stem, meta)
        if name is None:
            skipped += 1
            continue
        frame_meta.append((hpath, meta[name]))
    if not frame_meta:
        raise SystemExit("no frames matched the metadata file")

    chunks = chunk_by_distance(frame_meta, args.chunk_km * 1000.0)
    print(f"frames matched: {len(frame_meta)}  skipped: {skipped}")
    print(f"chunks: {len(chunks)}  (~{args.chunk_km} km each)")
    print(f"gsd: {args.gsd_m} m/px  threshold: {args.threshold} m  rectified: {args.rectified}")

    args.output.mkdir(parents=True, exist_ok=True)
    mosaics: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        chunk_id = f"{i:03d}"
        print(f"\n[chunk {chunk_id}/{len(chunks)}]  {len(chunk)} frames")
        mos = mosaic_chunk(chunk, args.inputs, args.gsd_m, args.focal_mm,
                           args.rectified, args.max_pixels)
        if mos is None:
            print(f"   no output")
            continue
        w, h = mos["size"]
        print(f"   raster {w}x{h} px  ({w*h*4 / 1e6:.0f} MB sum)")
        chunk_path = args.output / f"chunk_{chunk_id}.kmz"
        write_chunk_kmz(mos, chunk_path, args.threshold, color,
                        args.height_vmax, args.height_alpha_threshold, chunk_id,
                        args.include_mask, args.include_heights)
        size_mb = chunk_path.stat().st_size / 1e6
        print(f"   -> {chunk_path}  ({size_mb:.1f} MB)")
        mosaics.append(mos)

    if args.combined and mosaics:
        combined_path = args.output / "all_chunks.kmz"
        write_combined_kmz(mosaics, combined_path, args.threshold, color,
                           args.height_vmax, args.height_alpha_threshold,
                           args.include_mask, args.include_heights)
        size_mb = combined_path.stat().st_size / 1e6
        print(f"\nwrote combined {combined_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
