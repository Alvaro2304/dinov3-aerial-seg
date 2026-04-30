"""KMZ of rectified photos placed at their real GPS positions.

Standalone version: doesn't need CHMv2 inference outputs. Iterates over
the photos folder directly and computes each frame's 4 ground-corner
lat/lon from the flight metadata (GPS + AGL + yaw, with roll/pitch = 0
for rectified frames). Same geo conventions as mask_to_kmz.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

from mask_to_kmz import (
    DEFAULT_FOCAL_MM,
    PHOTO_EXTS,
    _folder_xml,
    _overlay_xml,
    photo_to_overlay_bytes,
    project_pixel_to_latlon,
    read_metadata,
)


def gather_photos(photos_dir: Path) -> list[Path]:
    return sorted(
        p for p in photos_dir.iterdir()
        if p.is_file() and p.suffix in PHOTO_EXTS
    )


def find_meta(photo: Path, meta: dict) -> dict | None:
    if photo.name in meta:
        return meta[photo.name]
    stem = photo.stem.lower()
    for key, m in meta.items():
        if key.split(".")[0].lower() == stem:
            return m
    return None


def build(photos_dir: Path, metadata_path: Path, out_path: Path,
          max_edge: int, photo_quality: int, dark_threshold: int,
          focal_mm: float, rectified: bool) -> None:
    meta = read_metadata(metadata_path)
    photos = gather_photos(photos_dir)
    if not photos:
        raise SystemExit(f"no images in {photos_dir}")

    print(f"photos: {len(photos)}  rectified: {rectified}  max-edge: {max_edge}")

    overlays: list[str] = []
    skipped = 0

    with ZipFile(out_path, "w", ZIP_DEFLATED) as kmz:
        for path in photos:
            m = find_meta(path, meta)
            if m is None:
                print(f"   skip {path.name}: not in metadata")
                skipped += 1
                continue

            with Image.open(path) as img:
                W, H = img.size

            roll = 0.0 if rectified else m["roll"]
            pitch = 0.0 if rectified else m["pitch"]

            corners = []
            for cx, cy in [(0, H), (W, H), (W, 0), (0, 0)]:  # LL, LR, UR, UL
                pt = project_pixel_to_latlon(
                    cx, cy, W, H, focal_mm,
                    m["lat"], m["lon"], m["h"],
                    roll, pitch, m["yaw"],
                )
                if pt is None:
                    break
                corners.append(pt)
            if len(corners) != 4:
                print(f"   skip {path.name}: ray misses ground")
                skipped += 1
                continue

            blob, ext = photo_to_overlay_bytes(
                path, max_edge, dark_threshold, photo_quality,
            )
            href = f"photos/{path.stem}.{ext}"
            kmz.writestr(href, blob)

            coords = " ".join(f"{lon:.7f},{lat:.7f},0" for (lat, lon) in corners)
            overlays.append(_overlay_xml(path.stem, href, coords, draw_order=0))

            if len(overlays) % 50 == 0:
                print(f"   ... {len(overlays)} photos")

        kml = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2" '
            'xmlns:gx="http://www.google.com/kml/ext/2.2">',
            "<Document>",
            f"  <name>{escape(f'Rectified photos ({len(overlays)} frames)')}</name>",
            *_folder_xml("Photos", True, overlays),
            "</Document>",
            "</kml>",
        ]
        kmz.writestr("doc.kml", "\n".join(kml))

    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path}  ({len(overlays)} photos, {skipped} skipped, {size_mb:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("photos_dir", type=Path,
                   help="folder of rectified JPGs (e.g. data_rectified/)")
    p.add_argument("metadata", type=Path,
                   help="flight metadata file (semicolon-separated)")
    p.add_argument("-o", "--output", type=Path, default=Path("photos.kmz"))
    p.add_argument("--max-edge", type=int, default=1024,
                   help="downsample to this max side (default 1024)")
    p.add_argument("--photo-quality", type=int, default=85,
                   help="JPEG quality (default 85)")
    p.add_argument("--dark-threshold", type=int, default=5,
                   help="pixels with max(R,G,B) <= this become transparent (default 5)")
    p.add_argument("--focal-mm", type=float, default=DEFAULT_FOCAL_MM)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--rectified", dest="rectified", action="store_true", default=True,
                   help="photos are rectified (use roll=pitch=0); default")
    g.add_argument("--unrectified", dest="rectified", action="store_false",
                   help="photos are raw tilted frames; project with their drone roll/pitch")
    args = p.parse_args()

    build(
        args.photos_dir, args.metadata, args.output,
        args.max_edge, args.photo_quality, args.dark_threshold,
        args.focal_mm, args.rectified,
    )


if __name__ == "__main__":
    main()
