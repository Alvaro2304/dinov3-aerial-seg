"""Single-image rectification: warp tilted aerial JPGs to a virtual nadir view.

Pure-rotation homography:

    H = K · R_mount⁻¹ · R_y(pitch) · R_x(roll) · R_mount · K⁻¹

Conventions follow geo_projection.html:
  - Camera frame:  +X right, +Y down, +Z forward (optical axis)
  - Mount:         R_mount = Rz(-90°), camera rigidly attached to airframe
  - ZYX Euler:     R_att = R_z(yaw) · R_y(pitch) · R_x(roll)
  - Yaw is preserved (image stays in-plane oriented as captured)

No DSM, no SfM. This removes tilt only — it does NOT correct parallax (tall
objects far from the principal point still lean) and assumes the camera
position is identical between the tilted view and the virtual nadir view
(it is — both are the same shutter).

Reads per-image attitude from the flight metadata CSV (semicolon-separated,
columns: name, ..., drone_roll, drone_pitch, drone_yaw, ...). Writes
rectified JPGs to --output, preserving filenames.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

SENSOR_W_MM = 35.7
SENSOR_H_MM = 23.8
DEFAULT_FOCAL_MM = 16.0

R_MOUNT = np.array([[0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]])


def _intrinsics(width: int, height: int, focal_mm: float) -> np.ndarray:
    fx = focal_mm * width / SENSOR_W_MM
    fy = focal_mm * height / SENSOR_H_MM
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]])


def _rdiff(roll_rad: float, pitch_rad: float) -> np.ndarray:
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return Ry @ Rx


def homography(width: int, height: int, roll_deg: float, pitch_deg: float,
               focal_mm: float = DEFAULT_FOCAL_MM) -> np.ndarray:
    K = _intrinsics(width, height, focal_mm)
    R_diff = _rdiff(np.radians(roll_deg), np.radians(pitch_deg))
    return K @ R_MOUNT.T @ R_diff @ R_MOUNT @ np.linalg.inv(K)


def rectify(img: np.ndarray, roll_deg: float, pitch_deg: float,
            focal_mm: float = DEFAULT_FOCAL_MM) -> np.ndarray:
    h, w = img.shape[:2]
    H = homography(w, h, roll_deg, pitch_deg, focal_mm)
    return cv2.warpPerspective(img, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))


def _read_metadata(csv_path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            out[row["name"]] = {
                "roll": float(row["drone_roll"]),
                "pitch": float(row["drone_pitch"]),
                "yaw": float(row["drone_yaw"]),
                "h": float(row["current_height"]),
            }
    return out


def _gather(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="JPG file or folder")
    p.add_argument("metadata", type=Path, help="flight CSV (semicolon-separated)")
    p.add_argument("-o", "--output", type=Path, default=Path("data_rectified"))
    p.add_argument("--focal-mm", type=float, default=DEFAULT_FOCAL_MM)
    p.add_argument("--max-tilt-deg", type=float, default=None,
                   help="skip frames whose |roll| or |pitch| exceeds this (deg)")
    args = p.parse_args()

    meta = _read_metadata(args.metadata)
    paths = _gather(args.input)
    if not paths:
        raise SystemExit(f"no JPGs in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"input={args.input}  metadata={args.metadata}  output={args.output}  images={len(paths)}")

    for path in paths:
        m = meta.get(path.name)
        if m is None:
            print(f"   skip {path.name}: not in metadata CSV")
            continue
        if args.max_tilt_deg is not None and (abs(m["roll"]) > args.max_tilt_deg
                                              or abs(m["pitch"]) > args.max_tilt_deg):
            print(f"   skip {path.name}: tilt > {args.max_tilt_deg}°")
            continue

        img = cv2.imread(str(path))
        if img is None:
            print(f"   skip {path.name}: cv2.imread failed")
            continue

        warped = rectify(img, m["roll"], m["pitch"], args.focal_mm)
        out_path = args.output / path.name
        cv2.imwrite(str(out_path), warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"-> {path.name}  roll={m['roll']:+6.2f}°  pitch={m['pitch']:+6.2f}°  -> {out_path}")


if __name__ == "__main__":
    main()
