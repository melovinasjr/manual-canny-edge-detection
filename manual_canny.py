#!/usr/bin/env python3
"""
Manual Canny Edge Detection
- Does NOT use cv2.Canny or other Canny-related out-of-the-box functions.
- Uses:
  1) Gaussian blur
  2) Sobel gradients
  3) Magnitude M(x, y)
  4) Direction classification into 4 bins
  5) Non-Maximum Suppression
  6) Double threshold
  7) Connected components to link broken boundaries

Usage:
    python manual_canny.py input.jpg --out_dir output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def gaussian_blur(gray: np.ndarray, ksize: int = 5, sigma: float = 1.4) -> np.ndarray:
    return cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def sobel_gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

    # M(x, y)
    magnitude = np.hypot(gx, gy)

    # Direction in degrees, mapped to [0, 180)
    angle = np.rad2deg(np.arctan2(gy, gx))
    angle[angle < 0] += 180.0

    return gx, gy, magnitude, angle


def quantize_direction(angle: np.ndarray) -> np.ndarray:
    """
    Classify gradient directions into 4 categories:
    0   : left-right   (0 deg)
    45  : diagonal     (45 deg)
    90  : up-down      (90 deg)
    135 : diagonal     (135 deg)
    """
    direction = np.zeros_like(angle, dtype=np.uint8)

    mask0 = ((angle >= 0) & (angle < 22.5)) | ((angle >= 157.5) & (angle < 180))
    mask45 = (angle >= 22.5) & (angle < 67.5)
    mask90 = (angle >= 67.5) & (angle < 112.5)
    mask135 = (angle >= 112.5) & (angle < 157.5)

    direction[mask0] = 0
    direction[mask45] = 45
    direction[mask90] = 90
    direction[mask135] = 135

    return direction


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    h, w = magnitude.shape
    nms = np.zeros((h, w), dtype=np.float32)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            m = magnitude[y, x]
            d = direction[y, x]

            if d == 0:
                q = magnitude[y, x + 1]
                r = magnitude[y, x - 1]
            elif d == 45:
                q = magnitude[y - 1, x + 1]
                r = magnitude[y + 1, x - 1]
            elif d == 90:
                q = magnitude[y - 1, x]
                r = magnitude[y + 1, x]
            else:  # 135
                q = magnitude[y - 1, x - 1]
                r = magnitude[y + 1, x + 1]

            if m >= q and m >= r:
                nms[y, x] = m
            else:
                nms[y, x] = 0.0

    return nms


def double_threshold(
    nms: np.ndarray,
    low_ratio: float = 0.10,
    high_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Classify pixels into:
    - strong edge
    - weak edge
    - non-edge
    Thresholds are based on max NMS response.
    """
    max_val = float(nms.max())
    high_th = max_val * high_ratio
    low_th = max_val * low_ratio

    strong = nms >= high_th
    weak = (nms >= low_th) & (nms < high_th)
    candidate = strong | weak

    return candidate, weak, strong, low_th, high_th


def connect_by_components(candidate: np.ndarray, strong: np.ndarray) -> np.ndarray:
    """
    Use connected components to keep only candidate components
    that contain at least one strong edge pixel.
    This links weak edges to strong edges through connectivity.
    """
    candidate_u8 = candidate.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(candidate_u8, connectivity=8)

    out = np.zeros_like(candidate_u8, dtype=np.uint8)

    for label in range(1, num_labels):
        component_mask = labels == label
        if np.any(strong & component_mask):
            out[component_mask] = 255

    return out


def normalize_to_u8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - mn) / (mx - mn)
    return (norm * 255).clip(0, 255).astype(np.uint8)


def direction_to_visual(direction: np.ndarray) -> np.ndarray:
    """
    Visualize the 4 direction classes.
    0, 45, 90, 135 -> mapped to 0, 85, 170, 255 for easy viewing.
    """
    vis = np.zeros_like(direction, dtype=np.uint8)
    vis[direction == 0] = 0
    vis[direction == 45] = 85
    vis[direction == 90] = 170
    vis[direction == 135] = 255
    return vis


def manual_canny(
    image_bgr: np.ndarray,
    blur_ksize: int = 5,
    blur_sigma: float = 1.4,
    low_ratio: float = 0.10,
    high_ratio: float = 0.20,
) -> dict[str, np.ndarray | float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = gaussian_blur(gray, ksize=blur_ksize, sigma=blur_sigma)

    gx, gy, magnitude, angle = sobel_gradients(blurred)
    direction = quantize_direction(angle)
    nms = non_maximum_suppression(magnitude, direction)
    candidate, weak, strong, low_th, high_th = double_threshold(
        nms, low_ratio=low_ratio, high_ratio=high_ratio
    )
    final_edges = connect_by_components(candidate, strong)

    return {
        "gray": gray,
        "blurred": blurred,
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "angle": angle,
        "direction": direction,
        "nms": nms,
        "weak": weak.astype(np.uint8) * 255,
        "strong": strong.astype(np.uint8) * 255,
        "candidate": candidate.astype(np.uint8) * 255,
        "final_edges": final_edges,
        "low_th": low_th,
        "high_th": high_th,
    }


def save_results(results: dict[str, np.ndarray | float], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / f"{stem}_1_gray.png"), results["gray"])
    cv2.imwrite(str(out_dir / f"{stem}_2_blurred.png"), results["blurred"])
    cv2.imwrite(str(out_dir / f"{stem}_3_gx.png"), normalize_to_u8(results["gx"]))
    cv2.imwrite(str(out_dir / f"{stem}_4_gy.png"), normalize_to_u8(results["gy"]))
    cv2.imwrite(str(out_dir / f"{stem}_5_magnitude.png"), normalize_to_u8(results["magnitude"]))
    cv2.imwrite(str(out_dir / f"{stem}_6_direction.png"), direction_to_visual(results["direction"]))
    cv2.imwrite(str(out_dir / f"{stem}_7_nms.png"), normalize_to_u8(results["nms"]))
    cv2.imwrite(str(out_dir / f"{stem}_8_candidate.png"), results["candidate"])
    cv2.imwrite(str(out_dir / f"{stem}_9_strong.png"), results["strong"])
    cv2.imwrite(str(out_dir / f"{stem}_10_final_edges.png"), results["final_edges"])

    info = [
        f"low threshold : {results['low_th']:.4f}",
        f"high threshold: {results['high_th']:.4f}",
        "direction classes: 0, 45, 90, 135 degrees",
    ]
    (out_dir / f"{stem}_thresholds.txt").write_text("\n".join(info), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual Canny Edge Detection")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory to save results")
    parser.add_argument("--blur_ksize", type=int, default=5, help="Gaussian kernel size (odd)")
    parser.add_argument("--blur_sigma", type=float, default=1.4, help="Gaussian sigma")
    parser.add_argument("--low_ratio", type=float, default=0.10, help="Low threshold ratio")
    parser.add_argument("--high_ratio", type=float, default=0.20, help="High threshold ratio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.input)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if args.blur_ksize % 2 == 0:
        raise ValueError("blur_ksize must be odd")

    results = manual_canny(
        image,
        blur_ksize=args.blur_ksize,
        blur_sigma=args.blur_sigma,
        low_ratio=args.low_ratio,
        high_ratio=args.high_ratio,
    )

    save_results(results, Path(args.out_dir), image_path.stem)
    print(f"Saved outputs to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
