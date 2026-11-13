"""Pixel Intelligence.

Direct measurement of pixel data. Every value here is deterministic and
traceable to the underlying array. No model, no guesswork.

Query styles supported:
    pixel_at(x, y)          -> exact sample
    region_stats(box)       -> avg/median/brightness over a box
    color_distribution(...) -> binned RGB histogram
    dominant_colors(...)    -> top-k exact colors (or reduced palette)
"""

from __future__ import annotations

import colorsys
from collections import Counter

import numpy as np

from visualis.core.schemas import (
    ChannelHistogramBin,
    Color,
    ColorHistogramBin,
    HSVColor,
    PixelSample,
    RegionColorStats,
)
from visualis.exceptions import InvalidRegionError
from visualis.processing.loader import LoadedImage


def _rgb(arr: np.ndarray) -> np.ndarray:
    """Ensure HxWx3 float array (dropping alpha)."""
    if arr.ndim == 2:
        return arr.astype(np.float64)[:, :, None]
    if arr.shape[2] == 4:
        return arr[:, :, :3].astype(np.float64)
    return arr.astype(np.float64)


def _rel(arr: np.ndarray, x: int, y: int) -> tuple[int, int]:
    h, w = arr.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        raise InvalidRegionError(f"({x},{y}) outside {w}x{h} image")
    return x, y


def pixel_at(image: LoadedImage, x: int, y: int) -> PixelSample:
    """Exact RGB(A) value at (x, y)."""
    x, y = _rel(image.array, x, y)
    px = image.array[y, x]
    if image.array.ndim == 3:
        r, g, b = int(px[0]), int(px[1]), int(px[2])
        a = int(px[3]) if image.array.shape[2] == 4 else 255
    else:
        r = g = b = int(px)
        a = 255
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return PixelSample(
        x=x,
        y=y,
        color=Color(r=r, g=g, b=b, a=a),
        hsv=HSVColor(h=round(h * 360.0, 2), s=round(s, 4), v=round(v, 4)),
        in_bounds=True,
    )


def _mask_box(arr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    H, W = arr.shape[:2]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
        raise InvalidRegionError(
            f"region ({x},{y},{w},{h}) outside {W}x{H} image"
        )
    return arr[y : y + h, x : x + w]


def region_stats(
    image: LoadedImage,
    x: int = 0,
    y: int = 0,
    w: int | None = None,
    h: int | None = None,
) -> RegionColorStats:
    """Average/dominant/brightness for a rectangle (defaults to full image)."""
    w = w or image.width
    h = h or image.height
    sub = _mask_box(image.array, x, y, w, h)
    rgb = _rgb(sub)
    flat = rgb.reshape(-1, 3)
    avg = flat.mean(axis=0)
    med = np.median(flat, axis=0)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    from visualis.core.schemas import Box

    top = Counter(map(tuple, (flat // 8 * 8).astype(int)))
    total = flat.shape[0]
    dominant = [
        ColorHistogramBin(
            color=Color(r=int(c[0]), g=int(c[1]), b=int(c[2])),
            count=int(n),
            fraction=round(n / total, 6),
        )
        for c, n in top.most_common(5)
    ]
    return RegionColorStats(
        region=Box(x=x, y=y, width=w, height=h),
        average=Color(r=int(avg[0]), g=int(avg[1]), b=int(avg[2])),
        dominant=dominant,
        brightness=round(float(lum.mean()) / 255.0, 4),
        contrast=round(float(lum.std()) / 255.0, 4),
    )


def color_distribution(
    image: LoadedImage,
    channels: tuple[str, ...] = ("r", "g", "b"),
    bins: int = 16,
) -> dict[str, list[ChannelHistogramBin]]:
    """Histogram per channel, quantized to ``bins`` levels."""
    rgb = _rgb(image.array)
    step = 256 // max(1, bins)
    out: dict[str, list[ChannelHistogramBin]] = {}
    for i, ch in enumerate(channels):
        if i >= 3:
            break
        vals = rgb[..., i].astype(np.int64).ravel()
        q = (vals // step * step)
        counts = Counter(q.tolist())
        total = vals.size
        out[ch] = [
            ChannelHistogramBin(
                channel=ch, bin=int(b), count=int(n), fraction=round(n / total, 6)
            )
            for b, n in sorted(counts.items())
        ]
    return out


def dominant_colors(
    image: LoadedImage,
    k: int = 5,
    method: str = "quantile",
) -> list[ColorHistogramBin]:
    """Top-k exact colors (quantile='quantile' rounds to 8-level buckets;
    'kmeans' reduces with a tiny deterministic k-means on sampled pixels)."""
    rgb = _rgb(image.array)
    flat = rgb.reshape(-1, 3)
    if flat.shape[0] > 200_000:
        idx = np.linspace(0, flat.shape[0] - 1, 200_000).astype(int)
        flat = flat[idx]
    if method == "quantile":
        quant = (flat // 8 * 8).astype(int)
        counts = Counter(map(tuple, quant.tolist()))
        total = flat.shape[0]
        return [
            ColorHistogramBin(
                color=Color(r=int(c[0]), g=int(c[1]), b=int(c[2])),
                count=int(n),
                fraction=round(n / total, 6),
            )
            for c, n in counts.most_common(k)
        ]
    if method == "kmeans":
        return _kmeans_palette(flat, k)
    raise InvalidRegionError(f"unknown method '{method}' (use quantile|kmeans)")


def _kmeans_palette(flat: np.ndarray, k: int) -> list[ColorHistogramBin]:
    rng = np.random.default_rng(1234)  # deterministic
    if flat.shape[0] < k:
        k = max(1, flat.shape[0])
    centers = flat[rng.choice(flat.shape[0], size=k, replace=False)].astype(np.float64)
    for _ in range(10):
        d = np.linalg.norm(flat[:, None, :] - centers[None, :, :], axis=2)
        assign = d.argmin(axis=1)
        new_centers = centers.copy()
        for i in range(k):
            m = assign == i
            if m.any():
                new_centers[i] = flat[m].mean(axis=0)
        if np.allclose(centers, new_centers, atol=0.5):
            centers = new_centers
            break
        centers = new_centers
    d = np.linalg.norm(flat[:, None, :] - centers[None, :, :], axis=2)
    assign = d.argmin(axis=1)
    total = flat.shape[0]
    out = []
    for i in range(k):
        n = int((assign == i).sum())
        if n == 0:
            continue
        c = centers[i]
        out.append(
            ColorHistogramBin(
                color=Color(r=int(round(c[0])), g=int(round(c[1])), b=int(round(c[2]))),
                count=n,
                fraction=round(n / total, 6),
            )
        )
    out.sort(key=lambda b: -b.count)
    return out[:k]


def average_brightness(image: LoadedImage) -> float:
    rgb = _rgb(image.array)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return round(float(lum.mean()) / 255.0, 4)


def average_color(image: LoadedImage) -> Color:
    rgb = _rgb(image.array)
    m = rgb.mean(axis=(0, 1))
    return Color(r=int(m[0]), g=int(m[1]), b=int(m[2]))


def median_color(image: LoadedImage) -> Color:
    rgb = _rgb(image.array)
    m = np.median(rgb.reshape(-1, 3), axis=0)
    return Color(r=int(m[0]), g=int(m[1]), b=int(m[2]))


def transparency_stats(image: LoadedImage) -> dict[str, float]:
    """Alpha-channel coverage. Returns zero when the image has no alpha."""
    if image.array.ndim < 3 or image.array.shape[2] != 4:
        return {"transparent_fraction": 0.0, "opaque_fraction": 1.0, "has_alpha": False}
    a = image.array[..., 3]
    tf = float((a < 128).mean())
    return {
        "transparent_fraction": round(tf, 4),
        "opaque_fraction": round(1.0 - tf, 4),
        "has_alpha": True,
    }


def tile_coordinates(
    width: int,
    height: int,
    tile: int,
    overlap: int = 0,
) -> list[tuple[int, int, int, int]]:
    """Grid of (x, y, w, h) tiles covering the image with the given overlap."""
    if tile <= 0:
        raise InvalidRegionError("tile size must be positive")
    step = max(1, tile - overlap)
    tiles: list[tuple[int, int, int, int]] = []
    y = 0
    while y < height:
        x = 0
        while x < width:
            tw = min(tile, width - x)
            th = min(tile, height - y)
            if tw > 0 and th > 0:
                tiles.append((x, y, tw, th))
            x += step
        y += step
    return tiles
