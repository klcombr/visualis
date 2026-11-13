"""Geometry, edges, noise, and quality — pure numpy, zero optional deps.

Everything here is deterministic and fast on CPU. We deliberately avoid a
hard OpenCV dependency: Sobel/Laplacian/Harris all have small, well-known
kernel forms that numpy handles in a few lines.

Conventions:
  * grayscale = Rec.709 luma
  * all fractions in 0..1
  * corner/line counts are thresholds, documented in each function
"""

from __future__ import annotations

import numpy as np
from collections import deque

from visualis.core.schemas import Box, EdgeInfo, GeometryStats, NoiseEstimate, Point, QualityMetrics
from visualis.processing.loader import LoadedImage


_LUMA = np.array([0.2126, 0.7152, 0.0722])


def _gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.shape[2] >= 3:
        return (arr[..., :3].astype(np.float64) * _LUMA).sum(axis=2)
    raise ValueError("cannot convert array to grayscale")


def _conv2(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution via brute-force sum of shifted planes; border replicated."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    p = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * p[i : i + img.shape[0], j : j + img.shape[1]]
    return out


def sobel(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = _conv2(arr, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64))
    gy = _conv2(arr, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64))
    return gx, gy


def edge_map(arr: np.ndarray, threshold: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    g = _gray(arr)
    gx, gy = sobel(g)
    mag = np.sqrt(gx**2 + gy**2)
    return (mag >= threshold).astype(np.uint8), mag


def detect_edges(image: LoadedImage, threshold: float = 40.0) -> EdgeInfo:
    edges, mag = edge_map(image.array, threshold)
    n = int(edges.sum())
    total = image.width * image.height
    return EdgeInfo(
        method="sobel",
        edge_pixel_count=n,
        edge_fraction=round(n / max(1, total), 4),
        edge_strength_mean=round(float(mag.mean()), 2),
    )


def _harris(gray: np.ndarray, k: float = 0.04, top_n: int = 20) -> list[Point]:
    gx, gy = sobel(gray)
    ix2, iy2 = gx**2, gy**2
    ixy = gx * gy

    def box3(m: np.ndarray) -> np.ndarray:
        s = np.zeros_like(m)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                s += np.roll(np.roll(m, dy, axis=0), dx, axis=1)
        return s

    s_xx, s_xy, s_yy = box3(ix2), box3(ixy), box3(iy2)
    det = s_xx * s_yy - s_xy**2
    tr = s_xx + s_yy
    resp = det - k * tr**2
    peaks: list[tuple[float, int, int]] = []
    for y in range(1, resp.shape[0] - 1):
        for x in range(1, resp.shape[1] - 1):
            win = resp[y - 1 : y + 2, x - 1 : x + 2]
            if resp[y, x] == win.max() and resp[y, x] > 0:
                peaks.append((float(resp[y, x]), x, y))
    peaks.sort(reverse=True)
    return [Point(x=int(x), y=int(y)) for _, x, y in peaks[:top_n]]


def _long_lines(gray: np.ndarray, min_len: int = 40, threshold: float = 40.0) -> int:
    edges, _ = edge_map(gray, threshold)
    ys, xs = np.nonzero(edges)
    if xs.size < min_len:
        return 0
    thetas = np.deg2rad(np.arange(0, 180, 2))
    best_per_thetas: dict[float, int] = {}
    for t in thetas:
        rho = (xs * np.cos(t) + ys * np.sin(t)).astype(int)
        vals, counts = np.unique(rho, return_counts=True)
        strong = counts[counts >= min_len]
        if len(strong) > best_per_thetas.get(t, 0):
            best_per_thetas[t] = len(strong)
    return max(best_per_thetas.values()) if best_per_thetas else 0


def _symmetry(gray: np.ndarray) -> tuple[float, float]:
    h, w = gray.shape
    wl = min(w // 2, gray.shape[1])
    left = gray[:, :wl]
    right = gray[:, w - wl :][:, ::-1]
    hsym = max(0.0, 1.0 - float(np.abs(left - right).mean()) / 255.0)
    wh = min(h // 2, gray.shape[0])
    top = gray[:wh, :]
    bottom = gray[h - wh :, :][::-1, :]
    vsym = max(0.0, 1.0 - float(np.abs(top - bottom).mean()) / 255.0)
    return round(hsym, 4), round(vsym, 4)


def _blobs(arr: np.ndarray, max_boxes: int = 8, min_area_frac: float = 0.001) -> list[Box]:
    g = _gray(arr)
    # downsample to a coarse grid for speed
    scale = min(1.0, (512 / max(arr.shape[:2])) ** 0.5) if max(arr.shape[:2]) > 512 else 1.0
    if scale < 1.0:
        from PIL import Image as _PIL

        small = _PIL.fromarray(g.astype(np.uint8)).resize(
            (max(1, int(g.shape[1] * scale)), max(1, int(g.shape[0] * scale))), _PIL.BILINEAR
        )
        gs = np.asarray(small).astype(np.float64)
    else:
        gs = g
    hist, _ = np.histogram(gs.ravel(), bins=256, range=(0, 256))
    total = gs.size
    sum_all = float(np.dot(np.arange(256), hist))
    w_b = 0.0
    sum_b = 0.0
    best_var = -1.0
    thresh = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            thresh = t
    mask = (gs > thresh).astype(np.uint8)
    if mask.mean() > 0.5:
        mask = 1 - mask
    if mask.mean() < min_area_frac:
        return []
    labels = np.zeros_like(mask, dtype=np.int32)
    label_id = 0
    visited = np.zeros_like(mask, dtype=bool)
    results: list[Box] = []
    h, w = mask.shape
    for y in range(0, h, 3):
        for x in range(0, w, 3):
            if not mask[y, x] or visited[y, x]:
                continue
            label_id += 1
            q = deque([(y, x)])
            visited[y, x] = True
            xs_list: list[int] = []
            ys_list: list[int] = []
            while q:
                cy, cx = q.popleft()
                xs_list.append(cx)
                ys_list.append(cy)
                if len(xs_list) > 200_000:
                    break
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
            if len(xs_list) < max(4, int(min_area_frac * h * w)):
                continue
            if label_id > max_boxes:
                break
            x0, x1 = min(xs_list), max(xs_list)
            y0, y1 = min(ys_list), max(ys_list)
            sx = 1.0 / scale
            sy = 1.0 / scale
            bx = Box(
                x=int(x0 * sx),
                y=int(y0 * sy),
                width=max(1, int((x1 - x0 + 1) * sx)),
                height=max(1, int((y1 - y0 + 1) * sy)),
            )
            # clamp to real image size
            rw = arr.shape[1]
            rh = arr.shape[0]
            bx.x = min(bx.x, rw - 1)
            bx.y = min(bx.y, rh - 1)
            bx.width = min(bx.width, rw - bx.x)
            bx.height = min(bx.height, rh - bx.y)
            if bx.width > 0 and bx.height > 0:
                results.append(bx)
    return results


def geometry_stats(image: LoadedImage) -> GeometryStats:
    g = _gray(image.array)
    hsym, vsym = _symmetry(g)
    blobs = _blobs(image.array)
    sd = g.std()
    if sd < 1e-9:
        solid_frac = 0.0
    else:
        solid_frac = min(1.0, float(np.abs(g - g.mean()).std()) / max(sd, 1.0))
    xs, ys = np.nonzero((np.abs(g - g.mean()) > 2 * max(sd, 1.0)).astype(bool))
    centroid: Point | None = Point(x=int(xs.mean()), y=int(ys.mean())) if xs.size else None
    return GeometryStats(
        symmetry_horizontal=hsym,
        symmetry_vertical=vsym,
        bounding_boxes=blobs,
        solid_fraction=solid_frac,
        content_centroid=centroid,
        corners=_harris(g),
        straight_lines=_long_lines(g),
        notes=["Otsu foreground blobs, Harris corners, coarse Hough lines"],
    )


def estimate_noise(image: LoadedImage) -> NoiseEstimate:
    g = _gray(image.array)
    lap = _conv2(g, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64))
    flat = np.abs(lap)
    lo, hi = np.percentile(flat, 10), np.percentile(flat, 90)
    band = flat[(flat >= lo) & (flat <= hi)]
    if band.size == 0:
        sigma = 0.0
    else:
        sigma = float(np.median(band)) / 0.6745
    sigma = min(sigma, 255.0)
    level = "low" if sigma < 3.0 else "medium" if sigma < 10.0 else "high"
    return NoiseEstimate(method="mad-laplacian", sigma=round(sigma, 3), estimate=level)


def quality_metrics(image: LoadedImage) -> QualityMetrics:
    g = _gray(image.array)
    lap = _conv2(g, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64))
    var = float(lap.var())
    if var < 100:
        blur = "strongly_blurred"
    elif var < 500:
        blur = "slightly_blurred"
    elif var < 2000:
        blur = "sharp"
    else:
        blur = "very_sharp_or_noisy"
    return QualityMetrics(
        sharpness=round(var, 1),
        laplacian_var=round(var, 1),
        blur_estimate=blur,
        notes=["variance of Laplacian on grayscale"],
    )


def position_name(x: int, y: int, w: int, h: int, image_w: int, image_h: int) -> str:
    cx = (x + w / 2) / max(1, image_w)
    cy = (y + h / 2) / max(1, image_h)
    hx = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    vy = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else ""
    if hx == "center" and vy == "":
        return "center"
    return f"{vy}-{hx}" if vy else hx
