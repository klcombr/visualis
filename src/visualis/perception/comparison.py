"""Image Comparison Module.

Supports pixel-level, structural, and color-based comparisons:
- Pixel difference (absolute, relative)
- Structural similarity (SSIM approximation)
- Histogram comparison (chi-squared, correlation)
- Region-aware diff (bounding boxes around differences)

Useful for: visual regression testing, change detection, quality control.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visualis.processing.loader import LoadedImage


def resize_for_compare(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Resize images to same dimensions if needed."""
    a_shape = img_a.shape[:2]
    b_shape = img_b.shape[:2]
    
    if a_shape == b_shape:
        return img_a, img_b, False
    
    # Resize B to match A using interpolation
    from PIL import Image as _PIL
    pil_b = _PIL.fromarray(img_b)
    pil_b = pil_b.resize((a_shape[1], a_shape[0]), _PIL.LANCZOS)
    return img_a, np.array(pil_b), True


def pixel_difference(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
) -> dict:
    """Compute pixel-level differences between two images.
    
    Returns dict with changed_pixels, changed_fraction, mean_abs_diff, etc.
    """
    a, b, resized = resize_for_compare(arr_a, arr_b)
    
    # Convert to float for arithmetic
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    
    # Absolute difference per channel
    abs_diff = np.abs(af - bf)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    
    # Count significantly different pixels (>5 threshold)
    thresh = 5.0
    changed_mask = abs_diff.any(axis=2) > thresh if abs_diff.ndim > 2 else abs_diff > thresh
    changed_px = int(changed_mask.sum())
    total_px = a.size // (3 if a.ndim == 3 else 1)
    changed_frac = changed_px / max(1, total_px)
    
    # Find bounding boxes around changed regions
    ys, xs = np.nonzero(changed_mask)
    changed_regions = []
    if len(xs) > 0:
        # Group nearby changed pixels into regions
        from collections import deque
        visited = set()
        for x, y in zip(xs, ys):
            if (y, x) in visited:
                continue
            
            # BFS to find connected component
            queue = deque([(y, x)])
            visited.add((y, x))
            comp_xs, comp_ys = [], []
            
            while queue:
                cy, cx = queue.popleft()
                comp_xs.append(cx)
                comp_ys.append(cy)
                
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < a.shape[0] and 0 <= nx < a.shape[1]:
                        if (ny, nx) not in visited and changed_mask[ny, nx]:
                            visited.add((ny, nx))
                            queue.append((ny, nx))
            
            if len(comp_xs) >= 9:  # Min 3x3 region
                x0, x1 = min(comp_xs), max(comp_xs)
                y0, y1 = min(comp_ys), max(comp_ys)
                changed_regions.append({
                    "x": x0, "y": y0,
                    "width": x1 - x0 + 1,
                    "height": y1 - y0 + 1,
                })
    
    return {
        "changed_px": changed_px,
        "changed_fraction": round(changed_frac, 6),
        "mean_abs_diff": round(float(mean_diff), 2),
        "max_abs_diff": round(float(max_diff), 2),
        "changed_regions": changed_regions,
        "resized_for_compare": resized,
    }


def compute_ssim_approx(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    window_size: int = 11,
) -> float:
    """Approximate SSIM (Structural Similarity Index) between two images.
    
    Full SSIM requires multiple passes; this is a lightweight approximation
    using covariance of local patches.
    """
    a, b, _ = resize_for_compare(arr_a, arr_b)
    
    # Convert to grayscale
    if a.ndim == 3:
        ga = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        gb = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    else:
        ga, gb = a.astype(np.float64), b.astype(np.float64)
    
    # Normalize to [0, 1]
    ga = ga / 255.0
    gb = gb / 255.0
    
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Compute local means using uniform filter
    h, w = ga.shape
    ssim_map = np.zeros((h, w))
    
    ws = window_size
    pad = ws // 2
    
    # Pad edges
    ga_p = np.pad(ga, pad, mode='edge')
    gb_p = np.pad(gb, pad, mode='edge')
    
    for i in range(pad, h + pad):
        for j in range(pad, w + pad):
            win_a = ga_p[i-pad:i+pad+1, j-pad:j+pad+1].ravel()
            win_b = gb_p[i-pad:i+pad+1, j-pad:j+pad+1].ravel()
            
            mu_a = win_a.mean()
            mu_b = win_b.mean()
            sigma_a = win_a.std()
            sigma_b = win_b.std()
            sigma_ab = ((win_a - mu_a) * (win_b - mu_b)).mean()
            
            num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
            den = (mu_a**2 + mu_b**2 + C1) * (sigma_a**2 + sigma_b**2 + C2)
            
            ssim_map[i-pad, j-pad] = num / den
    
    return float(ssim_map.mean())


def histogram_comparison(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
) -> dict:
    """Compare color histograms between two images.
    
    Returns correlation and chi-squared distance.
    """
    a, b, _ = resize_for_compare(arr_a, arr_b)
    
    def hist(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            # Flatten and compute per-channel
            channels = [arr[..., i].ravel() for i in range(3)]
            hist_vals = [np.histogram(c, bins=64, range=(0, 256))[0] for c in channels]
            return np.concatenate(hist_vals).astype(np.float64)
        else:
            return np.histogram(arr.ravel(), bins=64, range=(0, 256))[0].astype(np.float64)
    
    ha = hist(a)
    hb = hist(b)
    
    # Normalize
    ha = ha / (ha.sum() + 1e-10)
    hb = hb / (hb.sum() + 1e-10)
    
    # Correlation
    corr = float(np.corrcoef(ha, hb)[0, 1]) if np.all(ha > 0) and np.all(hb > 0) else 0.0
    
    # Chi-squared distance
    chi2 = float(((ha - hb)**2 / (ha + hb + 1e-10)).sum())
    
    return {
        "histogram_correlation": round(corr, 4),
        "chi_squared_distance": round(chi2, 4),
        "histogram_a_bins": ha[:16].tolist(),  # Summary
        "histogram_b_bins": hb[:16].tolist(),
    }


def compare_images(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
) -> dict:
    """Comprehensive image comparison.
    
    Returns pixel diff, structural similarity, and histogram comparison.
    """
    pixel_diff = pixel_difference(arr_a, arr_b)
    ssim = compute_ssim_approx(arr_a, arr_b)
    hist_comp = histogram_comparison(arr_a, arr_b)
    
    return {
        "pixel_difference": pixel_diff,
        "structural_similarity": round(ssim, 4),
        "histogram_comparison": hist_comp,
        "overall_similarity": "identical" if ssim > 0.99 else "similar" if ssim > 0.8 else "different",
    }
