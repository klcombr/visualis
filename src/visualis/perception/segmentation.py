"""Image Segmentation Module.

Provides region-based segmentation using multiple approaches:
- Connected component labeling (pure numpy, fast)
- GrabCut algorithm (requires OpenCV or manual implementation)
- K-means color clustering (deterministic, no external deps)
- Threshold-based segmentation (Otsu, adaptive)

Results include region bounding boxes, areas, centroids, and optional masks.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from visualis.core.schemas import SegmentRegion, SegmentationResult


def connected_components(arr: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Label connected components in a binary array.
    
    Returns (labeled_array, num_labels).
    Uses iterative BFS flood-fill for memory efficiency.
    """
    binary = (arr > 0).astype(np.uint8)
    labels = np.zeros_like(binary, dtype=np.int32)
    num_labels = 0
    h, w = binary.shape
    
    # 4-connectivity neighbors
    dirs_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # 8-connectivity neighbors  
    dirs_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dirs = dirs_8 if connectivity == 8 else dirs_4
    
    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0 or labels[y, x] != 0:
                continue
            
            num_labels += 1
            queue = deque([(y, x)])
            labels[y, x] = num_labels
            
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in dirs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if binary[ny, nx] == 1 and labels[ny, nx] == 0:
                            labels[ny, nx] = num_labels
                            queue.append((ny, nx))
    
    return labels, num_labels


def otsu_threshold(gray: np.ndarray) -> float:
    """Compute Otsu's optimal threshold value."""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    sum_all = float(np.dot(np.arange(256), hist))
    
    sum_b = 0.0
    w_b = 0.0
    best_var = -1.0
    threshold = 127.0
    
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
            threshold = t
    
    return threshold


def kmeans_cluster(arr: np.ndarray, k: int = 5, max_iter: int = 10) -> tuple[np.ndarray, list[np.ndarray]]:
    """Simple deterministic k-means clustering of pixels.
    
    Returns (labels, centers) where labels is HxW array of cluster IDs
    and centers is list of k RGB vectors.
    """
    pixels = arr.reshape(-1, 3).astype(np.float64)
    n = pixels.shape[0]
    
    # Deterministic initialization: pick evenly spaced pixels
    rng_state = np.random.get_state()
    np.random.seed(42)
    indices = np.linspace(0, n - 1, k, dtype=int)
    centers = pixels[indices].copy()
    np.random.set_state(rng_state)
    
    for _ in range(max_iter):
        # Assign to nearest center
        dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        
        # Update centers
        new_centers = np.zeros_like(centers)
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_centers[i] = pixels[mask].mean(axis=0)
        
        if np.allclose(centers, new_centers, atol=1.0):
            break
        centers = new_centers
    
    return labels.reshape(arr.shape[:2]), centers


def segment_by_color(
    arr: np.ndarray,
    k: int = 5,
    min_area_frac: float = 0.001,
) -> tuple[np.ndarray, list[float]]:
    """Segment image by color similarity using k-means.
    
    Returns (label_map, avg_colors) where label_map is HxW of cluster IDs
    and avg_colors is list of k RGB centroids.
    """
    rgb = arr.astype(np.float64) if arr.ndim == 3 else arr[:, :, None].astype(np.float64)
    labels, centers = kmeans_cluster(rgb, k=k)
    
    # Filter small clusters
    unique, counts = np.unique(labels, return_counts=True)
    valid = counts >= int(min_area_frac * arr.shape[0] * arr.shape[1])
    
    clean_labels = np.zeros_like(labels, dtype=np.int32)
    new_id = 0
    color_map = {}
    for old_id, count in zip(unique, counts):
        if not valid[old_id]:
            continue
        clean_labels[labels == old_id] = new_id
        color_map[new_id] = centers[old_id]
        new_id += 1
    
    return clean_labels, [c.tolist() for c in color_map.values()]


def find_regions(
    label_map: np.ndarray,
    min_area: int = 100,
) -> list[dict]:
    """Extract region info from labeled mask.
    
    Returns list of dicts with box, area, centroid, color.
    """
    unique = np.unique(label_map)
    regions = []
    
    for label in unique:
        if label == 0:
            continue
        
        mask = label_map == label
        area = int(mask.sum())
        if area < min_area:
            continue
        
        ys, xs = np.nonzero(mask)
        x0, y0 = xs.min(), ys.min()
        x1, y1 = xs.max(), ys.max()
        
        regions.append({
            "label": int(label),
            "box": {"x": int(x0), "y": int(y0), "width": int(x1 - x0 + 1), "height": int(y1 - y0 + 1)},
            "area": area,
            "centroid": {"x": int(xs.mean()), "y": int(ys.mean())},
        })
    
    regions.sort(key=lambda r: -r["area"])
    return regions


def threshold_segment(
    arr: np.ndarray,
    method: str = "otsu",
    connectivity: int = 4,
    min_area: int = 100,
) -> list[dict]:
    """Segment using intensity thresholding.
    
    Args:
        arr: Input image (HxW or HxWx3)
        method: "otsu" or "fixed" (default threshold 128)
        connectivity: 4 or 8 for component linking
        min_area: Minimum region size to keep
    """
    # Convert to grayscale
    if arr.ndim == 3:
        gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
    else:
        gray = arr.astype(np.uint8)
    
    # Compute threshold
    if method == "otsu":
        thresh = otsu_threshold(gray.astype(np.float64))
    else:
        thresh = 128.0
    
    # Binary segmentation
    binary = (gray > thresh).astype(np.uint8)
    labels, _ = connected_components(binary, connectivity)
    
    return find_regions(labels, min_area=min_area)


def grabcut_like(
    arr: np.ndarray,
    k: int = 2,
    bg_mask: np.ndarray | None = None,
) -> list[dict]:
    """Simple foreground/background segmentation inspired by GrabCut.
    
    Uses color clustering with optional initial background guess.
    """
    if arr.ndim == 3:
        rgb = arr.astype(np.float64)
    else:
        rgb = np.stack([arr] * 3, axis=2)
    
    labels, centers = kmeans_cluster(rgb, k=k)
    
    # Assume background is the largest border-touching region
    h, w = labels.shape
    border_labels = set()
    for y in [0, h-1]:
        border_labels.update(labels[y].tolist())
    for x in [0, w-1]:
        border_labels.update(labels[:, x].tolist())
    
    # Find largest border region -> likely background
    from collections import Counter
    border_counts = Counter(border_labels)
    bg_label = border_counts.most_common(1)[0][0] if border_counts else 0
    
    # Foreground is everything else
    fg_mask = labels != bg_label
    
    return find_regions(fg_mask.astype(np.int32), min_area=100)


def semantic_segment(
    arr: np.ndarray,
    method: str = "color",
    k: int = 5,
    **kwargs: object,
) -> list[dict]:
    """High-level segmentation interface.
    
    Methods:
        "color": K-means color clustering
        "threshold": Intensity-based with Otsu
        "grabcut": Simple foreground extraction
    """
    if method == "threshold":
        return threshold_segment(arr, **kwargs)
    elif method == "grabcut":
        return grabcut_like(arr, **kwargs)
    else:  # default to color
        labels, _ = segment_by_color(arr, k=k, **kwargs)
        min_area = kwargs.get("min_area", 100)
        return find_regions(labels, min_area=min_area)


def compute_region_stats(
    arr: np.ndarray,
    region_info: dict,
) -> dict:
    """Compute statistics for a segmented region."""
    x, y, w, h = region_info["box"]["x"], region_info["box"]["y"], region_info["box"]["width"], region_info["box"]["height"]
    
    sub = arr[y:y+h, x:x+w]
    if sub.size == 0:
        return {}
    
    if sub.ndim == 3:
        avg_color = sub.mean(axis=(0, 1))
    else:
        avg_color = np.array([sub.mean()] * 3)
    
    return {
        "average_color": [int(c) for c in avg_color],
        "brightness": float(avg_color.mean()) / 255.0,
        "saturation": float(np.std(avg_color) / 128.0),
    }
