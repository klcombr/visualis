"""Chart and Diagram Analyzer.

Detects and extracts information from:
- Bar charts (vertical/horizontal)
- Line charts
- Pie/donut charts
- Flowcharts and diagrams
- Tables and grids
- UI screenshots

Uses geometric analysis + OCR for text extraction.
Returns structured data about chart type, axes, data points (where readable).
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visualis.processing.loader import LoadedImage


def detect_chart_type(arr: np.ndarray) -> str:
    """Heuristically determine chart type from image geometry."""
    # Convert to grayscale
    if arr.ndim == 3:
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        gray = arr.astype(np.float64)
    
    h, w = gray.shape
    
    # Look for circular patterns -> pie chart
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 4
    
    # Sample circle around center
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = (center_x + radius * np.cos(theta)).astype(int)
    circle_y = (center_y + radius * np.sin(theta)).astype(int)
    
    in_bounds = (circle_x >= 0) & (circle_x < w) & (circle_y >= 0) & (circle_y < h)
    if in_bounds.sum() > 50:
        circle_pixels = gray[circle_y[in_bounds], circle_x[in_bounds]]
        # Check if circle has varying intensity (pie slices)
        if circle_pixels.std() > 30:
            return "pie"
    
    # Look for vertical bars -> bar chart
    bar_regions = []
    col_means = gray.mean(axis=0)  # Average per column
    
    # Find columns with low variance (likely bar edges)
    for x in range(10, w - 10):
        col = gray[:, x]
        if col.std() < 20:  # Flat column -> possible bar edge or axis
            bar_regions.append(x)
    
    if len(bar_regions) > 5:
        # Group into contiguous regions
        groups = []
        current_start = bar_regions[0]
        for i in range(1, len(bar_regions)):
            if bar_regions[i] > bar_regions[i-1] + 5:
                groups.append((current_start, bar_regions[i-1]))
                current_start = bar_regions[i]
        groups.append((current_start, bar_regions[-1]))
        
        # Multiple flat regions -> likely bar chart
        if len(groups) >= 3:
            return "bar_vertical"
    
    # Look for horizontal lines -> line chart or horizontal bars
    row_means = gray.mean(axis=1)
    edges = np.abs(np.diff(row_means)) > 30
    if edges.sum() > 5:
        return "line_chart"
    
    # Check for grid lines -> table
    horiz_edges = np.mean(np.abs(np.diff(gray, axis=0))) > 20
    vert_edges = np.mean(np.abs(np.diff(gray, axis=1))) > 20
    
    if horiz_edges and vert_edges:
        return "table"
    
    return "unknown"


def extract_bar_data(arr: np.ndarray) -> dict:
    """Extract bar heights from a vertical bar chart."""
    if arr.ndim == 3:
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        gray = arr.astype(np.float64)
    
    h, w = gray.shape
    
    # Find axis baseline (bottom dark region)
    bottom_region = gray[h//2:, :]
    baseline_y = h - int(bottom_region.mean())
    
    # Sample columns to find bar tops
    bars = []
    x_step = max(1, w // 20)
    
    for x in range(10, w - 10, x_step):
        col = gray[:, x]
        # Find where intensity drops significantly (bar top)
        transitions = np.where(np.abs(np.diff(col)) > 50)[0]
        if len(transitions) > 0:
            bar_top = transitions[-1]
            bar_height = baseline_y - bar_top
            if 10 < bar_height < h * 0.8:
                bars.append({"x": x, "height": bar_height})
    
    return {
        "chart_type": "bar_vertical",
        "num_bars": len(bars),
        "max_height_px": max((b["height"] for b in bars), default=0),
        "sample_bars": bars[:5],  # First few for reference
    }


def extract_pie_slices(arr: np.ndarray) -> dict:
    """Extract pie chart slice angles and colors."""
    if arr.ndim == 3:
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        gray = arr.astype(np.float64)
    
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 3
    
    # Sample at different angles
    samples = []
    n_angles = 36
    angle_step = 360 / n_angles
    
    for angle in range(0, 360, int(angle_step)):
        rad = np.deg2rad(angle)
        x = int(cx + radius * 0.7 * np.cos(rad))
        y = int(cy - radius * 0.7 * np.sin(rad))
        
        if 0 <= x < w and 0 <= y < h:
            pixel = arr[y, x] if arr.ndim == 3 else gray[y, x]
            samples.append({
                "angle": angle,
                "color": pixel.tolist() if arr.ndim == 3 else float(pixel),
            })
    
    # Group similar colors to count slices
    unique_colors = {}
    for s in samples:
        c = tuple(s["color"][:3]) if isinstance(s["color"], list) else int(s["color"])
        key = c[:3]  # RGB only
        if key not in unique_colors:
            unique_colors[key] = {"count": 0, "angles": []}
        unique_colors[key]["count"] += 1
        unique_colors[key]["angles"].append(s["angle"])
    
    slices = []
    for color, info in unique_colors.items():
        angles = sorted(info["angles"])
        if len(angles) >= 5:
            slices.append({
                "color_rgb": list(color),
                "coverage_degrees": len(angles) * angle_step,
                "coverage_percent": round(len(angles) / n_angles * 100, 1),
            })
    
    return {
        "chart_type": "pie",
        "num_slices": len(slices),
        "slices": slices[:8],  # Top 8 slices
    }


def analyze_chart(
    arr: np.ndarray,
) -> dict:
    """Full chart analysis: detect type and extract data."""
    chart_type = detect_chart_type(arr)
    
    result = {"type": chart_type, "reliability": "low"}
    
    if chart_type == "bar_vertical":
        result.update(extract_bar_data(arr))
        result["reliability"] = "medium"  # Heights measurable, labels need OCR
    elif chart_type == "pie":
        result.update(extract_pie_slices(arr))
        result["reliability"] = "medium"  # Slices visible, percentages unclear
    elif chart_type == "table":
        result["rows_estimated"] = int(arr.shape[0] / 50)
        result["cols_estimated"] = int(arr.shape[1] / 80)
        result["reliability"] = "low"
    else:
        result["note"] = "Chart type not clearly identifiable"
    
    return result
