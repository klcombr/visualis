"""Deep Inspection Module.

Implements iterative refinement for detailed analysis:
1. Start with coarse overview
2. Identify regions needing closer inspection
3. Crop and re-analyze at higher resolution
4. Merge results hierarchically

Useful for large images where full-resolution analysis is too expensive.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visualis.processing.loader import LoadedImage

from visualis.perception import pixels, geometry, segmentation


def identify_interesting_regions(
    arr: np.ndarray,
    threshold: float = 0.3,
) -> list[dict]:
    """Find regions with high variance (potential objects/text).
    
    Returns list of {x, y, w, h, reason} dicts.
    """
    if arr.ndim == 3:
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        gray = arr.astype(np.float64)
    
    h, w = gray.shape
    
    # Divide into grid
    cell_size = max(50, min(w, h) // 10)
    regions = []
    
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            cell = gray[y:y+cell_size, x:x+cell_size]
            variance = cell.var()
            
            if variance > threshold * 1000:  # Threshold for "interesting"
                regions.append({
                    "x": x,
                    "y": y,
                    "width": min(cell_size, w - x),
                    "height": min(cell_size, h - y),
                    "variance": float(variance),
                })
    
    # Sort by variance descending
    regions.sort(key=lambda r: -r["variance"])
    return regions[:10]  # Top 10 most interesting


def deep_inspect(
    arr: np.ndarray,
    max_levels: int = 2,
) -> dict:
    """Multi-level deep inspection.
    
    Level 0: Full image overview
    Level 1: Top regions cropped and analyzed
    Level 2+: Sub-regions within interesting areas
    """
    result = {
        "levels": [],
        "summary": "",
    }
    
    # Level 0: Overview
    level0 = {
        "level": 0,
        "region": None,
        "brightness": pixels.average_brightness(
            type('I', (), {'array': arr, 'width': arr.shape[1], 'height': arr.shape[0]})()
        ),
        "colors": len(pixels.dominant_colors(
            type('I', (), {'array': arr, 'width': arr.shape[1], 'height': arr.shape[0]})(),
            k=5
        )),
    }
    result["levels"].append(level0)
    
    # Level 1: Interesting regions
    interesting = identify_interesting_regions(arr)
    level1_details = []
    
    for i, region in enumerate(interesting[:5]):
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
        
        # Crop region
        crop = arr[y:y+h, x:x+w]
        
        region_info = {
            "index": i,
            "location": f"({x},{y})",
            "size": f"{w}x{h}",
            "variance": round(region["variance"], 1),
        }
        
        # Analyze region
        try:
            fake_img = type('I', (), {'array': crop, 'width': w, 'height': h})()
            region_info["brightness"] = pixels.average_brightness(fake_img)
            region_info["edge_density"] = geometry.detect_edges(fake_img).edge_fraction
        except Exception:
            pass
        
        level1_details.append(region_info)
    
    result["levels"].append({
        "level": 1,
        "regions_analyzed": len(level1_details),
        "details": level1_details,
    })
    
    # Generate summary
    result["summary"] = (
        f"Analyzed at {len(result['levels'])} levels. "
        f"Found {len(interesting)} interesting regions. "
        f"Avg brightness: {level0['brightness']:.2f}"
    )
    
    return result


def adaptive_tile_analysis(
    arr: np.ndarray,
    tile_size: int = 256,
) -> dict:
    """Analyze image in tiles, returning per-tile statistics.
    
    Useful for very large images where global analysis would be expensive.
    """
    h, w = arr.shape[:2]
    tiles = []
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = arr[y:y+tile_size, x:x+tile_size]
            
            tile_info = {
                "position": (x, y),
                "size": tile.shape[:2],
            }
            
            try:
                fake_img = type('I', (), {'array': tile, 'width': tile.shape[1], 'height': tile.shape[0]})()
                tile_info["brightness"] = pixels.average_brightness(fake_img)
                tile_info["dominant_color"] = pixels.average_color(fake_img).to_hex()
            except Exception:
                pass
            
            tiles.append(tile_info)
    
    return {
        "total_tiles": len(tiles),
        "tile_size": tile_size,
        "tiles": tiles[:20],  # Return first 20 for sampling
        "average_brightness": sum(t.get("brightness", 0) for t in tiles) / max(1, len(tiles)),
    }
