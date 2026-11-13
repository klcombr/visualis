"""Visualis — Universal Visual Intelligence Layer for Text-Based AI Agents.

Public API
----------
from visualis import inspect, analyze
from visualis.core.schemas import Box, Depth, TaskName
from visualis.core.config import VisualisConfig
from visualis.processing.loader import load_image
from visualis.perception import pixels, ocr, geometry
"""

from __future__ import annotations

# Re-export the main SDK functions
from visualis.core.engine import analyze, inspect  # noqa: F401
from visualis.core.schemas import (  # noqa: F401
    Box,
    ChartElement,
    Color,
    Detection,
    EvidenceSource,
    ModuleName,
    ModuleResult,
    OcrBox,
    Point,
    RegionDescription,
    SceneGraph,
    SegmentRegion,
    SkipReason,
    TaskName,
    VisualisContext,
    VisualisRequest,
)
from visualis.exceptions import (  # noqa: F401
    CorruptImageError,
    DependencyMissingError,
    ImageSourceNotFound,
    ImageTooLargeError,
    InputError,
    InvalidRegionError,
    PathTraversalError,
    ProviderError,
    ResourceLimitError,
    SchemaValidationError,
    UnsafeSvgError,
    UnsupportedFormatError,
    VisualisError,
)

__version__ = "0.1.0"
__all__ = [
    "inspect",
    "analyze",
    "Box",
    "ChartElement",
    "Color",
    "Detection",
    "EvidenceSource",
    "ModuleName",
    "ModuleResult",
    "OcrBox",
    "Point",
    "RegionDescription",
    "SceneGraph",
    "SegmentRegion",
    "SkipReason",
    "TaskName",
    "VisualisContext",
    "VisualisRequest",
    "CorruptImageError",
    "DependencyMissingError",
    "ImageSourceNotFound",
    "ImageTooLargeError",
    "InputError",
    "InvalidRegionError",
    "PathTraversalError",
    "ProviderError",
    "ResourceLimitError",
    "SchemaValidationError",
    "UnsafeSvgError",
    "UnsupportedFormatError",
    "VisualisError",
]
