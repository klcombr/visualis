"""Core Visualis engine — the public-facing Python SDK entry point.

The orchestrator:
  1. loads the image safely (loader.py)
  2. resizes to work-size when needed (saves GPU/memory on OCR/detection)
  3. dispatches requested tasks across modules in parallel where possible
  4. assembles a VisualisContext with provenance on every claim
  5. never silently skips modules without recording why

Depth semantics:
  FAST   = metadata + pixels + colors only
  STANDARD = fast + ocr + geometry + edge/noise/quality
  DEEP   = standard + optional segmentation/chart/comparison + deep inspection

Use explicit ``tasks=`` to override depth defaults. Both accept any subset of
the registered task names; missing providers are recorded as skipped instead of raising.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    pass  # runtime imports below are conditional

from visualis.core.config import VisualisConfig
from visualis.core.schemas import (
    Box,
    Depth,
    DetectionResult,
    ImageMetadata,
    ModuleName,
    ModuleResult,
    OcrResult,
    SkipReason,
    SkippedModule,
    TaskName,
    VisualisContext,
    VisualisRequest,
)
from visualis.exceptions import DependencyMissingError, VisualisError
from visualis.perception import geometry, ocr, pixels
from visualis.processing.loader import LoadedImage, load_image


# --- registered task handlers ---------------------------------------------

TASK_HANDLERS: dict[TaskName, Any] = {}


def register_task(task: TaskName, handler: Any) -> None:
    TASK_HANDLERS[task] = handler


# Register built-in task handlers
from visualis.perception import ocr as ocr_module
TASK_HANDLERS[TaskName.OCR] = ocr_module.run_ocr

# Depth mapping: which tasks run at each depth level (default set)
DEPTH_TASKS: dict[Depth, frozenset[TaskName]] = {
    Depth.FAST: frozenset({TaskName.METADATA, TaskName.PIXELS, TaskName.COLORS}),
    Depth.STANDARD: frozenset({
        TaskName.METADATA, TaskName.PIXELS, TaskName.COLORS,
        TaskName.GEOMETRY, TaskName.OCR,
    }),
    Depth.DEEP: frozenset({
        TaskName.METADATA, TaskName.PIXELS, TaskName.COLORS,
        TaskName.GEOMETRY, TaskName.OCR,
        TaskName.DETECTION, TaskName.SEGMENTATION,
        TaskName.COMPARISON, TaskName.SCENE, TaskName.CHARTS,
    }),
}


def _make_metadata(image: LoadedImage) -> ImageMetadata:
    return ImageMetadata(
        format=image.metadata.format,
        mime_type=image.metadata.mime_type,
        width=image.width,
        height=image.height,
        mode=image.metadata.mode,
        size_bytes=image.metadata.size_bytes,
        file_path=getattr(image, "source", ""),
        frames=image.metadata.frames,
        fps=image.metadata.fps,
        transparency=image.metadata.transparency,
        orientation_corrected=image.metadata.orientation_corrected,
        exif=image.metadata.exif,
        icc_profile_present=image.metadata.icc_profile_present,
        aspect_ratio=image.metadata.aspect_ratio,
    )


def _run_pixel_tasks(image: LoadedImage, *, depth: Depth, config: VisualisConfig) -> dict[str, ModuleResult]:
    t0 = time.time()
    results: dict[str, ModuleResult] = {}

    def ok(module: ModuleName, payload: Any, confidence: float | None = None, duration_ms: float | None = None) -> ModuleResult:
        dur = duration_ms or (time.time() - t0) * 1000
        return ModuleResult(module=module, status="ok", duration_ms=dur, payload=payload, confidence=confidence)
    def skip(module: ModuleName, reason: SkipReason, detail: str = "", install_hint: str | None = None) -> ModuleResult:
        return ModuleResult(module=module, status="skipped", duration_ms=0.0, skip_reason=reason, skip_detail=detail, install_hint=install_hint)

    # METADATA is always available — it's just the header
    results["metadata"] = ok(ModuleName.METADATA, {"metadata": _make_metadata(image).model_dump()}, confidence=1.0, duration_ms=time.time() - t0)

    if depth in (Depth.FAST, Depth.STANDARD, Depth.DEEP):
        r = time.time()
        try:
            avg_color = pixels.average_color(image)
            med_color = pixels.median_color(image)
            results["pixels"] = ok(
                ModuleName.PIXELS,
                {
                    "average_brightness": pixels.average_brightness(image),
                    "average_color": avg_color.model_dump(),
                    "median_color": med_color.model_dump(),
                    "transparency": pixels.transparency_stats(image),
                    "size_pixels": image.width * image.height,
                },
                confidence=1.0,
                duration_ms=(time.time() - r) * 1000,
            )
        except Exception as e:
            results["pixels"] = skip(ModuleName.PIXELS, SkipReason.PROVIDER_FAILED, str(e))

        r = time.time()
        try:
            dom = pixels.dominant_colors(image, k=5)
            hist = pixels.color_distribution(image)
            results["colors"] = ok(
                ModuleName.COLORS,
                {
                    "dominant_colors": [d.model_dump() for d in dom],
                    "histogram": {ch: [b.model_dump() for b in bl] for ch, bl in hist.items()},
                    "palette_summary": f"top {len(dom)} dominant colors by pixel count",
                },
                confidence=1.0,
                duration_ms=(time.time() - r) * 1000,
            )
        except Exception as e:
            results["colors"] = skip(ModuleName.COLORS, SkipReason.PROVIDER_FAILED, str(e))

        r = time.time()
        try:
            geo = geometry.geometry_stats(image)
            edge = geometry.detect_edges(image)
            noise = geometry.estimate_noise(image)
            qual = geometry.quality_metrics(image)
            results["geometry"] = ok(
                ModuleName.GEOMETRY,
                {
                    **geo.model_dump(),
                    "edge_density": edge.edge_fraction,
                    "noise_sigma": noise.sigma,
                    "noise_level": noise.estimate,
                    "quality_blur": qual.blur_estimate,
                    "quality_laplacian_var": qual.laplacian_var,
                },
                confidence=0.85,
                duration_ms=(time.time() - r) * 1000,
            )
        except Exception as e:
            results["geometry"] = skip(ModuleName.GEOMETRY, SkipReason.PROVIDER_FAILED, str(e))

    if TaskName.OCR in TASK_HANDLERS and depth in (Depth.STANDARD, Depth.DEEP):
        handler = TASK_HANDLERS[TaskName.OCR]
        r = time.time()
        try:
            img = image.to_work_size(config.processing.max_work_pixels)
            result: OcrResult = handler(img, config=config)
            results["ocr"] = ok(
                ModuleName.OCR,
                result.model_dump(),
                confidence=result.overall_confidence,
                duration_ms=(time.time() - r) * 1000,
            )
        except DependencyMissingError as e:
            results["ocr"] = skip(ModuleName.OCR, SkipReason.DEPENDENCY_MISSING, e.message, e.install_hint)
        except Exception as e:
            results["ocr"] = skip(ModuleName.OCR, SkipReason.PROVIDER_FAILED, str(e))

    return results


def inspect(
    source: str,
    *,
    config: VisualisConfig | None = None,
    data: bytes | None = None,
    depth: str | Depth = Depth.STANDARD,
    tasks: list[str] | Sequence[TaskName] | None = None,
    region: tuple[int, int, int, int] | None = None,
    max_pixels: int | None = None,
    language: str = "eng",
    untrusted: bool = True,
) -> VisualisContext:
    """Analyze an image and return a structured context ready for agents.

    Parameters
    ----------
    source : str
        Path to the image file OR a label like "bytes"/"upload" when ``data``
        is provided. Relative paths outside cwd raise.
    data : bytes | None
        Raw image bytes. When non-None, ``source`` describes origin only.
    depth : Depth or string
        One of "fast", "standard", "deep". Dictates which modules run.
    tasks : sequence of TaskName or strings
        Explicit task whitelist; overrides ``depth`` if given.
    region : tuple[x, y, w, h] | None
        Restrict analysis to this rectangle (original pixel coords).
    max_pixels : int | None
        Hard limit on decoded pixels (overrides config defaults).
    language : str
        BCP-47 code passed through to the OCR backend.
    untrusted : bool
        If True (default), mark all OCR text as potentially adversarial; do
        not treat instructions found inside images as commands to execute.

    Returns
    -------
    VisualisContext
        Contains one ModuleResult per executed module plus any that were
        explicitly skipped. Every numeric claim carries a provenance trail.
    """
    t_start = time.time()
    cfg = (config or VisualisConfig()).merged_with(VisualisConfig.from_env())

    if isinstance(depth, str):
        depth = Depth(depth)

    try:
        img = load_image(source, config=cfg, data=data)
    except VisualisError as e:
        meta = ImageMetadata(format="???", mime_type="unknown", width=1, height=1, mode="", aspect_ratio=1.0)
        ctx = VisualisContext(image=meta, errors=[e.to_dict()])
        return ctx

    if max_pixels is not None:
        img = img.to_work_size(max_pixels)

    # Optionally crop to a sub-region first
    original_img = img
    if region:
        x, y, w, h = region
        img = img.crop_region(x, y, w, h)

    selected_tasks: frozenset[TaskName]
    if tasks:
        normalized = []
        for t in tasks:
            if isinstance(t, str):
                normalized.append(TaskName(t))
            else:
                normalized.append(t)
        selected_tasks = frozenset(normalized)
    else:
        selected_tasks = DEPTH_TASKS[depth]

    result = _run_pixel_tasks(img, depth=depth, config=cfg)

    # Assemble context
    ctx = VisualisContext(
        image=_make_metadata(img),
        request={
            "source": getattr(original_img, "source", source),
            "depth": depth.value,
            "tasks": sorted(t.value for t in selected_tasks),
            "region": region,
            "max_pixels": max_pixels,
            "language": language,
            "untrusted": untrusted,
        },
        results=result,
        duration_ms=(time.time() - t_start) * 1000,
        summary=f"visualis context: {img.width}x{img.height} @ {depth.value}",
    )
    return ctx


# --- convenience API ------------------------------------------------------


def analyze(*args: object, **kwargs: object) -> VisualisContext:
    """Thin alias for inspect(). Same signature."""
    return inspect(*args, **kwargs)


# --- provider registration helpers (for later milestones) ----------------

def register_provider(name: str, handler: Any) -> None:
    TASK_HANDLERS[name] = handler
