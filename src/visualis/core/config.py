"""Runtime configuration.

Layered, explicit, env-overridable. Precedence (low to high):
defaults < VisualisConfig(...) constructor kwargs < VISUALIS_* env vars.

Environment variables use the ``VISUALIS_`` prefix, e.g. ``VISUALIS_MAX_PIXELS``.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else None


def _env_int(name: str) -> int | None:
    v = _env(name)
    return int(v) if v is not None else None


def _env_float(name: str) -> float | None:
    v = _env(name)
    return float(v) if v is not None else None


def _env_bool(name: str) -> bool | None:
    v = _env(name)
    if v is None:
        return None
    return v.strip().lower() in ("1", "true", "yes", "on")


class ResourceLimits(BaseModel):
    """Security + performance budgets.

    ``max_pixels`` is the decoded-pixel ceiling (decompression-bomb guard).
    Pillow's own max is raised to this value; anything above is rejected.
    """

    max_bytes: int = Field(default=64 * 1024 * 1024, description="max input file size in bytes")
    max_pixels: int = Field(
        default=16_000_000, description="max decoded pixels (w*h) after EXIF orientation"
    )
    max_side: int = Field(default=16_384, description="max single dimension in pixels")
    max_frames: int = Field(default=512, description="max frames for animated images")
    max_temp_dir_bytes: int = Field(default=256 * 1024 * 1024)
    request_timeout_s: float | None = Field(default=None, gt=0)

    @classmethod
    def from_env(cls) -> "ResourceLimits":
        cfg = cls()
        cfg.max_bytes = _env_int("VISUALIS_MAX_BYTES") or cfg.max_bytes
        cfg.max_pixels = _env_int("VISUALIS_MAX_PIXELS") or cfg.max_pixels
        cfg.max_side = _env_int("VISUALIS_MAX_SIDE") or cfg.max_side
        cfg.max_frames = _env_int("VISUALIS_MAX_FRAMES") or cfg.max_frames
        t = _env_float("VISUALIS_REQUEST_TIMEOUT_S")
        if t:
            cfg.request_timeout_s = t
        return cfg


class ProcessingConfig(BaseModel):
    max_work_pixels: int = Field(
        default=2_000_000,
        description="downscale to this many pixels before heavy analysis (OCR/detection)",
    )
    tile_size: int = Field(default=1024, description="tile edge in pixels for DEEP tiling")
    tile_overlap: int = Field(default=64)
    cache: bool = True
    cache_dir: str | None = Field(default=None, description="defaults to a per-user temp dir")
    ocr_language: str = "eng"
    ocr_confidence_floor: float = Field(default=0.5, ge=0.0, le=1.0)
    detection_min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    onnx_intra_threads: int = Field(default=2, ge=1)

    @classmethod
    def from_env(cls) -> "ProcessingConfig":
        cfg = cls()
        cfg.max_work_pixels = _env_int("VISUALIS_MAX_WORK_PIXELS") or cfg.max_work_pixels
        cfg.tile_size = _env_int("VISUALIS_TILE_SIZE") or cfg.tile_size
        if (c := _env("VISUALIS_OCR_LANGUAGE")):
            cfg.ocr_language = c
        cfg.ocr_confidence_floor = (
            _env_float("VISUALIS_OCR_CONFIDENCE_FLOOR") or cfg.ocr_confidence_floor
        )
        cfg.detection_min_confidence = (
            _env_float("VISUALIS_DETECTION_MIN_CONFIDENCE") or cfg.detection_min_confidence
        )
        if (th := _env_int("VISUALIS_ONNX_INTRA_THREADS")):
            cfg.onnx_intra_threads = th
        if (cd := _env("VISUALIS_CACHE_DIR")):
            cfg.cache_dir = cd
        if (c := _env_bool("VISUALIS_CACHE")) is not None:
            cfg.cache = c
        return cfg


class ProviderConfig(BaseModel):
    """Which optional providers are enabled. Everything is opt-in via config
    or env; the base install enables nothing that needs a model or network."""

    mode: str = Field(default="local", pattern="^(local|hybrid|remote)$")
    semantic_provider: str | None = Field(
        default=None, description="e.g. 'openai-vision' or 'anthropic'; None = disabled"
    )
    semantic_model: str | None = None
    detection_model_path: str | None = Field(
        default=None, description="path to an ONNX YOLO object-detection model"
    )
    detection_conf: str | None = Field(
        default=None, description="path to a YOLO data.yaml (class names) for the model"
    )
    segmentation_model_path: str | None = None
    timeout_s: float = Field(default=120.0, gt=0)
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        cfg = cls()
        if (m := _env("VISUALIS_MODE")):
            cfg.mode = m
        if (p := _env("VISUALIS_SEMANTIC_PROVIDER")):
            cfg.semantic_provider = p
        if (m := _env("VISUALIS_SEMANTIC_MODEL")):
            cfg.semantic_model = m
        if (p := _env("VISUALIS_DETECTION_MODEL")):
            cfg.detection_model_path = p
        if (c := _env("VISUALIS_DETECTION_CONF")):
            cfg.detection_conf = c
        if (p := _env("VISUALIS_SEGMENTATION_MODEL")):
            cfg.segmentation_model_path = p
        cfg.timeout_s = _env_float("VISUALIS_PROVIDER_TIMEOUT_S") or cfg.timeout_s
        # API keys are read lazily by the provider itself and never copied here,
        # so they can never appear in config dumps or logs.
        return cfg


class VisualisConfig(BaseModel):
    """Top-level config; composes the three sections above."""

    limits: ResourceLimits = Field(default_factory=ResourceLimits)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    providers: ProviderConfig = Field(default_factory=ProviderConfig)
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "VisualisConfig":
        return cls(
            limits=ResourceLimits.from_env(),
            processing=ProcessingConfig.from_env(),
            providers=ProviderConfig.from_env(),
            log_level=_env("VISUALIS_LOG_LEVEL") or "INFO",
        )

    def merged_with(self, other: "VisualisConfig | None") -> "VisualisConfig":
        """Returns other (shallow-merged on top of self). None is a no-op."""
        if other is None:
            return self
        return other
