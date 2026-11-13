"""Safe image loading.

This module is the security boundary for all pixel data. Responsibilities:

* byte-size and decoded-pixel limits (decompression-bomb protection)
* path-traversal protection for ``file://`` / relative paths
* EXIF orientation correction (pixel space == visual space thereafter)
* animated GIF frame handling (first-frame by default, all on request)
* SVG rasterisation via a safe pipeline: XML-level allowlist + subprocess
  rasteriser (rsvg-convert / inkscape) OR cairosvg with a no-external-script
  guard; never via a browser engine.

The returned :class:`LoadedImage` is the ONLY object modules receive. All
coordinates are in the orientation-corrected pixel space.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from visualis.core.config import VisualisConfig
from visualis.core.schemas import ImageMetadata
from visualis.exceptions import (
    CorruptImageError,
    ImageSourceNotFound,
    ImageTooLargeError,
    InvalidRegionError,
    PathTraversalError,
    UnsafeSvgError,
    UnsupportedFormatError,
)

SUPPORTED_STATIC = {"PNG", "JPEG", "WEBP", "BMP", "TIFF", "GIF", "SVG"}
MIME_BY_FORMAT = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
    "GIF": "image/gif",
    "SVG": "image/svg+xml",
}

# SVG: refuse anything that can execute, import remote content, or escape the
# sandbox. Allowed: pure presentation markup.
_SVG_UNSAFE_PATTERNS = re.compile(
    r"<\s*(script|foreignObject)\b|on\w+\s*=|xlink:href\s*=\s*[\"']data:|"
    r"xlink:href\s*=\s*[\"']\s*javascript:|href\s*=\s*[\"']\s*(data:|javascript:)|"
    r"\bimport\b.*\.module|\.load\s*\(",
    re.IGNORECASE,
)
_SVG_TAG_ALLOWLIST = {
    "svg", "g", "defs", "symbol", "use", "path", "rect", "circle", "ellipse",
    "line", "polyline", "polygon", "text", "tspan", "title", "desc", "linearGradient",
    "radialGradient", "stop", "clipPath", "mask", "pattern", "image", "marker",
    "filter", "feGaussianBlur", "feOffset", "feMerge", "feMergeNode", "feFlood",
    "feComposite", "feBlend", "animate", "animateTransform", "a",
}


def _check_svg(source: str) -> None:
    if _SVG_UNSAFE_PATTERNS.search(source):
        raise UnsafeSvgError("SVG contains a construct that is not allowed")
    tags = {t for t in re.findall(r"<\s*([a-zA-Z][\w:-]*)", source)}
    bad = tags - _SVG_TAG_ALLOWLIST
    if bad:
        raise UnsafeSvgError(f"SVG contains disallowed tags: {sorted(bad)[:8]}")


@dataclass
class LoadedImage:
    """A decoded image plus everything modules need about it.

    ``array`` is ``HxW`` or ``HxWxC`` uint8, RGB(A) — never BGR, never float,
    so pixel math is predictable everywhere.
    """

    image: Image.Image
    array: np.ndarray
    metadata: ImageMetadata
    source: str  # description of origin (path / "bytes" / "url:<hash8>")
    content_hash: str = field(repr=False)
    frame: int = 0
    _tmp_path: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        Image.MAX_IMAGE_PIXELS = None  # limits enforced here, not by Pillow

    @property
    def width(self) -> int:
        return self.metadata.width

    @property
    def height(self) -> int:
        return self.metadata.height

    def close(self) -> None:
        self.image.close()
        if self._tmp_path is not None:
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass

    def as_context_manager(self) -> "LoadedImage":
        return self

    def __enter__(self) -> "LoadedImage":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- region helpers ----------------------------------------------------

    def crop_region(self, x: int, y: int, w: int, h: int) -> "LoadedImage":
        """Return a new LoadedImage for a sub-rectangle (coords in original space)."""
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > self.width or y + h > self.height:
            raise InvalidRegionError(
                f"region ({x},{y},{w},{h}) outside {self.width}x{self.height} image"
            )
        sub = self.image.crop((x, y, x + w, y + h))
        arr = np.asarray(sub)
        meta = ImageMetadata(
            format=self.metadata.format,
            mime_type=self.metadata.mime_type,
            width=w,
            height=h,
            mode=sub.mode,
            size_bytes=None,
            file_path=None,
            transparency=self.metadata.transparency,
            orientation_corrected=self.metadata.orientation_corrected,
            aspect_ratio=w / max(1, h),
        )
        return LoadedImage(
            image=sub,
            array=arr,
            metadata=meta,
            source=f"{self.source}#crop({x},{y},{w},{h})",
            content_hash=self.content_hash[:16] + "-crop",
        )

    def to_work_size(self, max_pixels: int) -> "LoadedImage":
        """Scaled copy (Lanczos) with at most ``max_pixels`` total pixels."""
        total = self.width * self.height
        if total <= max_pixels:
            return self
        scale = (max_pixels / total) ** 0.5
        w = max(1, int(self.width * scale))
        h = max(1, int(self.height * scale))
        sub = self.image.resize((w, h), Image.LANCZOS)
        arr = np.asarray(sub)
        meta = self.metadata.model_copy(
            update={
                "width": w,
                "height": h,
                "aspect_ratio": w / max(1, h),
                "mode": sub.mode,
            }
        )
        return LoadedImage(
            image=sub,
            array=arr,
            metadata=meta,
            source=f"{self.source}#scale({w}x{h})",
            content_hash=self.content_hash + "-scale",
        )


def _read_svg(path: Path, dest_dir: str, size_bytes: int) -> LoadedImage:
    raw = path.read_bytes()
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise UnsafeSvgError("SVG is not valid UTF-8") from e
    _check_svg(source)

    tmp_svg = os.path.join(dest_dir, "input.svg")
    tmp_png = os.path.join(dest_dir, "raster.png")
    Path(tmp_svg).write_bytes(raw)

    rsvg = shutil.which("rsvg-convert")
    inkscape = shutil.which("inkscape")
    used = ""
    if rsvg:
        proc = subprocess.run(
            [rsvg, "--format=png", "--keep-aspect-ratio", "-o", tmp_png, tmp_svg],
            capture_output=True,
            timeout=30,
        )
        used = "rsvg-convert"
    elif inkscape:
        proc = subprocess.run(
            [inkscape, "--export-type=png", f"--export-filename={tmp_png}", tmp_svg],
            capture_output=True,
            timeout=60,
        )
        used = "inkscape"
    else:
        try:
            import cairosvg  # type: ignore

            cairosvg.svg2png(bytestring=raw, write_to=tmp_png, output_width=None)
            used = "cairosvg"
        except ImportError as e:
            raise UnsupportedFormatError(
                "No SVG rasteriser available. Install one of: rsvg-convert, inkscape, "
                "pip install cairosvg"
            ) from e

    if used in ("rsvg-convert", "inkscape") and proc.returncode != 0:
        raise CorruptImageError(f"SVG rasterisation failed: {proc.stderr.decode()[:200]}")
    if not Path(tmp_png).exists():
        raise CorruptImageError("SVG rasteriser produced no output")
    out_bytes = Path(tmp_png).read_bytes()

    img = _decode_bytes(
        out_bytes,
        limits_max_pixels=16_000_000,
        max_side=16_384,
        mime_hint="image/png",
        source=f"svg:{used}",
    )
    img.metadata.size_bytes = size_bytes
    img.metadata.format = "SVG"
    img.metadata.mime_type = "image/svg+xml"
    return img


def _decode_bytes(
    data: bytes,
    *,
    limits_max_pixels: int,
    max_side: int,
    mime_hint: str | None = None,
    source: str = "bytes",
) -> LoadedImage:
    buf = io.BytesIO(data)
    try:
        with Image.open(buf) as probe:
            fmt = probe.format
            # orientation-corrected size (what the pixels will actually be)
            corrected = ImageOps.exif_transpose(probe) or probe
            w, h = corrected.size
    except UnidentifiedImageError as e:
        raise CorruptImageError("Cannot identify image format") from e
    except Exception as e:  # struct, zlib, ...
        raise CorruptImageError(f"Failed to parse image: {type(e).__name__}") from e

    if fmt not in SUPPORTED_STATIC:
        raise UnsupportedFormatError(f"format '{fmt}' not supported (have: {sorted(SUPPORTED_STATIC)})")

    if w > max_side or h > max_side:
        raise ImageTooLargeError(f"side {max(w, h)}px exceeds limit {max_side}px")
    if w * h > limits_max_pixels:
        raise ImageTooLargeError(
            f"decoded size {w}x{h}={w * h} px exceeds limit {limits_max_pixels} px"
        )

    # full decode (forces bomb materialization inside the guard above)
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            frames = getattr(img, "n_frames", 1)
            duration = img.info.get("duration", 0) or 0
            fps = (1000.0 / float(duration)) if duration else None
            exif = {}
            try:
                raw_exif = img.getexif()
                for k, v in raw_exif.items():
                    try:
                        v = str(v)
                    except Exception:
                        continue
                    if len(v) < 256:
                        exif[str(k)] = v
            except Exception:
                pass
            orientation_tag = int(exif.get("274", 0) or 0)
            corrected = ImageOps.exif_transpose(img)
            orientation_corrected = corrected is not img and orientation_tag != 1
            has_alpha = corrected.mode in ("RGBA", "LA", "PA") or (
                corrected.mode == "P" and "transparency" in corrected.info
            )
            corrected = corrected.convert("RGBA" if has_alpha else "RGB")
            arr = np.asarray(corrected)
            meta = ImageMetadata(
                format=fmt,
                mime_type=MIME_BY_FORMAT.get(fmt, mime_hint or "application/octet-stream"),
                width=corrected.width,
                height=corrected.height,
                mode=corrected.mode,
                size_bytes=len(data),
                frames=frames,
                fps=fps,
                transparency=has_alpha,
                orientation_corrected=orientation_corrected,
                exif=exif,
                icc_profile_present="icc_profile" in corrected.info,
                aspect_ratio=corrected.width / max(1, corrected.height),
            )
    except (OSError, MemoryError, Image.DecompressionBombError, ValueError) as e:
        raise CorruptImageError(f"Image failed full decode: {type(e).__name__}") from e

    return LoadedImage(
        image=corrected,
        array=arr,
        metadata=meta,
        source=source,
        content_hash=hashlib.sha256(data).hexdigest(),
    )


def load_image(
    source: str,
    *,
    config: VisualisConfig | None = None,
    data: bytes | None = None,
) -> LoadedImage:
    """Load an image from a file path or raw bytes.

    ``source`` may be an absolute/relative path, ``file:///abs/path``, or a
    label like ``"bytes"`` / ``"upload"`` when ``data`` is provided. Relative
    paths are resolved against the working directory; ``..`` segments are
    rejected to keep API surfaces honest about where they read from.
    """
    cfg = config or VisualisConfig()
    limits = cfg.limits

    if data is not None:
        if len(data) > limits.max_bytes:
            raise ImageTooLargeError(f"{len(data)} bytes exceeds max {limits.max_bytes}")
        return _decode_bytes(
            data,
            limits_max_pixels=limits.max_pixels,
            max_side=limits.max_side,
            source=source or "bytes",
        )

    path = Path(source).expanduser()
    if not path.is_absolute():
        # reject traversal out of cwd for API/CLI callers
        if ".." in path.parts:
            raise PathTraversalError(f"refusing to traverse outside working dir: {source}")
        path = Path.cwd() / path

    if not path.exists() or not path.is_file():
        raise ImageSourceNotFound(f"no such image file: {path}")

    size = path.stat().st_size
    if size > limits.max_bytes:
        raise ImageTooLargeError(f"file is {size} bytes, max is {limits.max_bytes}")
    if size == 0:
        raise CorruptImageError("file is empty")

    suffix = path.suffix.lower()
    if suffix == ".svg":
        with tempfile.TemporaryDirectory(prefix="visualis-svg-") as td:
            img = _read_svg(path, td, size)
            img.source = str(path)
            return img

    raw = path.read_bytes()
    img = _decode_bytes(
        raw,
        limits_max_pixels=limits.max_pixels,
        max_side=limits.max_side,
        source=str(path),
    )
    return img


def load_base64(data_b64: str, config: VisualisConfig | None = None) -> LoadedImage:
    """Load from a base64 payload (REST API / MCP use case)."""
    try:
        raw = base64.b64decode(data_b64, validate=True)
    except (ValueError, TypeError) as e:
        raise CorruptImageError("not valid base64") from e
    return load_image("base64", config=config, data=raw)
