"""Visualis exceptions.

Every failure surfaces as a typed exception. There is no silent failure path:
modules either produce a result, produce a result with explicit skips, or raise.
"""

from __future__ import annotations


class VisualisError(Exception):
    """Base class for all Visualis errors."""

    #: machine-readable code, safe to expose over MCP/REST
    code: str = "visualis_error"

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_dict(self) -> dict[str, object]:
        return {"error": self.code, "type": type(self).__name__, "message": self.message, **self.context}


# --- input / security ------------------------------------------------------


class InputError(VisualisError):
    code = "invalid_input"


class ImageSourceNotFound(InputError):
    code = "image_source_not_found"


class UnsupportedFormatError(InputError):
    code = "unsupported_format"


class ImageTooLargeError(InputError):
    """Byte-size or decoded-pixel limits were exceeded (decompression-bomb guard)."""

    code = "image_too_large"


class CorruptImageError(InputError):
    code = "corrupt_image"


class PathTraversalError(InputError):
    code = "path_traversal"


class UnsafeSvgError(InputError):
    """SVG contained a construct we refuse to rasterise."""

    code = "unsafe_svg"


class AuthorizationError(VisualisError):
    code = "unauthorized"


class ResourceLimitError(VisualisError):
    """A runtime budget (memory, time, pixels) was hit mid-analysis."""

    code = "resource_limit"


# --- capability / configuration -------------------------------------------


class DependencyMissingError(VisualisError):
    """An optional backend is not installed.

    Callers normally do *not* catch this: the orchestrator converts it into a
    ModuleSkip so partial results still arrive.
    """

    code = "dependency_missing"

    def __init__(self, dependency: str, purpose: str, install_hint: str = "") -> None:
        hint = f" Install with: {install_hint}" if install_hint else ""
        super().__init__(f"'{dependency}' is required for {purpose}.{hint}", dependency=dependency, purpose=purpose)
        self.dependency = dependency
        self.purpose = purpose
        self.install_hint = install_hint


class ModuleUnavailableError(VisualisError):
    code = "module_unavailable"


class InvalidRegionError(InputError):
    code = "invalid_region"


class ProviderError(VisualisError):
    code = "provider_error"


class ProviderAuthError(ProviderError):
    code = "provider_auth"


class ProviderTimeout(ProviderError):
    code = "provider_timeout"


class ModelNotInstalledError(VisualisError):
    """A provider capability is configured but its weights are absent."""

    code = "model_not_installed"


class SchemaValidationError(VisualisError):
    code = "schema_validation"
