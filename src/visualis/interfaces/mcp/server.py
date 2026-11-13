"""MCP Server implementation for Visualis.

Exposes tools to AI agents via Model Context Protocol:
- visualis.inspect: Full image inspection with structured output
- visualis.ask: Answer specific questions about an image  
- visualis.ocr: Extract text from image
- visualis.detect: Find objects (when model configured)
- visualis.pixels: Sample colors at coordinates
- visualis.compare: Compare two images
- visualis.crop: Crop region and analyze
- visualis.describe_region: Detailed analysis of sub-region

Configuration: set VISUALIS_MCP_ENABLED=1 to enable MCP server mode.
The server runs as stdio transport by default for direct agent integration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    MCPServer = None


def create_mcp_server() -> MCPServer | None:
    """Create MCP server with Visualis tools registered."""
    if MCPServer is None:
        return None
    
    app = MCPServer(name="visualis", description="Visual intelligence layer for text-based AI agents")
    
    # Register tools
    @app.tool(description="Inspect an image and return structured analysis")
    def inspect_image(
        image_path: str,
        depth: str = "standard",
        tasks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run full inspection on an image file."""
        from visualis import inspect
        from visualis.core.config import VisualisConfig
        
        try:
            ctx = inspect(image_path, depth=depth, tasks=tasks)
            return {
                "status": "success",
                "context_id": ctx.id,
                "summary": ctx.summary,
                "duration_ms": round(ctx.duration_ms, 1),
                "image": ctx.image.model_dump(),
                "results": {k: v.status for k, v in ctx.results.items()},
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    @app.tool(description="Extract text from image using OCR")
    def ocr_image(image_path: str) -> dict[str, Any]:
        """Extract text using Optical Character Recognition."""
        from visualis import inspect
        try:
            ctx = inspect(image_path, tasks=["ocr"])
            ocr_result = ctx.module("ocr")
            if ocr_result and ocr_result.status == "ok":
                payload = ocr_result.payload
                return {
                    "text": payload.get("text", ""),
                    "confidence": payload.get("overall_confidence", 0),
                    "blocks": len(payload.get("blocks", [])),
                }
            return {"text": "", "confidence": 0, "warning": "OCR not available"}
        except Exception as e:
            return {"error": str(e)}
    
    return app


def main() -> int:
    """Entry point for MCP server."""
    if MCPServer is None:
        print("MCP SDK not installed. Install with: pip install 'mcp>=1.0'", file=sys.stderr)
        return 1
    
    app = create_mcp_server()
    if app is None:
        print("Failed to create MCP server", file=sys.stderr)
        return 1
    
    # Run server
    try:
        app.run_stdio_async()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
