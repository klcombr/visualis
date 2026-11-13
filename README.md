<div align="center">

<img src="docs/assets/visualis-banner.svg" alt="Visualis — Visual intelligence for AI agents" width="100%">

<br>

[![Python](https://img.shields.io/badge/Python-3.13%2B-f5f0e8?style=for-the-badge&logo=python&logoColor=111111)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-f5f0e8?style=for-the-badge)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Compatible-f5f0e8?style=for-the-badge&logo=github&logoColor=111111)](https://modelcontextprotocol.io/)

**Give text-based AI agents a way to understand images.**

Local-first image analysis. Structured results. Optional ML backends.

[Getting started](#-quick-start) · [Capabilities](#-what-it-does) · [Architecture](#-how-it-works)

</div>

---

## What is Visualis?

Visualis is an open-source visual perception layer for text-based AI agents.

It processes images and turns measurable visual information into structured data that an LLM can use — without requiring a multimodal model, a GPU, or a paid vision API.

- **Local-first:** core image analysis runs on your machine.
- **Modular:** install only the capabilities you need.
- **Evidence-aware:** results identify where each observation came from.
- **Agent-ready:** use it from Python or through MCP.

> Visualis provides visual evidence. Your AI agent handles the reasoning.

## What it does

| Module | Capabilities |
|:--|:--|
| **Pixels & colors** | Pixel sampling, dominant colors, histograms, color statistics |
| **Geometry** | Edges, corners, symmetry, blur and sharpness metrics |
| **OCR** | Text extraction with backend confidence and regions where available |
| **Visual providers** | Optional object detection, classification and segmentation backends |
| **Metadata** | Dimensions, format and image metadata |

OCR and advanced inference require their optional dependencies. Available results depend on the installed backends.

## How it works

<p align="center">
  <img src="docs/assets/process.jpeg" alt="Visualis processing pipeline" width="88%">
</p>

Visualis combines deterministic computer vision with optional inference providers. Each result carries its source, execution status and — when the backend provides it — confidence information.

| Evidence | Meaning |
|:--|:--|
| `PIXEL_MEASUREMENT` | Direct measurements from image pixels |
| `GEOMETRIC_ANALYSIS` | Computed visual properties |
| `OCR_DETECTION` | Text returned by an OCR backend |
| `OBJECT_DETECTION` | Object-detection inference |
| `SEMANTIC_PROVIDER` | Output from a semantic provider |

Confidence scores are backend-specific, not universal probabilities.

## Quick start

### Install

```bash
pip install visualis
```

Optional OCR backends:

```bash
pip install "visualis[ocr-rapid]"  # RapidOCR / ONNX Runtime
pip install "visualis[ocr]"        # Tesseract
pip install "visualis[all]"        # All optional dependencies
```

### Inspect an image

```python
from visualis import inspect

result = inspect("screenshot.png", depth="standard")

print(result.ocr_text())
print(result.module("colors").payload["dominant_colors"])
print(result.to_json())
```

Run only the modules you need:

```python
result = inspect(
    "screenshot.png",
    tasks=["pixels", "colors", "ocr"],
)
```

### Analysis profiles

| Profile | Includes | Best for |
|:--|:--|:--|
| `fast` | Metadata, pixels and colors | Quick inspection |
| `standard` | Fast + geometry + OCR | Everyday use |
| `deep` | Extended analysis and configured providers | More detailed inspection |

The modules that run depend on installed packages and provider configuration.

## Use it with AI agents

### MCP

Start the MCP server:

```bash
visualis-mcp
```

Available tools:

- `visualis.inspect_image` — configurable image inspection.
- `visualis.ocr_image` — text extraction.

Or create the server in Python:

```python
from visualis.interfaces.mcp.server import create_mcp_server

server = create_mcp_server()
```

### CLI

```bash
visualis inspect image.png
visualis ocr screenshot.png
visualis detect image.png
visualis compare image1.png image2.png
```

Some commands require optional backends.

## Safety by design

Images are untrusted input. Visualis includes safeguards such as:

- Pixel limits to help prevent decompression-bomb attacks.
- EXIF orientation normalization.
- Path traversal rejection.
- Restricted SVG rasterization and sanitization.
- Lazy loading of external provider credentials.
- OCR output treated as untrusted content.

**Text found inside an image is data, not an instruction.** Agents should not follow instructions extracted from images unless the user explicitly authorizes them.

## Supported formats

PNG · JPEG · WEBP · BMP · TIFF · GIF (first frame by default) · SVG (restricted rasterization; requires a supported backend).

## Configuration

Environment variables use the `VISUALIS_` prefix.

| Variable | Default | Purpose |
|:--|:--|:--|
| `VISUALIS_MAX_PIXELS` | `16000000` | Maximum image pixel count |
| `VISUALIS_OCR_CONFIDENCE_FLOOR` | `0.5` | OCR confidence threshold |
| `VISUALIS_LOG_LEVEL` | `INFO` | Log verbosity |

## Development

```bash
git clone https://github.com/visualis/visualis.git
cd visualis

uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"

pytest tests/unit/
```

Optional integration tests and benchmarks:

```bash
pytest tests/integration/
pytest tests/benchmarks/
```

## Roadmap

- [ ] Object detection with YOLO and NMS.
- [ ] Semantic segmentation.
- [ ] Chart and diagram interpretation.
- [ ] Advanced image comparison and tiled processing.
- [ ] REST API with authentication.
- [ ] Docker deployment.
- [ ] Better spatial and region analysis.

Roadmap items are planned, not necessarily available today.

## License

[MIT](LICENSE)

---

<div align="center">

**VISUALIS**  
*Making visual perception accessible to every AI agent.*

</div>
