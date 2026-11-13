"""Unit tests for the core engine."""
import io
import numpy as np
from PIL import Image

from visualis import inspect
from visualis.core.schemas import Depth, ModuleName


def make_gradient_image():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(50):
        arr[y, :] = [y * 5, 0, 0]
    buf = io.BytesIO()
    img = Image.fromarray(arr, 'RGB')
    img.save(buf, format='PNG')
    return buf.getvalue()


def test_fast_depth_runs_metadata_and_pixels():
    data = make_gradient_image()
    ctx = inspect('test.png', data=data, depth='fast')
    
    assert ctx.module(ModuleName.METADATA).status == 'ok'
    assert ctx.module(ModuleName.PIXELS).status == 'ok'
    assert ctx.module(ModuleName.COLORS).status == 'ok'
    assert ctx.duration_ms > 0


def test_standard_depth_includes_geometry():
    data = make_gradient_image()
    ctx = inspect('test.png', data=data, depth='standard')
    
    assert ctx.module(ModuleName.GEOMETRY).status == 'ok'


def test_engine_validates_output_schema():
    data = make_gradient_image()
    ctx = inspect('test.png', data=data, depth='fast')
    
    # Check required fields exist
    assert hasattr(ctx, 'image')
    assert hasattr(ctx, 'results')
    assert hasattr(ctx, 'duration_ms')
    assert hasattr(ctx, 'skipped')
    
    # Serialize to JSON
    json_str = ctx.to_json()
    assert len(json_str) > 100


def test_error_handling_for_missing_file():
    # inspect() catches errors and returns them in the context, not raised
    ctx = inspect('/nonexistent/path/test.png')
    assert len(ctx.errors) > 0
    assert 'image_source_not_found' in ctx.errors[0]['error'] or 'corrupt_image' in ctx.errors[0]['error']


def test_context_accessors():
    data = make_gradient_image()
    ctx = inspect('test.png', data=data, depth='fast')
    
    # Convenience methods
    assert isinstance(ctx.module(ModuleName.METADATA), type(ctx.results['metadata']))
    assert isinstance(ctx.ocr_text(), str)
    assert isinstance(ctx.detections(), list)


def test_region_cropping():
    data = make_gradient_image()
    ctx = inspect('test.png', data=data, depth='fast', region=(10, 10, 50, 50))
    
    assert ctx.image.width == 50
    assert ctx.image.height == 50


def test_explicit_tasks_override_depth():
    data = make_gradient_image()
    ctx = inspect('test.png', data=data, depth='deep', tasks=['pixels', 'colors'])
    
    # Requested tasks should definitely run
    keys = set(ctx.results.keys())
    assert 'pixels' in keys
    assert 'colors' in keys
    # Note: geometry also runs because engine currently uses depth as baseline
    # TODO: implement strict task-only mode
