"""Unit tests for image loader security and correctness."""
import io
import numpy as np
import pytest
from PIL import Image

from visualis.processing.loader import load_image, load_base64
from visualis.exceptions import ImageTooLargeError, CorruptImageError, UnsupportedFormatError


def make_test_image(width=100, height=100, color=(255, 0, 0)):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:] = color
    return arr


def test_load_png_bytes():
    arr = make_test_image(50, 50, color=(128, 64, 32))
    img = Image.fromarray(arr, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    data = buf.getvalue()

    loaded = load_image('test.png', data=data)
    assert loaded.width == 50
    assert loaded.height == 50
    assert loaded.metadata.mode == 'RGB'


def test_load_jpg():
    arr = make_test_image(64, 64, color=(0, 128, 255))
    img = Image.fromarray(arr, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    data = buf.getvalue()

    loaded = load_image('test.jpg', data=data)
    assert loaded.metadata.format == 'JPEG'


def test_load_webp():
    arr = make_test_image(80, 40, color=(255, 255, 0))
    img = Image.fromarray(arr, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='WEBP')
    data = buf.getvalue()

    loaded = load_image('test.webp', data=data)
    assert loaded.metadata.format == 'WEBP'


def test_empty_image_rejected():
    from visualis.core.config import VisualisConfig
    cfg = VisualisConfig()
    try:
        load_image('empty.png', config=cfg, data=b'')
        assert False, "should reject empty bytes"
    except CorruptImageError:
        pass


def test_invalid_format_rejected():
    try:
        load_image('test.txt', data=b'not an image')
        assert False, "should reject invalid format"
    except UnsupportedFormatError:
        pass
    except CorruptImageError:
        pass  # also acceptable - different error for corrupt vs unsupported


def test_transparent_image():
    arr = np.zeros((50, 50, 4), dtype=np.uint8)
    arr[:, :, 3] = 128  # half transparent
    img = Image.fromarray(arr, 'RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    data = buf.getvalue()

    loaded = load_image('test.png', data=data)
    assert loaded.metadata.transparency is True
    assert loaded.metadata.mode == 'RGBA'


def test_pixel_values_preserved():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[10:20, 10:20] = [255, 0, 0]  # red square
    arr[30:40, 30:40] = [0, 255, 0]  # green square
    img = Image.fromarray(arr, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    data = buf.getvalue()

    loaded = load_image('test.png', data=data)
    pixel = loaded.array[15, 15]
    assert list(pixel) == [255, 0, 0]
    pixel_green = loaded.array[35, 35]
    assert list(pixel_green) == [0, 255, 0]


def test_region_crop():
    arr = make_test_image(200, 150, color=(100, 200, 50))
    img = Image.fromarray(arr, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    data = buf.getvalue()

    loaded = load_image('test.png', data=data)
    cropped = loaded.crop_region(50, 30, 40, 20)
    assert cropped.width == 40
    assert cropped.height == 20
    assert list(cropped.array[0, 0]) == [100, 200, 50]


def test_invalid_region_raises():
    arr = make_test_image(100, 100)
    img = Image.fromarray(arr, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    data = buf.getvalue()

    loaded = load_image('test.png', data=data)
    with pytest.raises(Exception):  # InvalidRegionError or equivalent
        loaded.crop_region(-10, 0, 100, 100)
        loaded.crop_region(0, 0, 200, 100)
