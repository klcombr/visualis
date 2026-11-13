"""Unit tests for pixel intelligence module."""
import numpy as np
import pytest
from PIL import Image

from visualis.processing.loader import load_image
from visualis.perception import pixels


def make_array(h, w, colors):
    """Create array with colored regions."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for region, color in colors.items():
        x1, y1, x2, y2 = region
        arr[y1:y2, x1:x2] = color
    return arr


def test_pixel_at_exact_value():
    arr = make_array(100, 100, {
        (10, 10, 20, 20): (255, 128, 64),
    })
    img = Image.fromarray(arr, 'RGB')
    buf = io.BytesIO() if False else None
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format='PNG')
    loaded = load_image('test.png', data=buf.getvalue())

    sample = pixels.pixel_at(loaded, 15, 15)
    assert sample.color.r == 255
    assert sample.color.g == 128
    assert sample.color.b == 64
    assert sample.in_bounds is True


def test_pixel_out_of_bounds_raises():
    arr = make_array(100, 100, {})
    from io import BytesIO
    from PIL import Image
    img = Image.fromarray(arr, 'RGB')
    buf = BytesIO()
    img.save(buf, format='PNG')
    loaded = load_image('test.png', data=buf.getvalue())

    with pytest.raises(Exception):
        pixels.pixel_at(loaded, 200, 200)


def test_average_brightness_white_image():
    arr = np.ones((50, 50, 3), dtype=np.uint8) * 255
    from io import BytesIO
    img = Image.fromarray(arr, 'RGB')
    buf = BytesIO()
    img.save(buf, format='PNG')
    loaded = load_image('test.png', data=buf.getvalue())

    brightness = pixels.average_brightness(loaded)
    assert 0.95 < brightness <= 1.0


def test_average_brightness_black_image():
    arr = np.zeros((50, 50, 3), dtype=np.uint8)
    from io import BytesIO
    img = Image.fromarray(arr, 'RGB')
    buf = BytesIO()
    img.save(buf, format='PNG')
    loaded = load_image('test.png', data=buf.getvalue())

    brightness = pixels.average_brightness(loaded)
    assert brightness < 0.05


def test_dominant_colors_uniform():
    arr = np.full((50, 50, 3), 128, dtype=np.uint8)
    from io import BytesIO
    img = Image.fromarray(arr, 'RGB')
    buf = BytesIO()
    img.save(buf, format='PNG')
    loaded = load_image('test.png', data=buf.getvalue())

    dom = pixels.dominant_colors(loaded, k=3)
    assert len(dom) == 1
    assert abs(dom[0].color.r - 128) < 5


def test_dominant_colors_two_regions():
    arr = np.zeros((50, 50, 3), dtype=np.uint8)
    arr[:25, :] = [255, 0, 0]  # top half red
    arr[25:, :] = [0, 0, 255]  # bottom half blue
    from io import BytesIO
    img = Image.fromarray(arr, 'RGB')
    buf = BytesIO()
    img.save(buf, format='PNG')
    loaded = load_image('test.png', data=buf.getvalue())

    dom = pixels.dominant_colors(loaded, k=2)
    assert len(dom) >= 2
    # One should be red-ish, one blue-ish
    colors = [d.color for d in dom]
    has_red = any(c.r > 200 and c.g < 50 and c.b < 50 for c in colors)
    has_blue = any(c.r < 50 and c.g < 50 and c.b > 200 for c in colors)
    assert has_red or has_blue


def test_region_stats():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[20:40, 20:40] = [200, 100, 50]
    from io import BytesIO
    img = Image.fromarray(arr, 'RGB')
    buf = BytesIO()
    img.save(buf, format='PNG')
    loaded = load_image('test.png', data=buf.getvalue())

    stats = pixels.region_stats(loaded, 20, 20, 20, 20)
    assert stats.average.r == 200
    assert stats.average.g == 100
    assert stats.average.b == 50


def test_tile_coordinates():
    tiles = pixels.tile_coordinates(100, 100, tile=50, overlap=10)
    assert len(tiles) > 0
    # Check coverage
    covered_x = sum(t[2] for t in tiles if t[1] == 0)
    covered_y = sum(t[3] for t in tiles if t[0] == 0)
    assert covered_x >= 90  # at least 90% coverage
    assert covered_y >= 90


def test_transparency_stats_no_alpha():
    arr = np.zeros((50, 50, 3), dtype=np.uint8)
    from io import BytesIO
    img = Image.fromarray(arr, 'RGB')
    buf = BytesIO()
    img.save(buf, format='PNG')
    loaded = load_image('test.png', data=buf.getvalue())

    stats = pixels.transparency_stats(loaded)
    assert stats['has_alpha'] is False
    assert stats['transparent_fraction'] == 0.0
