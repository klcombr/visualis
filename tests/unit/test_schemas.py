"""Unit tests for schemas."""
from visualis.core.schemas import Box, Color, Point


def test_box_creation():
    b = Box(x=10, y=20, width=100, height=50)
    assert b.to_list() == [10, 20, 100, 50]
    assert b.center == (60.0, 45.0)


def test_box_zero_size_rejected():
    from pydantic import ValidationError
    try:
        Box(x=0, y=0, width=0, height=10)
        assert False, "should reject zero size"
    except ValidationError:
        pass


def test_box_iou():
    a = Box(x=0, y=0, width=100, height=100)
    b = Box(x=50, y=0, width=100, height=100)
    # Intersection is 50x100 = 5000, union is 15000, so IoU = 1/3
    assert abs(a.iou(b) - 1/3) < 0.01


def test_box_intersects():
    a = Box(x=0, y=0, width=100, height=100)
    b = Box(x=50, y=50, width=100, height=100)
    c = Box(x=200, y=200, width=50, height=50)
    assert a.intersects(b)
    assert not a.intersects(c)


def test_color_to_hex():
    c = Color(r=255, g=0, b=128)
    assert c.to_hex() == "#ff0080"
    c_rgba = Color(r=255, g=0, b=128, a=128)
    assert c_rgba.to_hex() == "#ff008080"


def test_point():
    p = Point(x=100, y=200)
    assert p.x == 100
    assert p.y == 200
