"""RapidOCR return format verification."""
from PIL import ImageDraw
from rapidocr_onnxruntime import RapidOCR


def test_rapidocr_format():
    """Show exact structure returned by RapidOCR."""
    img = ImageDraw.Image.new('RGB', (420, 120), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 45), 'VISUALIS TEST TEXT', fill=(0, 0, 0))

    engine = RapidOCR(rec_model_path=None, det_model_path=None, cls_model_path=None)
    outer = engine(img)

    # First element should be list of [box, text, conf] tuples
    items = outer[0]
    assert isinstance(items, list), f"expected list, got {type(items)}"
    if not items:
        return

    first = items[0]
    assert isinstance(first, list), f"expected list for first item, got {type(first)}"
    assert len(first) >= 3, f"expected at least 3 elements, got {len(first)}"

    box, text, conf = first[0], first[1], first[2]
    print(f'box={box}, text={text!r}, conf={conf}')


if __name__ == "__main__":
    test_rapidocr_format()
