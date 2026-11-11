from PIL import Image, ImageDraw
import io


def test_preprocess_returns_png():
    from webapp.api import preprocess_for_ocr

    img = Image.new("RGB", (100, 100), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.line((10, 10, 90, 90), fill=(0, 0, 0), width=5)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    result = preprocess_for_ocr(img_bytes)

    assert result[:8] == b'\x89PNG\r\n\x1a\n'
    assert len(result) > 100


def test_check_empty_canvas():
    from webapp.api import check_empty_canvas

    img = Image.new("RGB", (100, 100), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    assert check_empty_canvas(buf.getvalue()) == True

    img2 = Image.new("RGB", (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img2)
    draw.line((10, 10, 90, 90), fill=(0, 0, 0), width=5)
    buf2 = io.BytesIO()
    img2.save(buf2, format="PNG")
    assert check_empty_canvas(buf2.getvalue()) == False