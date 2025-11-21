from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

def draw_boxes(image: Image.Image, boxes: List[Tuple[float, float, float, float]], labels: List[str]) -> Image.Image:
    """Draw bounding boxes with labels on a PIL image.

    Args:
        image: PIL Image to annotate.
        boxes: List of (x0, y0, x1, y1) in pixel coordinates.
        labels: Corresponding list of label strings.
    Returns:
        Annotated PIL Image.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()
    for (x0, y0, x1, y1), label in zip(boxes, labels):
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        text = label
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_bg = [x0, y0 - text_height, x0 + text_width, y0]
        draw.rectangle(text_bg, fill="red")
        draw.text((x0, y0 - text_height), text, fill="white", font=font)
    return image
