from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

def draw_boxes(image: Image.Image, boxes: List[Tuple[float, float, float, float]], labels: List[str], colors: List[str] = None, line_width: int = 3, font_size: int = 16) -> Image.Image:
    """Draw bounding boxes with labels on a PIL image.

    Args:
        image: PIL Image to annotate.
        boxes: List of (x0, y0, x1, y1) in pixel coordinates.
        labels: Corresponding list of label strings.
        colors: List of color strings for each box. Defaults to red if not provided.
        line_width: Width of the bounding box lines.
        font_size: Size of the label font.
    Returns:
        Annotated PIL Image.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except Exception:
        # Fallback to default font with size support (Pillow 10+)
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            # Fallback for older Pillow versions (though we confirmed 12.0.0)
            font = ImageFont.load_default()
    
    if colors is None:
        colors = ["red"] * len(boxes)

    for (x0, y0, x1, y1), label, color in zip(boxes, labels, colors):
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
        text = label
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_bg = [x0, y0 - text_height, x0 + text_width, y0]
        draw.rectangle(text_bg, fill=color)
        draw.text((x0, y0 - text_height), text, fill="white", font=font)
    return image
