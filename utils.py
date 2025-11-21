from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

def draw_boxes(image: Image.Image, boxes: List[Tuple[float, float, float, float]], labels: List[str], colors: List[str] = None, line_width: int = 3, font_size: int = 16) -> Image.Image:
    """PIL画像にラベル付きのバウンディングボックスを描画します。

    Args:
        image: 注釈を付けるPIL画像。
        boxes: ピクセル座標の(x0, y0, x1, y1)のリスト。
        labels: 対応するラベル文字列のリスト。
        colors: 各ボックスの色文字列のリスト。指定されない場合は赤がデフォルトになります。
        line_width: バウンディングボックスの線の太さ。
        font_size: ラベルフォントのサイズ。
    Returns:
        注釈付きのPIL画像。
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except Exception:
        # Pillow 10+ でサイズサポート付きのデフォルトフォントにフォールバック
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            # 古いPillowバージョンのためのフォールバック（12.0.0を確認済みだが念のため）
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
