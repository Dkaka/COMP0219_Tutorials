"""
Draws the Wind Tunnel Logger logo: a black rounded square with "Wind Tunnel Logger", three wind streamlines and
"COMP0219".
Writes src/wind_tunnel_logger/icon.png (window icon), icon.ico (exe icon, 16-256 px) and docs/img/logo.png.

    .venv\\Scripts\\python.exe tools\\make_logo.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "wind_tunnel_logger"
FONT = Path(r"C:\Windows\Fonts\segoeuib.ttf")    # Segoe UI Bold
BG, FG, WIND = (17, 17, 17), (255, 255, 255), (0, 200, 220)     # WIND = the app's tunnel colour
S = 1024                                          # drawn at this size, then scaled down


def text_centered(d, y, text, size, color=FG):
    font = ImageFont.truetype(str(FONT), size)
    left, top, right, bottom = d.textbbox((0, 0), text, font=font)
    d.text(((S - (right - left)) / 2 - left, y - top), text, font=font, fill=color)


def draw(text=True):
    """text=False: the streamlines alone, for 16-32 px icons where no text can be read."""
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, S - 1, S - 1), radius=190, fill=BG)
    if text:
        text_centered(d, 140, "Wind Tunnel", 128)
        text_centered(d, 290, "Logger", 128)
    w = 46 if text else 100                        # streamlines: staggered rounded strokes, centred as a group
    y0, gap = (535, 76) if text else (300, 212)
    for i, (x0, x1) in enumerate([(190, 610), (300, 834), (190, 530)]):
        y = y0 + i * gap
        d.line((x0, y, x1, y), fill=WIND, width=w)
        for x in (x0, x1):
            d.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=WIND)
    if text:
        text_centered(d, 815, "COMP0219", 112, (175, 175, 175))
    return im


def main():
    full, small = draw(True), draw(False)
    PKG.mkdir(exist_ok=True)
    full.resize((256, 256), Image.LANCZOS).save(PKG / "icon.png")
    full.resize((512, 512), Image.LANCZOS).save(ROOT / "docs" / "img" / "logo.png")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [(small if n <= 32 else full).resize((n, n), Image.LANCZOS) for n in sizes]
    frames[-1].save(PKG / "icon.ico", sizes=[(n, n) for n in sizes], append_images=frames[:-1])
    print("wrote", PKG / "icon.png", PKG / "icon.ico", ROOT / "docs" / "img" / "logo.png")


if __name__ == "__main__":
    main()
