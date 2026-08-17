#!/usr/bin/env python3
"""Generate the ManipuLens extension icon set (16/32/48/128 px PNGs).

Design: a magnifying lens sweeping over three "headline" bars on a deep-navy
rounded tile; the middle bar glows amber inside the lens — the moment a
manipulative headline is spotted. Drawn at 1024 px with supersampling, then
downscaled with Lanczos so the small sizes stay crisp.

  python tools/make_icon.py     # writes apps/extension/assets/icons/icon{N}.png

Requires: pip install pillow
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

EXT = Path(__file__).resolve().parents[1] / "apps" / "extension"
OUT = EXT / "assets" / "icons"
SIZES = (16, 32, 48, 128)

S = 1024  # master canvas

# palette
NAVY_TOP = (24, 32, 54)
NAVY_BOT = (13, 18, 33)
BAR = (148, 163, 194)  # muted slate — ordinary headlines
BAR_HOT = (245, 166, 35)  # amber — the flagged headline
LENS_RING = (233, 238, 248)
LENS_GLASS = (36, 48, 78)
HANDLE = (233, 238, 248)


def rounded_tile(d: ImageDraw.ImageDraw) -> None:
    # vertical gradient inside a rounded square
    r = 224
    grad = Image.new("RGB", (1, S))
    for y in range(S):
        t = y / (S - 1)
        grad.putpixel(
            (0, y),
            tuple(int(a + (b - a) * t) for a, b in zip(NAVY_TOP, NAVY_BOT, strict=True)),
        )
    grad = grad.resize((S, S))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=r, fill=255)
    d._image.paste(grad, (0, 0), mask)


def bars(d: ImageDraw.ImageDraw) -> None:
    x0, w, h, r = 176, 512, 74, 37
    ys = (268, 476, 684)
    widths = (w, int(w * 0.86), int(w * 0.68))
    for y, wi, color in zip(ys, widths, (BAR, BAR_HOT, BAR), strict=True):
        d.rounded_rectangle([x0, y, x0 + wi, y + h], radius=r, fill=color)


def lens(d: ImageDraw.ImageDraw) -> None:
    cx, cy, radius, ring = 618, 512, 250, 58
    # glass (slightly translucent navy so bars read through)
    glass = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glass)
    gd.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=(*LENS_GLASS, 110),
    )
    d._image.alpha_composite(glass)
    # re-draw the hot bar segment inside the lens, brighter (magnified find)
    x0, w, h, r = 176, 512, 74, 37
    y = 476
    hot = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hot)
    hd.rounded_rectangle([x0, y, x0 + int(w * 0.86), y + h], radius=r, fill=(*BAR_HOT, 255))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse(
        [cx - radius + 8, cy - radius + 8, cx + radius - 8, cy + radius - 8], fill=255
    )
    d._image.paste(hot, (0, 0), Image.composite(hot.split()[3], Image.new("L", (S, S), 0), mask))
    # ring
    d.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline=LENS_RING,
        width=ring,
    )
    # handle at 45°
    hw = 64
    d.line(
        [cx + int(radius * 0.72), cy + int(radius * 0.72), cx + 445, cy + 445],
        fill=HANDLE,
        width=hw * 2,
    )
    d.ellipse([cx + 445 - hw, cy + 445 - hw, cx + 445 + hw, cy + 445 + hw], fill=HANDLE)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rounded_tile(d)
    bars(d)
    lens(d)
    for n in SIZES:
        img.resize((n, n), Image.LANCZOS).save(OUT / f"icon{n}.png")
        print(f"assets/icons/icon{n}.png")


if __name__ == "__main__":
    main()
