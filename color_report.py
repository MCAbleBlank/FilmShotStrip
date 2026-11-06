#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
color_report.py — 从「每秒图」生成电影色调报告

输入一张每秒图（宽 = 秒数 × 每秒像素，例如 film_shot_strip.py --per-second 1），
输出：
  - 矢量示波器（类似 DaVinci Resolve 的 Vectorscope：圆形刻度 + 十字线 +
    R/Mg/B/Cy/G/Yl 目标 + UV 色度云）
  - RGB / 亮度直方图
  - 一张包含两者和色调时间线的报告图

用法：
  python color_report.py out/liz_persecond.png -o out/liz_color_report.png \
      --title "Liz to aoi tori (2018)"

只依赖 numpy + Pillow。
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SS = 2  # 超采样倍数

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BG = (14, 14, 18, 255)
PANEL = (22, 22, 27, 255)
GRID = (46, 46, 56, 255)
TEXT = (232, 232, 238, 255)
MUTED = (150, 150, 162, 255)
DIM = (100, 100, 112, 255)
CH_RGB = ((255, 86, 86), (86, 225, 120), (86, 150, 255))

HUE_NAMES = [
    "Red", "Orange", "Yellow", "Lime", "Green", "Spring",
    "Cyan", "Azure", "Blue", "Violet", "Magenta", "Rose",
]

# 标准矢量示波器目标角度（度，0° = 3 点钟，逆时针）
VECTOR_TARGETS = [
    ("R", 103.5, (255, 74, 74)),
    ("Mg", 60.7, (235, 82, 225)),
    ("B", 347.1, (86, 122, 255)),
    ("Cy", 283.5, (82, 222, 232)),
    ("G", 240.7, (92, 224, 112)),
    ("Yl", 167.1, (238, 222, 92)),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size=size)


def rgb_to_hsv(rgb: np.ndarray):
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    diff = mx - mn
    sat = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    hue = np.zeros_like(mx)
    m = diff > 1e-6
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    idx = m & (mx == r)
    hue[idx] = ((g - b)[idx] / diff[idx]) % 6.0
    idx = m & (mx == g) & ~(mx == r)
    hue[idx] = ((b - r)[idx] / diff[idx]) + 2.0
    idx = m & (mx == b) & ~(mx == r) & ~(mx == g)
    hue[idx] = ((r - g)[idx] / diff[idx]) + 4.0
    return (hue / 6.0) % 1.0, np.clip(sat, 0, 1), np.clip(mx, 0, 1)


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    i = int(h * 6.0) % 6
    f = h * 6.0 - math.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return int(r * 255), int(g * 255), int(b * 255)


def hsv_to_rgb_arr(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1)


def hex_of(rgb: np.ndarray) -> str:
    r, g, b = [int(round(c)) for c in np.clip(rgb, 0, 255)]
    return f"#{r:02x}{g:02x}{b:02x}"


def analyze(image_path: str, px_per_second: int = 1, crop_bottom: int = 0) -> dict:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    if crop_bottom > 0:
        arr = arr[: max(2, arr.shape[0] - crop_bottom)]
    height, width, _ = arr.shape

    per = arr.mean(axis=0)  # 每一列（每秒）的纵向平均色
    if px_per_second > 1:
        n = width // px_per_second
        if n >= 1:
            per = per[: n * px_per_second].reshape(n, px_per_second, 3).mean(axis=1)
    per = per.astype(np.float32)

    rgb01 = per / 255.0
    hue, sat, val = rgb_to_hsv(rgb01)
    luma = rgb01 @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    neutral = (sat < 0.10) | (val < 0.06)
    weights = np.where(neutral, 0.0, sat * np.clip(val, 0, 1))
    hues = np.floor(hue * 12).astype(int) % 12
    shares = np.zeros(12, dtype=np.float64)
    for k in range(12):
        shares[k] = weights[hues == k].sum()
    total = shares.sum()
    shares = shares / total if total > 0 else np.zeros(12)
    dominant = int(np.argmax(shares)) if total > 0 else -1

    # 矢量示波器用的点云：对整张每秒图降采样
    step_y = max(1, height // 300)
    step_x = max(1, width // 2000)
    sample = arr[::step_y, ::step_x] / 255.0
    sample = sample.reshape(-1, 3)
    y = sample @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    u = 0.493 * (sample[:, 2] - y)
    v = 0.877 * (sample[:, 0] - y)

    bins = 64
    hists = {}
    for name, values in (("r", per[:, 0]), ("g", per[:, 1]), ("b", per[:, 2]),
                         ("luma", luma * 255.0)):
        h, _ = np.histogram(values, bins=bins, range=(0, 255))
        hists[name] = h.astype(np.float32)
    peak = max(float(h.max()) for h in hists.values()) or 1.0
    for k in hists:
        hists[k] /= peak

    return {
        "path": image_path,
        "width": width,
        "height": height,
        "seconds": len(per),
        "per": per,
        "hue": hue,
        "sat": sat,
        "val": val,
        "luma": luma,
        "neutral_share": float(neutral.mean()) if len(neutral) else 0.0,
        "shares": shares,
        "dominant": dominant,
        "hists": hists,
        "mean_rgb": per.mean(axis=0),
        "mean_luma": float(luma.mean()) if len(luma) else 0.0,
        "mean_sat": float(sat.mean()) if len(sat) else 0.0,
        "cloud_rgb": sample,
        "cloud_u": u,
        "cloud_v": v,
    }


def panel(size: tuple[int, int], title: str) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    w, h = size
    img = Image.new("RGBA", (w * SS, h * SS), PANEL)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w * SS - 1, h * SS - 1], outline=(38, 38, 47, 255), width=SS)
    d.text((24 * SS, 18 * SS), title, font=font(17 * SS, True), fill=TEXT)
    return img, d, 52 * SS


def _conv2(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """可分离卷积，a 可以是 (h,w) 或 (h,w,3)。"""
    def conv_axis(arr, kernel, axis):
        pad = len(kernel) // 2
        pads = [(pad, pad) if i == axis else (0, 0) for i in range(arr.ndim)]
        ap = np.pad(arr, pads, mode="edge")
        out = np.zeros_like(arr)
        n = arr.shape[axis]
        for i, ki in enumerate(kernel):
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(i, i + n)
            out += ki * ap[tuple(sl)]
        return out
    return conv_axis(conv_axis(a, k, 0), k, 1)


def _cloud_layer(u: np.ndarray, v: np.ndarray, rgb: np.ndarray,
                 cx: float, cy: float, R: float, gain: float,
                 alpha_scale: float = 1.0, sat_scale: float = 1.0,
                 white_scale: float = 1.0, density_gamma: float = 0.55) -> tuple[Image.Image, float]:
    """把 UV 点云渲染成带辉光的 RGBA 层（高分辨率双线性泼溅 + 高斯平滑）。"""
    res = 900
    rmag = np.hypot(u, v)
    p_fit = float(np.percentile(rmag, 99.9))
    if gain <= 0:
        # 按 99.9 百分位适配到半径 90%，给外圈留余量，避免顶到圆环
        gain = min(6.0, 0.98 / (0.96 * max(p_fit, 1e-4)))

    half = res / 2
    x = half + u * gain * half * 0.96
    y = half - v * gain * half * 0.96
    inside = (x >= 0) & (x < res - 1) & (y >= 0) & (y < res - 1)
    if not inside.any():
        return Image.new("RGBA", (res, res), (0, 0, 0, 0)), gain
    x = x[inside]
    y = y[inside]
    rgb = rgb[inside]

    # 双线性泼溅，先得到细腻的密度/颜色场
    fx = x - 0.5
    fy = y - 0.5
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    wx = (fx - x0).astype(np.float32)
    wy = (fy - y0).astype(np.float32)

    # 双线性泼溅需要 4 个邻格，落在网格外的点直接丢弃，避免负索引
    valid = (x0 >= 0) & (x0 < res - 1) & (y0 >= 0) & (y0 < res - 1)
    if not valid.any():
        return Image.new("RGBA", (res, res), (0, 0, 0, 0)), gain
    x0 = x0[valid]
    y0 = y0[valid]
    wx = wx[valid]
    wy = wy[valid]
    rgb = rgb[valid]
    npx = res * res
    counts = np.zeros(npx, np.float32)
    colors = np.zeros((npx, 3), np.float32)
    for dx, dy, w in (
        (0, 0, (1 - wx) * (1 - wy)),
        (1, 0, wx * (1 - wy)),
        (0, 1, (1 - wx) * wy),
        (1, 1, wx * wy),
    ):
        idx = (y0 + dy) * res + (x0 + dx)
        counts += np.bincount(idx, weights=w, minlength=npx).astype(np.float32)
        for c in range(3):
            colors[:, c] += np.bincount(idx, weights=rgb[:, c] * w, minlength=npx)

    counts = counts.reshape(res, res)
    colors = colors.reshape(res, res, 3)

    # 高斯平滑（sigma 4px），把离散色阶融成连续云，同时保留形状
    radius = 24
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs * xs) / (2 * 7.5 * 7.5))
    kernel /= kernel.sum()
    counts = _conv2(counts, kernel)
    colors = _conv2(colors, kernel)

    avg = (colors / np.maximum(counts[..., None], 1e-6)).reshape(-1, 3)
    counts = counts.reshape(-1)

    # 提升饱和度和亮度，让颜色更明显
    hh, ss, vv = rgb_to_hsv(np.clip(avg, 0, 1))
    ss = np.clip(ss * (1.8 * sat_scale) + 0.10, 0, 1)
    vv = np.clip(vv * 1.3 + 0.08, 0, 1)
    avg = np.clip(hsv_to_rgb_arr(hh, ss, vv), 0, 1)

    nz = counts[counts > 1e-3]
    scale = float(np.percentile(nz, 99.8)) if len(nz) else 1.0
    raw = np.clip(counts / max(scale, 1e-4), 0, 1)
    # 密度曲线：gamma 越大，稀疏区域越淡
    dens = np.power(raw, density_gamma)
    alpha = np.clip(dens * 255 * 1.1 * alpha_scale, 0, 255)

    yy, xx = np.mgrid[0:res, 0:res]
    rad = np.hypot(xx - res / 2, yy - res / 2) / (res / 2)
    center = np.clip(1.0 - rad / 0.5, 0, 1).reshape(-1)
    white = (np.power(raw, 3.0) * 0.25 * center * white_scale)[:, None]
    avg = np.clip(avg * (1 - white) + white, 0, 1)
    avg = np.clip(avg * 1.08 + 0.02, 0, 1)
    # 稀疏区域同时降低亮度，密集区域保持原色
    bright = 0.50 + 0.50 * dens
    avg = np.clip(avg * bright[:, None], 0, 1)

    arr = np.zeros((npx, 4), np.uint8)
    arr[:, :3] = np.clip(avg * 255, 0, 255).astype(np.uint8)
    arr[:, 3] = alpha.astype(np.uint8)
    img = Image.fromarray(arr.reshape(res, res, 4), "RGBA")
    size = int(round(R * 2.3))
    img = img.resize((size, size), Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(1.8 * SS))
    # 圆环内侧柔和淡出，避免色云被硬裁成一条边
    cw, ch = img.size
    yy, xx = np.mgrid[0:ch, 0:cw]
    dist = np.hypot(xx - cw / 2, yy - ch / 2) / (cw / 2)
    fade = np.clip((1.0 - dist) / 0.08, 0, 1).astype(np.float32)
    alpha = np.asarray(img.getchannel("A"), dtype=np.float32) * fade
    img.putalpha(Image.fromarray(np.clip(alpha, 0, 255).astype(np.uint8)))
    return img, gain


def make_vectorscope(an: dict, size: int = 740, gain: float = 0.0,
                     unit: str = "second", alpha_scale: float = 1.0,
                     sat_scale: float = 1.0, white_scale: float = 1.0,
                     density_gamma: float = 0.55) -> Image.Image:
    img, d, top = panel((size, size), "VECTORSCOPE  ·  UV chroma")
    W, H = img.size
    cx, cy = W / 2, top + (H - top) / 2
    R = min(W, H - top) * 0.46 * 0.92

    # 圆形黑底
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(0, 0, 2, 255),
              outline=(66, 66, 78, 255), width=SS)

    # 色度云：加色混合
    cloud, used_gain = _cloud_layer(an["cloud_u"], an["cloud_v"], an["cloud_rgb"],
                                    cx, cy, R, gain, alpha_scale, sat_scale, white_scale,
                                    density_gamma)
    if cloud.width > 4:
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        canvas.alpha_composite(cloud, (int(cx - cloud.width / 2), int(cy - cloud.height / 2)))
        base = np.asarray(img.convert("RGB"), dtype=np.float32)
        cl = np.asarray(canvas, dtype=np.float32)
        a = (cl[..., 3:4] / 255.0)
        raw_add = cl[..., :3] * a * 4.2
        # 高光滚降：亮部不再硬切到白，保留颜色层次
        add = 255.0 * (1.0 - np.exp(-np.clip(raw_add, 0, None) / 130.0))
        blended = 255.0 - (255.0 - base) * (255.0 - np.clip(add, 0, 255)) / 255.0
        img = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        d = ImageDraw.Draw(img)

    # 圆周刻度
    for deg in range(0, 360, 5):
        a = math.radians(deg)
        long_tick = (deg % 15 == 0)
        r0 = R * (0.925 if long_tick else 0.955)
        r1 = R * 0.995
        x0, y0 = cx + math.cos(a) * r0, cy - math.sin(a) * r0
        x1, y1 = cx + math.cos(a) * r1, cy - math.sin(a) * r1
        d.line([(x0, y0), (x1, y1)], fill=(82, 82, 94, 255), width=SS)

    # 内圈
    d.ellipse([cx - R * 0.995, cy - R * 0.995, cx + R * 0.995, cy + R * 0.995],
              outline=(84, 84, 96, 255), width=SS)

    # 十字准线
    d.line([(cx - R, cy), (cx + R, cy)], fill=(70, 70, 82, 255), width=SS)
    d.line([(cx, cy - R), (cx, cy + R)], fill=(70, 70, 82, 255), width=SS)
    d.ellipse([cx - 3 * SS, cy - 3 * SS, cx + 3 * SS, cy + 3 * SS],
              fill=(220, 220, 228, 255))

    # 六个目标点
    for name, deg, color in VECTOR_TARGETS:
        a = math.radians(deg)
        tx = cx + math.cos(a) * R * 0.88
        ty = cy - math.sin(a) * R * 0.88
        b = 6 * SS
        for sx in (-1, 1):
            for sy in (-1, 1):
                x = tx + sx * b
                y = ty + sy * b
                d.line([(x, y), (x - sx * b * 0.55, y)], fill=color + (255,), width=SS)
                d.line([(x, y), (x, y - sy * b * 0.55)], fill=color + (255,), width=SS)
        f = font(11 * SS, True)
        tw = d.textlength(name, font=f)
        d.text((tx - tw / 2, ty + b + 2 * SS), name, font=f, fill=(210, 210, 220, 255))

    # 底部说明
    neutral = an["neutral_share"] * 100
    d.text((24 * SS, H - 34 * SS),
           f"neutral {neutral:.0f}%  ·  {an['seconds']} {unit}s  ·  chroma gain ×{used_gain:.1f}",
           font=font(11 * SS), fill=DIM)
    return img.resize((size, size), Image.LANCZOS)


def make_hist(an: dict, w: int = 900, h: int = 740) -> Image.Image:
    img, d, top = panel((w, h), "RGB HISTOGRAM  ·  time-weighted")
    W, H = img.size
    left, right = 56 * SS, 34 * SS
    bottom = H - 58 * SS
    plot_w = W - left - right
    plot_top = top + 10 * SS
    plot_h = bottom - plot_top
    base_y = bottom

    lx = W - 330 * SS
    for name, color in (("R", CH_RGB[0]), ("G", CH_RGB[1]), ("B", CH_RGB[2]),
                        ("Luma", (235, 235, 242))):
        d.rectangle([lx, 24 * SS, lx + 11 * SS, 35 * SS], fill=color + (255,))
        d.text((lx + 16 * SS, 21 * SS), name, font=font(10 * SS), fill=MUTED)
        lx += 74 * SS

    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel /= kernel.sum()
    smooth = {k: np.convolve(v, kernel, mode="same") for k, v in an["hists"].items()}
    peak = max(float(v.max()) for v in smooth.values()) or 1.0
    for k in smooth:
        smooth[k] = smooth[k] / peak

    for frac in (0.25, 0.5, 0.75, 1.0):
        y = base_y - plot_h * frac
        d.line([(left, y), (left + plot_w, y)], fill=GRID, width=SS)
    for value in (0, 64, 128, 192, 255):
        x = left + plot_w * value / 255.0
        d.line([(x, plot_top), (x, base_y)], fill=(34, 34, 42, 255), width=SS)
        label = str(value)
        f = font(10 * SS)
        d.text((x - d.textlength(label, font=f) / 2, base_y + 8 * SS), label, font=f, fill=DIM)
    d.line([(left, base_y), (left + plot_w, base_y)], fill=(80, 80, 94, 255), width=SS)

    def curve(hist, color, filled=True, width=3):
        pts = []
        n = len(hist)
        for i, value in enumerate(hist):
            if value <= 0.001:
                continue
            x = left + plot_w * (i + 0.5) / n
            y = base_y - plot_h * min(1.0, float(value))
            pts.append((x, y))
        if len(pts) < 2:
            return
        if filled:
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            ld.polygon(pts + [(pts[-1][0], base_y), (pts[0][0], base_y)],
                       fill=color + (24,))
            img.alpha_composite(layer)
        d.line(pts, fill=color + (235,), width=width)

    for key, color in zip(("r", "g", "b"), CH_RGB):
        curve(smooth[key], color, width=3 * SS)
    curve(smooth["luma"], (235, 235, 242), filled=False, width=2 * SS)

    mx = left + plot_w * min(1.0, an["mean_luma"])
    d.line([(mx, plot_top), (mx, base_y)], fill=(235, 235, 242, 90), width=SS)
    label = f"mean luma {an['mean_luma'] * 100:.0f}%"
    d.text((mx + 6 * SS, base_y - 18 * SS), label, font=font(10 * SS), fill=MUTED)
    return img.resize((w, h), Image.LANCZOS)


def make_timeline(an: dict, w: int = 1680, h: int = 96, unit: str = "second") -> Image.Image:
    img = Image.new("RGBA", (w, h), PANEL)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, h - 1], outline=(38, 38, 47, 255), width=1)
    d.text((20, 10), f"TONE TIMELINE  ·  1 px = 1 {unit}", font=font(12, True), fill=TEXT)
    strip_h = h - 44
    per = np.clip(an["per"], 0, 255).astype(np.uint8)
    strip = Image.fromarray(per.reshape(1, len(per), 3), "RGB").resize((w - 40, strip_h), Image.LANCZOS)
    img.paste(strip, (20, 32))
    return img


def compose(an: dict, scope: Image.Image, hist: Image.Image,
            title: str, subtitle: str, out_path: str, unit: str = "second") -> None:
    W, H = 1800, 1210
    canvas = Image.new("RGB", (W, H), BG[:3])
    d = ImageDraw.Draw(canvas)
    d.text((60, 44), title, font=font(34, True), fill=TEXT)
    d.text((60, 92), subtitle, font=font(14), fill=MUTED)

    m = an["mean_rgb"]
    stats = [
        ("MEAN", hex_of(m)),
        ("LUMA", f"{an['mean_luma'] * 100:.0f}%"),
        ("SAT", f"{an['mean_sat'] * 100:.0f}%"),
        ("NEUTRAL", f"{an['neutral_share'] * 100:.0f}%"),
        ("CHROMA", HUE_NAMES[an["dominant"]] if an["dominant"] >= 0 else "—"),
    ]
    x = 980
    for label, value in stats:
        d.text((x, 52), label, font=font(11, True), fill=DIM)
        d.text((x, 68), value, font=font(17, True), fill=TEXT)
        x += 160

    canvas.paste(scope, (60, 150))
    canvas.paste(hist, (840, 150))
    timeline = make_timeline(an, 1680, 96, unit)
    canvas.paste(timeline, (60, 928))
    d.text((60, 1044), f"Source: {os.path.basename(an['path'])}  ·  "
                       f"{an['seconds']} {unit}s  ·  {an['width']}×{an['height']} px",
           font=font(11), fill=DIM)
    d.text((60, 1066), "Generated by color_report.py", font=font(11), fill=DIM)
    canvas.save(out_path)
    print(f"report: {out_path}  {canvas.size[0]}×{canvas.size[1]}")


def main() -> int:
    p = argparse.ArgumentParser(description="从每秒图生成矢量示波器与 RGB/亮度直方图")
    p.add_argument("image", help="每秒图（--per-second 的输出）")
    p.add_argument("-o", "--output", default=None, help="报告输出路径（PNG）")
    p.add_argument("--px-per-second", type=int, default=1, help="每秒图里每秒占多少像素")
    p.add_argument("--crop-bottom", type=int, default=0, help="裁掉底部时间轴像素（有刻度时一般 44）")
    p.add_argument("--title", default=None, help="报告标题")
    p.add_argument("--unit", choices=["second", "shot", "column"], default="second",
                   help="每个颜色样本的单位：second / shot / column（任意条带）")
    p.add_argument("--scope-size", type=int, default=740)
    p.add_argument("--chroma-gain", type=float, default=0.0,
                   help="矢量示波器色度放大倍数；0 = 按 99 百分位自动")
    p.add_argument("--cloud-alpha", type=float, default=1.0,
                   help="色云强度；越小越不炸，建议 0.5~1.0")
    p.add_argument("--cloud-saturation", type=float, default=1.0,
                   help="色云饱和度增益；越小越柔和，建议 0.6~1.0")
    p.add_argument("--white-core", type=float, default=1.0,
                   help="中心白核强度；0 关闭，建议 0.2~0.5")
    p.add_argument("--density-gamma", type=float, default=0.55,
                   help="密度曲线；越大稀疏区域越淡，建议 0.5~1.2")
    args = p.parse_args()

    if not os.path.isfile(args.image):
        print(f"找不到图片：{args.image}")
        return 2
    an = analyze(args.image, args.px_per_second, args.crop_bottom)
    if an["seconds"] == 0:
        print("图片里没有可分析的列")
        return 1

    scope = make_vectorscope(an, args.scope_size, args.chroma_gain, args.unit,
                             args.cloud_alpha, args.cloud_saturation, args.white_core,
                             args.density_gamma)
    hist = make_hist(an, 900, args.scope_size)
    stem = os.path.splitext(args.image)[0]
    out = args.output or (stem + "_color_report.png")
    title = args.title or os.path.basename(args.image)
    subtitle = (f"{an['seconds']} {args.unit}s  ·  {an['width']}×{an['height']} px  ·  "
                f"mean {hex_of(an['mean_rgb'])}")
    compose(an, scope, hist, title, subtitle, out, args.unit)

    root, _ = os.path.splitext(out)
    scope.save(root + "_vectorscope.png")
    hist.save(root + "_hist.png")
    print(f"vectorscope: {root}_vectorscope.png")
    print(f"hist:        {root}_hist.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
