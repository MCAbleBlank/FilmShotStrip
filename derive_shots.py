#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
derive_shots.py — 从「每秒图」的缓存聚合出「一分镜一列」图和分镜清单

适用场景：已经跑过 film_shot_strip.py --per-second 1（生成 out/.cache/<片源>.persecond.*），
但不想再花一遍全片解码去做镜头检测。这里在每秒颜色序列上做切点检测，
再把每个镜头内的秒颜色平均成 1 列。

用法：
  python derive_shots.py --cache out/.cache/<片源>.persecond --output out/xx_1px.png \
      --height 1000 --fps 23.976
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from film_shot_strip import Shot, save_image, with_mapped_ruler  # noqa: E402


def load_persecond(base: str):
    with open(base + ".shots.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    z = np.load(base + ".traces.npz", allow_pickle=False)
    n = len(meta["shots"])
    traces = []
    for i in range(n):
        key = f"trace_{i}"
        if key in z.files:
            traces.append(z[key].astype(np.float32))
        else:
            traces.append(np.zeros((1, 2, 3), np.float32))
    return meta, traces


def detect_cuts(colors: np.ndarray, fps: float, min_shot: float = 2.0) -> np.ndarray:
    """在每秒颜色序列上找切点（索引 = 新镜头开始的秒）。"""
    if len(colors) < 4:
        return np.zeros(0, dtype=np.int64)
    diff = np.abs(np.diff(colors, axis=0)).sum(axis=1)
    win = 9
    pad = np.pad(diff, win // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, win)
    local = np.median(windows, axis=1)
    thr = np.maximum(20.0, local * 2.6)
    cand = np.flatnonzero(diff >= thr) + 1
    if len(cand) == 0:
        return cand

    # 合并 min_shot 内的多个候选，保留差异最大的
    merged = []
    for idx in cand:
        if merged and (idx - merged[-1]) < min_shot:
            if diff[idx - 1] > diff[merged[-1] - 1]:
                merged[-1] = idx
        else:
            merged.append(idx)
    # 太短的镜头并入相邻
    cuts = [0] + merged
    if len(cuts) > 1 and cuts[-1] - cuts[-2] < min_shot:
        cuts.pop()
    return np.asarray(cuts, dtype=np.int64)


def normalize_rows(rows: np.ndarray, target: int) -> np.ndarray:
    if rows.shape[0] == target:
        return rows
    if rows.shape[0] <= 2:  # 缺失秒的占位色带
        return np.tile(rows.mean(axis=0, keepdims=True), (target, 1))
    idx = np.linspace(0, rows.shape[0] - 1, target)
    out = np.empty((target, 3), np.float32)
    for c in range(3):
        out[:, c] = np.interp(idx, np.arange(rows.shape[0]), rows[:, c])
    return out


def build_columns(rows_per_sec: np.ndarray, cuts, seconds: int, height: int):
    cols = []
    spans = []
    bounds = list(cuts) + [seconds]
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        row = rows_per_sec[a:min(b, seconds)].mean(axis=0)  # (rows,3)
        col = np.clip(row, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        cols.append(Image.fromarray(col, "RGB").resize((1, height), Image.BICUBIC))
        spans.append((a, b))
    if not cols:
        return None, []
    img = Image.new("RGB", (len(cols), height))
    for i, col in enumerate(cols):
        img.paste(col, (i, 0))
    return img, spans


def main() -> int:
    p = argparse.ArgumentParser(description="从每秒缓存聚合一分镜一列图")
    p.add_argument("--cache", required=True, help="每秒缓存前缀（不带 .shots.json）")
    p.add_argument("--output", required=True, help="输出 PNG")
    p.add_argument("--height", type=int, default=1000)
    p.add_argument("--fps", type=float, default=23.976)
    p.add_argument("--min-shot", type=float, default=2.0)
    p.add_argument("--ruler", action="store_true", help="底部画映射到分镜的时间刻度")
    args = p.parse_args()

    meta, traces = load_persecond(args.cache)
    raw = [t.mean(axis=0).astype(np.float32) for t in traces]
    target_rows = max(r.shape[0] for r in raw)
    rows_per_sec = np.stack([normalize_rows(r, target_rows) for r in raw])
    colors = rows_per_sec.mean(axis=1)
    seconds = len(colors)
    cuts = detect_cuts(colors, args.fps, args.min_shot)
    img, spans = build_columns(rows_per_sec, cuts, seconds, args.height)
    if img is None:
        print("没有可用的每秒数据")
        return 1

    shots = [Shot(float(a), float(b)) for a, b in spans]
    for s in shots:
        s.width = 1
    if args.ruler:
        img = with_mapped_ruler(img, shots)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_image(img, args.output, 92)
    print(f"image: {args.output}  {img.size[0]}×{img.size[1]}  shots={len(shots)}")

    stem = os.path.splitext(args.output)[0]
    payload = {
        "source": meta.get("source"),
        "source_hash": meta.get("source_hash"),
        "time_unit": "frames",
        "fps": args.fps,
        "frame_count": meta.get("frame_count"),
        "mode": "derived_from_persecond",
        "min_shot": args.min_shot,
        "shots": [
            {
                "index": i,
                "start": int(round(s.start * args.fps)),
                "end": int(round(s.end * args.fps)),
                "duration": int(round((s.end - s.start) * args.fps)),
                "width_px": 1,
            }
            for i, s in enumerate(shots)
        ],
    }
    with open(stem + "_shots.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"shots: {stem}_shots.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
