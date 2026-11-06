#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
film_shot_strip.py  —  把电影切成分镜，再按「色彩 + 时长」拼成一张长图

思路（参考 yyyeying/FilmToLongPicture，但按“分镜”重构）：
  1. 先用 PySceneDetect 把片子切成一个个分镜（镜头），没装时回退到 ffmpeg；
  2. 每个分镜低分辨率采样若干帧，把每一行像素做水平平均，得到
     「时间 × 画面纵向位置」的颜色矩阵（和参考项目一样是竖向渐变色带）；
  3. 每个分镜在长图里占的像素宽度 = 分镜时长 × 每秒像素数，
     所以长图横向就是时间轴，宽的地方就是长镜头；
  4. 分镜之间自然形成接缝；最后按最大宽度切段导出，并附一张缩略预览。

分镜结果按片源 SHA-256 缓存在 out/ 下，第二次生成时跳过检测和采样。
分镜 JSON 里的 start / end / duration 都是帧数。只依赖 ffmpeg + numpy + Pillow。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

__version__ = "1.9.1"

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

JPEG_MAX_DIM = 65000
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
]


# ---------------------------------------------------------------- 基础工具

def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


PROGRESS_ENABLED = True


class ProgressBar:
    """终端进度条：TTY 原地刷新，非 TTY 每 10% 打一行，都带预估剩余时间。"""

    def __init__(self, total: float, label: str, quiet: bool = False, width: int = 22):
        self.total = max(1.0, float(total or 0))
        self.label = label
        self.width = width
        self.enabled = bool(PROGRESS_ENABLED and not quiet and total and total > 0)
        self.tty = sys.stderr.isatty()
        self.current = 0.0
        self.start = time.monotonic()
        self.last_draw = 0.0
        self.last_milestone = -1
        self.printed_full = False

    @staticmethod
    def _fmt(sec: float) -> str:
        sec = max(0, int(round(sec)))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _line(self) -> str:
        pct = min(100.0, self.current / self.total * 100.0)
        filled = int(round(self.width * pct / 100.0))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.monotonic() - self.start
        text = f"  {self.label} [{bar}] {pct:4.0f}%"
        if self.current < self.total:
            if self.current < self.total * 0.01:
                text += "  剩 --:--"
            else:
                eta = elapsed * (self.total - self.current) / self.current
                text += f"  剩 {self._fmt(eta)}"
        else:
            text += f"  用时 {self._fmt(elapsed)}"
        return text

    def update(self, n: float | None = None, force: bool = False) -> None:
        if not self.enabled:
            return
        if n is not None:
            self.current = min(self.total, float(n))
        now = time.monotonic()
        pct = self.current / self.total * 100.0
        if not force:
            if self.tty:
                if now - self.last_draw < 0.15 and pct < 100.0:
                    return
            else:
                milestone = int(pct // 10)
                if milestone <= self.last_milestone:
                    return
                self.last_milestone = milestone
        self.last_draw = now
        line = self._line()
        if self.tty:
            cols = max(40, shutil.get_terminal_size((80, 20)).columns - 1)
            print("\r" + line[:cols].ljust(cols), end="", file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)
        self.printed_full = pct >= 99.999

    def finish(self) -> None:
        if not self.enabled:
            return
        if not self.printed_full:
            self.update(self.total, force=True)
        if self.tty and not self.printed_full:
            print(file=sys.stderr, flush=True)
        elif self.tty:
            print(file=sys.stderr, flush=True)
        self.enabled = False


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def probe_media(path: str) -> dict:
    """读取时长、分辨率、旋转信息（旋转过的手机视频要交换宽高）。"""
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration:stream_tags=rotate:stream_side_data=rotation:format=duration",
        "-of", "json", path,
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffprobe 读取失败：" + proc.stderr.decode("utf-8", "replace").strip()
        )
    data = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"文件里没有视频流：{path}")
    st = streams[0]
    width = int(st.get("width") or 0)
    height = int(st.get("height") or 0)

    rotation = 0
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if not rotation:
        try:
            rotation = int(round(float((st.get("tags") or {}).get("rotate", 0))))
        except (TypeError, ValueError):
            rotation = 0
    if abs(rotation) % 180 == 90:
        width, height = height, width

    duration = 0.0
    try:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        try:
            duration = float(st.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
    if duration <= 0:
        duration = _duration_from_stderr(path)
    if duration <= 0:
        raise RuntimeError("无法获取视频时长，请确认文件可正常播放。")

    fps = parse_rate(st.get("r_frame_rate")) or parse_rate(st.get("avg_frame_rate")) or 25.0
    return {
        "width": max(width, 2),
        "height": max(height, 2),
        "duration": duration,
        "fps": fps,
        "fps_raw": st.get("r_frame_rate", "0/0"),
    }


def _duration_from_stderr(path: str) -> float:
    proc = run_cmd([FFMPEG, "-hide_banner", "-i", path])
    text = proc.stderr.decode("utf-8", "replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not m:
        return 0.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def parse_rate(rate) -> float:
    """把 ffprobe 的 \"24000/1001\" 之类帧率转成浮点数。"""
    if not rate:
        return 0.0
    try:
        text = str(rate)
        if "/" in text:
            num, den = text.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def sanitize_name(name: str) -> str:
    """让片源名能安全地当文件名用。"""
    name = re.sub(r"[^\w\-.]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._")
    return (name[:80] or "video")


def file_sha256(path: str, quiet: bool = False) -> str:
    """对片源整体取 SHA-256（大文件时每 512MB 打一行进度）。"""
    h = hashlib.sha256()
    size = os.path.getsize(path)
    done = 0
    bar = ProgressBar(size if size >= (32 << 20) else 0, "计算哈希", quiet=quiet)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            bar.update(done)
    bar.finish()
    return "sha256:" + h.hexdigest()


def shot_frame_ranges(shots: list["Shot"], fps: float, total_frames: int) -> list[tuple[int, int]]:
    """把分镜的秒数换算成首尾相接的帧区间。"""
    n = len(shots)
    if n == 0:
        return []
    ends = [int(round(s.end * fps)) for s in shots]
    ends[-1] = total_frames
    ranges: list[tuple[int, int]] = []
    prev = 0
    for i, raw_end in enumerate(ends):
        start = prev
        end = max(raw_end, start + 1)
        if i == n - 1:
            end = max(total_frames, start + 1)
        else:
            end = min(end, total_frames - (n - 1 - i))
        end = max(end, start + 1)
        ranges.append((start, end))
        prev = end
    return ranges


CACHE_VERSION = 1


def load_shot_cache(path: str, source_hash: str, det_params: dict):
    """读取分镜缓存；返回 (数据, 未命中原因)。"""
    if not os.path.exists(path):
        return None, "没有缓存文件"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"缓存读取失败（{exc}）"
    if data.get("cache_version") != CACHE_VERSION:
        return None, "缓存版本不匹配"
    if data.get("source_hash") != source_hash:
        return None, "片源哈希已变化"
    if data.get("detection") != det_params:
        return None, "检测参数已变化"
    if not data.get("shots"):
        return None, "缓存里没有分镜"
    return data, ""


def write_shot_cache(path: str, traces_path: str | None, video: str, source_hash: str,
                     fps: float, frame_count: int, duration: float,
                     det_params: dict, ranges: list[tuple[int, int]],
                     shots: list["Shot"]) -> None:
    payload = {
        "cache_version": CACHE_VERSION,
        "source": os.path.abspath(video),
        "source_name": os.path.basename(video),
        "source_size": os.path.getsize(video),
        "source_hash": source_hash,
        "time_unit": "frames",
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "duration_seconds": round(duration, 3),
        "detection": det_params,
        "traces_file": os.path.basename(traces_path) if traces_path else None,
        "shots": [
            {
                "index": i,
                "start": st,
                "end": en,
                "duration": en - st,
                "color": "#%02x%02x%02x" % tuple(int(c) for c in s.color),
            }
            for i, (s, (st, en)) in enumerate(zip(shots, ranges))
        ],
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def save_cached_traces(path: str | None, shots: list["Shot"]) -> None:
    """把每个分镜的色带（和 frame 风格的缩略图）存成 npz，重绘时免采样。"""
    if not path:
        return
    arrays = {"colors": np.array([s.color for s in shots], dtype=np.uint8)}
    for i, s in enumerate(shots):
        if s.trace is not None:
            arrays[f"trace_{i}"] = s.trace
        if s.thumb:
            arrays[f"thumb_{i}"] = np.frombuffer(s.thumb, dtype=np.uint8)
        if s.edges is not None:
            arrays[f"edge_{i}"] = s.edges
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def load_cached_traces(path: str | None, shots: list["Shot"]) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            colors = z["colors"]
            if colors.shape[0] != len(shots):
                return False
            for i, s in enumerate(shots):
                s.trace = np.array(z[f"trace_{i}"], copy=True)
                s.color = tuple(int(v) for v in colors[i])
                key = f"thumb_{i}"
                if key in z.files:
                    s.thumb = z[key].tobytes()
                ekey = f"edge_{i}"
                if ekey in z.files:
                    s.edges = np.array(z[ekey], copy=True)
    except (OSError, KeyError, ValueError):
        return False
    return True


# ---------------------------------------------------------------- 分镜检测

def detect_cuts_scdet(path: str, threshold: float, quiet: bool = False) -> list[float]:
    """用 ffmpeg 内置 scdet 检测硬切点，返回秒数。

    scdet 的分数比 select=scene 更稳定，且 sc_pass=1 只输出命中的帧。
    """
    vf = (
        f"scale=160:-2,scdet=threshold={threshold:.3f}:sc_pass=1,"
        f"metadata=print:file=-"
    )
    cmd = [
        FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path, "-map", "0:v:0", "-an", "-sn",
        "-vf", vf, "-f", "null", "-",
    ]
    proc = run_cmd(cmd)
    text = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        if not quiet:
            log("  scdet 不可用，回退到 scene 过滤" + (f"（{err[-1]}）" if err else ""))
        return _detect_cuts_scene(path, min(0.9, max(0.05, threshold / 40.0)), quiet)
    times = [float(x) for x in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", text)]
    log(f"  scdet 检测到 {len(times)} 个切点", quiet)
    return times


def _detect_cuts_scene(path: str, threshold: float, quiet: bool = False) -> list[float]:
    """旧版回退方案：按 scene 分数（0~1）找硬切点。"""
    vf = (
        f"scale=160:-2,select='gt(scene,{threshold:.4f})',"
        f"metadata=print:file=-"
    )
    cmd = [
        FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path, "-map", "0:v:0", "-an", "-sn",
        "-vf", vf, "-f", "null", "-",
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg 分镜检测失败：" + proc.stderr.decode("utf-8", "replace").strip()
        )
    text = proc.stdout.decode("utf-8", "replace")
    times = [float(x) for x in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", text)]
    log(f"  scene 过滤检测到 {len(times)} 个切点", quiet)
    return times


def detect_cuts_scenedetect(path: str, threshold: float, quiet: bool = False,
                           detector: str = "content",
                           total_frames: int = 0) -> list[float]:
    """用 PySceneDetect 检测硬切点（推荐方式）。

    threshold 是统一的 0~1 灵敏度，这里换算成 PySceneDetect 的阈值：
    ContentDetector 默认约 27，AdaptiveDetector 默认约 3。
    """
    try:
        from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video
    except ImportError as exc:  # pragma: no cover - 可选依赖
        raise RuntimeError(
            "未安装 PySceneDetect，请执行：pip install scenedetect[opencv]"
        ) from exc

    if not _opencv_can_decode(path):
        raise RuntimeError("OpenCV 无法解码该视频（AV1 等编码需要走 ffmpeg）")

    if detector == "adaptive":
        det = AdaptiveDetector(adaptive_threshold=max(0.5, threshold * 9.0))
    else:
        det = ContentDetector(threshold=min(100.0, max(5.0, threshold * 80.0)))

    video = open_video(path)
    bar = ProgressBar(total_frames, "检测分镜", quiet=quiet)
    stop = threading.Event()

    def _monitor() -> None:
        # PySceneDetect 的 callback 只在切点触发，这里轮询解码位置拿连续进度
        while not stop.wait(0.2):
            try:
                pos = video.position
            except Exception:
                return
            if pos is not None:
                bar.update(int(pos) + 1)

    watcher = threading.Thread(target=_monitor, daemon=True)
    watcher.start()
    try:
        manager = SceneManager()
        manager.add_detector(det)
        manager.detect_scenes(video, show_progress=False)
        scenes = manager.get_scene_list()
    except Exception as exc:  # 编码异常、OpenCV 后端问题等
        raise RuntimeError(f"PySceneDetect 检测失败：{exc}") from exc
    finally:
        stop.set()
        watcher.join(timeout=1.0)
        bar.finish()

    cuts = [_tc_seconds(start) for start, _end in scenes]
    log(f"  PySceneDetect（{detector}）检测到 {max(0, len(cuts) - 1)} 个切点", quiet)
    return cuts


def _opencv_can_decode(path: str) -> bool:
    """PySceneDetect 的 OpenCV 后端对 AV1 等编码会静默返回 0 帧，先试读一帧。"""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        return bool(ok and frame is not None)
    except Exception:
        return False


def _tc_seconds(tc) -> float:
    """兼容 PySceneDetect 0.6（get_seconds）与 0.7（seconds）。"""
    try:
        return float(tc.seconds)
    except AttributeError:
        return float(tc.get_seconds())


def normalize_shots(
    cuts: list[float],
    duration: float,
    min_shot: float,
    max_shot: float,
) -> list["Shot"]:
    """把切点整理成首尾相接、且不短于 min_shot 的分镜序列。"""
    cuts = sorted({round(float(c), 3) for c in cuts if 0.0 < c < duration})
    bounds = [0.0]
    for c in cuts:
        if c - bounds[-1] >= 0.05:
            bounds.append(c)
    if duration - bounds[-1] >= 0.05:
        bounds.append(duration)
    else:
        bounds[-1] = duration

    shots = [Shot(a, b) for a, b in zip(bounds, bounds[1:]) if b - a > 1e-6]
    if not shots:
        return [Shot(0.0, duration)]

    # 太短的镜头（闪白、叠化、误检）并到相邻镜头里
    if min_shot > 0:
        changed = True
        while changed and len(shots) > 1:
            changed = False
            for i, s in enumerate(shots):
                if s.dur >= min_shot:
                    continue
                if i == 0:
                    shots[1].start = s.start
                    shots.pop(0)
                else:
                    shots[i - 1].end = s.end
                    shots.pop(i)
                changed = True
                break

    # 可选：把超长镜头平均切开，避免一张图被一个长镜头吃掉全部宽度
    if max_shot and max_shot > 0:
        out: list[Shot] = []
        for s in shots:
            n = max(1, math.ceil(s.dur / max_shot))
            if n == 1:
                out.append(s)
                continue
            step = s.dur / n
            for k in range(n):
                out.append(Shot(s.start + k * step, s.start + (k + 1) * step))
        shots = out
    return shots


# ---------------------------------------------------------------- 采样与色彩

@dataclass
class Shot:
    start: float
    end: float
    trace: np.ndarray | None = None      # (bins, rows, 3) uint8，时间 × 高度颜色
    color: tuple = (0, 0, 0)             # 代表色（中位色）
    thumb: bytes | None = None           # --style frame 用的中帧 JPEG
    edges: np.ndarray | None = None      # (3, rows, 3) 首帧/中间帧/末帧的行均值
    empty: bool = False                  # 该分镜没有分到任何采样帧
    width: int = 1

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


def sample_dimensions(media: dict, sample_w: int) -> tuple[int, int]:
    aspect = media["width"] / media["height"]
    w = max(2, sample_w)
    h = max(2, int(round(w / aspect)))
    if h % 2:
        h += 1
    return w, h


def iter_frames(path: str, fps: float, width: int, height: int, quiet: bool = False):
    """流式解码低分辨率帧，产出 (时间戳秒, HxWx3 uint8)。"""
    vf = f"fps={fps:.6f},scale={width}:{height}:flags=area"
    cmd = [
        FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path, "-map", "0:v:0", "-an", "-sn",
        "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=width * height * 3 * 4
    )
    frame_bytes = width * height * 3
    index = 0
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            yield index / fps, arr
            index += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        err = b""
        if proc.stderr:
            err = proc.stderr.read()
            proc.stderr.close()
        proc.wait()
        if proc.returncode not in (0, None) and not quiet:
            msg = err.decode("utf-8", "replace").strip()
            if msg:
                log("  ffmpeg 采样警告：" + msg.splitlines()[-1], quiet)


def pool_rows(arr: np.ndarray, max_rows: int) -> np.ndarray:
    rows = arr.shape[0]
    if rows <= max_rows:
        return arr
    k = math.ceil(rows / max_rows)
    pad = (-rows) % k
    if pad:
        arr = np.pad(arr, ((0, pad), (0, 0), (0, 0)), mode="edge")
    return arr.reshape(-1, k, arr.shape[1], 3).mean(axis=1)


def fill_empty_bins(mean: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """某些 bin 没采到帧时，用最近的有效 bin 填上。"""
    valid = np.flatnonzero(counts > 0)
    if len(valid) == 0:
        mean[:] = 0
        return mean
    if len(valid) == len(counts):
        return mean
    idx = np.arange(len(counts))
    nearest = valid[np.argmin(np.abs(idx[:, None] - valid[None, :]), axis=1)]
    return mean[nearest]


def fill_empty_shots(shots: list["Shot"]) -> int:
    """给没分到帧的分镜补一份最近的可用色带。"""
    valid = [i for i, s in enumerate(shots) if not s.empty and s.trace is not None]
    if not valid:
        return 0
    fixed = 0
    for i, s in enumerate(shots):
        if not s.empty:
            continue
        j = min(valid, key=lambda k: abs(shots[k].start - s.start))
        src = shots[j]
        s.trace = src.trace.copy()
        s.color = src.color
        s.thumb = src.thumb
        s.edges = None if src.edges is None else src.edges.copy()
        s.empty = False
        fixed += 1
    return fixed


def repair_isolated_black(shots: list["Shot"]) -> int:
    """修补旧缓存里“前后都不是黑帧、自己却全黑”的缺失秒采样。"""
    n = len(shots)
    fixed = 0
    for i, s in enumerate(shots):
        if s.trace is None or int(s.trace.max()) > 0:
            continue
        if 0 < i < n - 1:
            prev, nxt = shots[i - 1].trace, shots[i + 1].trace
            if (prev is not None and nxt is not None
                    and int(prev.max()) > 0 and int(nxt.max()) > 0):
                s.trace = ((prev.astype(np.float32) + nxt.astype(np.float32)) / 2.0)
                s.trace = np.clip(s.trace, 0, 255).astype(np.uint8)
                s.color = tuple(int(v) for v in np.median(s.trace.reshape(-1, 3), axis=0))
                fixed += 1
    return fixed


def collect_shot_data(
    shots: list[Shot],
    path: str,
    fps: float,
    width: int,
    height: int,
    style: str,
    max_trace_rows: int = 108,
    quiet: bool = False,
) -> None:
    """流式采样，把每个分镜压成 (时间 bin, 行, 3) 的颜色轨迹。"""
    total = shots[-1].end
    states: list[dict] = [
        {
            "bins": max(1, min(64, int(math.ceil(s.dur * fps - 1e-6)))),
            "sums": None,
            "counts": None,
            "best_t": 1e18,
            "best_frame": None,
            "edge_t": [1e18, 1e18, 1e18],
            "edge_rows": [None, None, None],
        }
        for s in shots
    ]
    want_thumb = style == "frame"
    bar = ProgressBar(total * fps, "采样", quiet=quiet)
    processed = 0

    def finalize(i: int) -> None:
        s, st = shots[i], states[i]
        counts = st["counts"]
        sums = st["sums"]
        if sums is None or counts is None or counts.sum() == 0:
            s.trace = np.zeros((1, 2, 3), dtype=np.uint8)
            s.color = (0, 0, 0)
            s.empty = True
            return
        safe = np.maximum(counts, 1).astype(np.float32)
        mean = sums / safe[:, None, None]
        mean = fill_empty_bins(mean, counts)
        mean = pool_rows(mean, max_trace_rows)
        s.trace = np.clip(mean, 0, 255).astype(np.uint8)
        color = np.median(s.trace.reshape(-1, 3), axis=0)
        s.color = tuple(int(v) for v in np.clip(color, 0, 255))
        if all(r is not None for r in st["edge_rows"]):
            s.edges = np.clip(np.stack(st["edge_rows"]), 0, 255).astype(np.uint8)
        if want_thumb and st["best_frame"] is not None:
            img = Image.fromarray(st["best_frame"], "RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            s.thumb = buf.getvalue()
        st["sums"] = st["counts"] = st["best_frame"] = None

    idx = 0
    last_frame = None
    for t, arr in iter_frames(path, fps, width, height, quiet=quiet):
        if t > total + 1.0 / fps:
            break
        if idx == len(shots) - 1 and t >= shots[-1].end - 1e-9:
            break
        tol = 0.05  # 帧时间戳与镜头边界的舍入误差上限
        while idx + 1 < len(shots) and t >= shots[idx].end + tol:
            finalize(idx)
            idx += 1
        while idx < len(shots) and t < shots[idx].start - tol:
            finalize(idx)
            idx += 1
        if idx >= len(shots):
            break

        s, st = shots[idx], states[idx]
        rows = arr.mean(axis=1, dtype=np.float32)  # (h, 3) 每行平均色
        if st["sums"] is None:
            st["sums"] = np.zeros((st["bins"], rows.shape[0], 3), dtype=np.float32)
            st["counts"] = np.zeros(st["bins"], dtype=np.int32)
        b = int((t - s.start) / max(s.dur, 1e-6) * st["bins"])
        b = min(max(b, 0), st["bins"] - 1)
        st["sums"][b] += rows
        st["counts"][b] += 1
        for k, target in enumerate((s.start, s.start + s.dur / 2.0, s.end - 1e-3)):
            d = abs(t - target)
            if d < st["edge_t"][k]:
                st["edge_t"][k] = d
                st["edge_rows"][k] = rows.copy()
        last_frame = arr

        if want_thumb:
            mid = s.start + s.dur / 2.0
            if abs(t - mid) < st["best_t"]:
                st["best_t"] = abs(t - mid)
                st["best_frame"] = arr.copy()
        processed += 1
        bar.update(processed)

    bar.finish()
    for i in range(idx, len(shots)):
        finalize(i)
    fixed = fill_empty_shots(shots)
    if fixed:
        log(f"  为 {fixed} 个空镜头补了相邻色带", quiet)
    log(f"  采样 {len(shots)} 个分镜完成", quiet)
    del last_frame


def collect_shot_edges(shots: list["Shot"], path: str, fps: float, width: int,
                       height: int, quiet: bool = False) -> None:
    """缓存里没有首/中/末帧时，单独扫一遍补取行均值。"""
    starts = np.array([s.start for s in shots], np.float64)
    best = np.full((len(shots), 3), np.inf)
    edge_rows: list[list[np.ndarray | None]] = [[None, None, None] for _ in shots]
    bar = ProgressBar(shots[-1].end * fps, "取首/中/末帧", quiet=quiet)
    processed = 0

    for t, arr in iter_frames(path, fps, width, height, quiet=quiet):
        processed += 1
        bar.update(processed)
        i = int(np.searchsorted(starts, t, side="right") - 1)
        if i < 0 or i >= len(shots):
            continue
        s = shots[i]
        if t < s.start - 1e-9 or t >= s.end + 1e-9:
            continue
        rows = arr.mean(axis=1, dtype=np.float32)
        for k, target in enumerate((s.start, s.start + s.dur / 2.0, s.end - 1e-3)):
            d = abs(t - target)
            if d < best[i, k]:
                best[i, k] = d
                edge_rows[i][k] = np.clip(rows, 0, 255).astype(np.uint8)

    bar.finish()
    for s, rows3 in zip(shots, edge_rows):
        if all(r is not None for r in rows3):
            s.edges = np.stack(rows3)


def collect_thumbnails(shots: list["Shot"], path: str, media: dict,
                       quiet: bool = False, width: int = 480, fps: float = 3.0) -> None:
    """缓存命中但缺少缩略图时，单独扫一遍片子补取每个分镜的中间帧。"""
    sw, sh = sample_dimensions(media, max(160, width))
    starts = np.array([s.start for s in shots], np.float64)
    mids = np.array([s.start + s.dur / 2.0 for s in shots], np.float64)
    best_diff = np.full(len(shots), np.inf, np.float64)
    best_frame: list[np.ndarray | None] = [None] * len(shots)
    bar = ProgressBar(shots[-1].end * fps, "取中间帧", quiet=quiet)
    processed = 0

    for t, arr in iter_frames(path, fps, sw, sh, quiet=quiet):
        processed += 1
        bar.update(processed)
        i = int(np.searchsorted(starts, t, side="right") - 1)
        i = max(0, min(i, len(shots) - 1))
        if t >= shots[i].end - 1e-9:
            continue
        d = abs(t - mids[i])
        if d < best_diff[i]:
            best_diff[i] = d
            best_frame[i] = arr.copy()

    bar.finish()
    for s, frame in zip(shots, best_frame):
        if frame is None:
            continue
        buf = io.BytesIO()
        Image.fromarray(frame, "RGB").save(buf, format="JPEG", quality=88)
        s.thumb = buf.getvalue()


def resample_trace(trace: np.ndarray, n: int) -> np.ndarray:
    """沿时间轴线性重采样到 n 个 bin。"""
    if trace.shape[0] == n:
        return trace.astype(np.float32)
    pos = np.linspace(0, trace.shape[0] - 1, n)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, trace.shape[0] - 1)
    w = (pos - lo)[:, None, None].astype(np.float32)
    return trace[lo].astype(np.float32) * (1.0 - w) + trace[hi].astype(np.float32) * w


def merge_similar_shots(shots: list[Shot], max_dist: float) -> list[Shot]:
    """把代表色几乎相同的相邻分镜并成一条，减少无意义的接缝。"""
    if max_dist <= 0 or len(shots) < 2:
        return shots
    merged = [shots[0]]
    for s in shots[1:]:
        prev = merged[-1]
        dist = float(np.linalg.norm(np.array(prev.color, float) - np.array(s.color, float)))
        if dist < max_dist:
            merged[-1] = _combine_two(prev, s)
        else:
            merged.append(s)
    return merged


def _combine_two(a: Shot, b: Shot) -> Shot:
    bins = min(64, max(2, a.trace.shape[0] + b.trace.shape[0]))
    na = max(1, int(round(bins * a.dur / max(a.dur + b.dur, 1e-9))))
    nb = max(1, bins - na)
    trace = np.concatenate(
        [resample_trace(a.trace, na), resample_trace(b.trace, nb)], axis=0
    )
    trace = np.clip(trace, 0, 255).astype(np.uint8)
    color = tuple(int(v) for v in np.clip(np.median(trace.reshape(-1, 3), axis=0), 0, 255))
    thumb = a.thumb if a.dur >= b.dur else b.thumb
    return Shot(a.start, b.end, trace=trace, color=color, thumb=thumb)


# ---------------------------------------------------------------- 长图渲染

def compute_widths(shots: list[Shot], px_per_second: float, min_px: int = 1,
                   px_per_shot: int = 0) -> int:
    """分配每个分镜的像素宽度。

    默认按「时长 × 每秒像素」并用最大余数法补齐；
    px_per_shot > 0 时每个分镜固定宽度（1 就是“一分镜一像素”）。
    """
    if not shots:
        return 0
    if px_per_shot > 0:
        for s in shots:
            s.width = int(px_per_shot)
        return int(px_per_shot) * len(shots)
    total_w = max(1, int(round(sum(s.dur for s in shots) * px_per_second)))
    raw = np.array([s.dur for s in shots]) * px_per_second
    widths = np.maximum(min_px, np.floor(raw).astype(np.int64))
    diff = total_w - int(widths.sum())
    if diff > 0:
        fractions = raw - np.floor(raw)
        order = np.argsort(-fractions)
        for k in range(diff):
            widths[order[k % len(widths)]] += 1
    elif diff < 0:
        order = np.argsort(-widths)
        k = 0
        while diff < 0:
            i = order[k % len(order)]
            if widths[i] > min_px:
                widths[i] -= 1
                diff += 1
            k += 1
            if k > 10 * len(widths):
                break
    for s, w in zip(shots, widths):
        s.width = int(w)
    return int(widths.sum())


def color_column(rows: np.ndarray, px: int, height: int) -> Image.Image:
    """把一行的平均色（rows×3）渲染成 px 宽、height 高的竖向色带。"""
    col = np.clip(rows, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    return Image.fromarray(col, "RGB").resize((max(1, px), height), Image.BICUBIC)


def fml_strip(shot: Shot, px: int, height: int) -> Image.Image:
    """首帧/中间帧/末帧各 px 像素宽，拼成一个分镜。"""
    if shot.edges is not None:
        frames = [shot.edges[0], shot.edges[1], shot.edges[2]]
    elif shot.trace is not None:
        tr = shot.trace
        frames = [tr[0], tr[len(tr) // 2], tr[-1]]
    else:
        frames = [np.full((2, 3), shot.color, np.uint8)] * 3
    w = max(1, px)
    out = Image.new("RGB", (w * 3, height))
    for k, rows in enumerate(frames):
        out.paste(color_column(rows, w, height), (k * w, 0))
    return out


def shot_strip(shot: Shot, height: int, style: str) -> Image.Image:
    w = max(1, shot.width)
    if w == 1 and style == "gradient" and shot.trace is not None:
        # 一列：先沿时间取平均，再纵向拉伸
        col = shot.trace.astype(np.float32).mean(axis=0)
        col = np.clip(col, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        return Image.fromarray(col, "RGB").resize((1, height), Image.BICUBIC)
    if style == "dominant":
        color = tuple(int(c) for c in shot.color)
        return Image.new("RGB", (w, height), color)
    if style == "frame" and shot.thumb:
        img = Image.open(io.BytesIO(shot.thumb)).convert("RGB")
    elif shot.trace is not None:
        img = Image.fromarray(np.ascontiguousarray(shot.trace), "RGB")
    else:
        img = Image.new("RGB", (w, height), tuple(int(c) for c in shot.color))
    return img.resize((w, height), Image.BICUBIC)


def compute_equal_widths(shots: list["Shot"], total_width: int) -> int:
    """每个分镜等宽，总宽精确等于 total_width（余数分给前几个）。"""
    if not shots:
        return 0
    n = len(shots)
    base, rem = divmod(max(n, int(total_width)), n)
    for i, s in enumerate(shots):
        s.width = base + (1 if i < rem else 0)
    return base * n + rem


def split_segments(widths: list[int], max_width: int) -> list[tuple[int, int]]:
    total = sum(widths)
    if total <= max_width or len(widths) <= 1:
        return [(0, len(widths))]
    n = math.ceil(total / max_width)
    target = math.ceil(total / n)
    segments: list[tuple[int, int]] = []
    start, acc = 0, 0
    for i, w in enumerate(widths):
        if acc > 0 and acc + w > target and len(segments) < n - 1:
            segments.append((start, i))
            start, acc = i, 0
        acc += w
    segments.append((start, len(widths)))
    return segments


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # 老版本 Pillow 没有 size 参数
        return ImageFont.load_default()


def timecode(seed: float, long_fmt: bool) -> str:
    seed = max(0, int(round(seed)))
    h, rem = divmod(seed, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if long_fmt else f"{m:02d}:{s:02d}"


def pick_tick_interval(t0: float, t1: float, px_per_second: float, width: int) -> float:
    target_px = 190
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200):
        if step * px_per_second >= target_px or step >= t1 - t0:
            return float(step)
    return float(max(1, int(t1 - t0)))


def with_ruler(img: Image.Image, t0: float, t1: float, draw_scale: float = 1.0) -> Image.Image:
    """在长图底部加一条时间刻度（深色底、浅色字）。"""
    width, height = img.size
    if t1 <= t0 or width <= 4:
        return img
    pps = width / (t1 - t0)
    footer = int(round(42 * max(1.0, draw_scale)))
    out = Image.new("RGB", (width, height + footer), (12, 12, 14))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    font = load_font(int(round(17 * max(1.0, draw_scale))))
    long_fmt = t1 >= 3600
    interval = pick_tick_interval(t0, t1, pps, width)

    first = math.floor(t0 / interval) * interval
    tick = first
    while tick <= t1 + 1e-6:
        x = (tick - t0) * pps
        if -1 <= x <= width:
            x = min(max(x, 0), width - 1)
            draw.line([(x, height), (x, height + int(8 * draw_scale))], fill=(120, 120, 128), width=1)
            label = timecode(tick, long_fmt)
            tw = draw.textlength(label, font=font)
            tx = min(max(0, x - tw / 2), width - tw - 2)
            draw.text((tx, height + int(10 * draw_scale)), label, font=font, fill=(190, 190, 196))
        tick += interval
    draw.line([(0, height), (width - 1, height)], fill=(60, 60, 66), width=1)
    return out


def preview_scale(total_width: int, height: int, max_width: int,
                  max_height: int = 0) -> float:
    """预览缩放比例：默认按宽度等比，max_height>0 时再加高度上限。"""
    scale = min(1.0, max_width / max(total_width, 1))
    if max_height and max_height > 0 and height * scale > max_height:
        scale = max_height / max(height, 1)
    return scale


NICE_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200)


def with_mapped_ruler(img: Image.Image, shots: list[Shot]) -> Image.Image:
    """每镜头等宽时，按镜头在总宽中的位置映射原片时间。"""
    if not shots:
        return img
    t0, t1 = shots[0].start, shots[-1].end
    width, height = img.size
    if t1 <= t0 or width < 4:
        return img
    starts = np.array([s.start for s in shots], np.float64)
    cum = np.concatenate([[0.0], np.cumsum([s.width for s in shots[:-1]])])

    def x_of(t: float) -> float:
        i = int(np.searchsorted(starts, t, side="right") - 1)
        i = min(max(i, 0), len(shots) - 1)
        s = shots[i]
        frac = min(max((t - s.start) / max(s.dur, 1e-9), 0.0), 1.0)
        return float(cum[i] + frac * s.width)

    span = t1 - t0
    interval = next((st for st in NICE_STEPS if span / st <= 12), 7200)
    footer = 44
    out = Image.new("RGB", (width, height + footer), (12, 12, 14))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    font = load_font(17)
    long_fmt = t1 >= 3600
    last_label_x = -1e9
    last_label_right = -1e9
    tick = math.floor(t0 / interval) * interval
    while tick <= t1 + 1e-6:
        x = x_of(tick)
        if -1 <= x <= width:
            x = min(max(x, 0), width - 1)
            draw.line([(x, height), (x, height + 8)], fill=(120, 120, 128), width=1)
            if x - last_label_x >= 70:
                label = timecode(tick, long_fmt)
                tw = draw.textlength(label, font=font)
                tx = min(max(0, x - tw / 2), width - tw - 2)
                if tx >= last_label_right + 8:
                    draw.text((tx, height + 11), label, font=font, fill=(190, 190, 196))
                    last_label_x = x
                    last_label_right = tx + tw
        tick += interval
    draw.line([(0, height), (width - 1, height)], fill=(60, 60, 66), width=1)
    return out


def render_preview(
    shots: list[Shot], total_width: int, height: int, style: str,
    max_width: int = 2600, max_height: int = 0,
    px_per_shot: int = 0, shot_samples: str = "avg",
) -> Image.Image:
    scale = preview_scale(total_width, height, max_width, max_height)
    out_w = max(1, int(round(total_width * scale)))
    out_h = max(1, int(round(height * scale)))
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x = 0
    for s in shots:
        w = max(1, int(round(s.width * scale)))
        strip = shot_strip(s, height, style)
        strip = strip.resize((w, out_h), Image.LANCZOS)
        arr = np.asarray(strip, dtype=np.uint8)
        end_x = min(out_w, x + w)
        if end_x > x:
            canvas[:, x:end_x] = arr[:, : end_x - x]
        x += w
    return Image.fromarray(canvas, "RGB")


def save_image(img: Image.Image, path: str, quality: int) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".webp"):
        img.save(path)
    else:
        img.save(path, quality=quality, subsampling=1, optimize=True)


# ---------------------------------------------------------------- 主流程

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="film_shot_strip.py",
        description="把电影按分镜切成色带长图：横向 = 时间，宽度 ∝ 分镜时长，颜色 = 分镜色彩。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="输入视频文件")
    p.add_argument("-o", "--output", default=None,
                   help="输出路径（.jpg/.png，或作为文件名前缀）；默认 <视频名>_strip.jpg")
    p.add_argument("--height", type=int, default=argparse.SUPPRESS,
                   help="长图高度（默认 1080；--px-per-shot 模式默认 1000）")
    p.add_argument("--px-per-second", default="auto",
                   help="每秒时间占多少像素；auto 会按目标总宽自动取值")
    p.add_argument("--target-width", type=int, default=14000,
                   help="--px-per-second auto 时的目标总宽度")
    p.add_argument("--max-width", type=int, default=24000,
                   help="单张图最大宽度，超出则自动切成多张")
    p.add_argument("--min-shot-px", type=int, default=2, help="分镜在图中最小像素宽度")
    p.add_argument("--px-per-shot", type=int, default=0,
                   help="每个取色帧固定像素宽度，1 = 每个取色帧 1px（默认首/中/末帧，故每镜 3px）")
    p.add_argument("--equal-width", action="store_true",
                   help="每个分镜等宽，总宽取 --target-width（忽略时长比例）")
    p.add_argument("--per-second", nargs="?", const=1, type=int, default=None, metavar="PX",
                   help="每秒抽 1 帧拼长图，每帧占 PX 像素（默认 1）；跳过镜头检测")
    p.add_argument("--shot-samples", choices=["avg", "fml"], default="fml",
                   help="一分镜一像素时取色方式：avg=整镜平均；fml=首帧/中间帧/末帧各一列")
    p.add_argument("--style", choices=["gradient", "dominant", "frame"], default="gradient",
                   help="gradient=竖向渐变色带；dominant=单色块；frame=真实画面缩略")
    p.add_argument("--method", choices=["auto", "scenedetect", "scdet", "uniform"],
                   default="auto",
                   help="分镜检测方式；auto 优先用 PySceneDetect，缺失时回退 ffmpeg scdet")
    p.add_argument("--detector", choices=["content", "adaptive"], default="content",
                   help="PySceneDetect 检测器：content 通用，adaptive 更适合手持/运动镜头")
    p.add_argument("--scene-threshold", type=float, default=0.35,
                   help="切分灵敏度（0~1，越小切得越碎；0.25~0.5 常用）")
    p.add_argument("--shot-length", type=float, default=3.0,
                   help="--method uniform 时每个分镜的秒数")
    p.add_argument("--min-shot", type=float, default=0.6,
                   help="短于该秒数的分镜并入相邻镜头")
    p.add_argument("--max-shot", type=float, default=0.0,
                   help="长于该秒数的镜头平均切开；0 表示不切")
    p.add_argument("--merge-similar", type=float, default=0.0,
                   help="合并代表色相近（RGB 距离小于该值）的相邻分镜；0 关闭，建议 6~12")
    p.add_argument("--sample-fps", type=float, default=0.0,
                   help="采样帧率；0 表示按目标采样数自动计算")
    p.add_argument("--sample-width", type=int, default=0,
                   help="采样帧宽度；0 表示按风格自动（gradient 160 / frame 480）")
    p.add_argument("--blur", type=float, default=0.7, help="整图高斯柔化半径，0 关闭")
    p.add_argument("--median", type=int, default=0,
                   help="中值滤波尺寸（3/5），模仿参考项目的颗粒感；0 关闭")
    p.add_argument("--ruler", dest="ruler", action="store_true", default=True,
                   help="在图底部画时间刻度（默认开启）")
    p.add_argument("--no-ruler", dest="ruler", action="store_false",
                   default=argparse.SUPPRESS, help="不画底部时间刻度")
    p.add_argument("--quality", type=int, default=92, help="JPEG 质量")
    p.add_argument("--preview-width", type=int, default=2600, help="预览图最大宽度")
    p.add_argument("--preview-height", type=int, default=0,
                   help="预览图最大高度；0 表示按宽度等比缩放")
    p.add_argument("--no-preview", action="store_true", help="不生成预览图")
    p.add_argument("--no-json", action="store_true", help="不输出分镜清单 JSON")
    p.add_argument("--cache-dir", default="out/.cache",
                   help="分镜缓存目录（片源哈希 + 帧区间 + 色带）")
    p.add_argument("--shots-dir", default="out",
                   help="分镜清单 JSON 的输出目录")
    p.add_argument("--no-cache", action="store_true", help="不读也不写分镜缓存")
    p.add_argument("--refresh-shots", action="store_true", help="忽略缓存，强制重新检测分镜")
    p.add_argument("-q", "--quiet", action="store_true", help="减少日志输出")
    p.add_argument("--no-progress", action="store_true", help="不显示进度条")
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    global PROGRESS_ENABLED
    args = build_arg_parser().parse_args(argv)
    PROGRESS_ENABLED = not args.no_progress
    if not hasattr(args, "height"):
        args.height = 1000 if args.px_per_shot > 0 else 1080

    if not os.path.isfile(args.video):
        log(f"找不到视频：{args.video}")
        return 2
    for tool, name in ((FFMPEG, "ffmpeg"), (FFPROBE, "ffprobe")):
        if not shutil.which(tool) and not os.path.isabs(tool):
            log(f"未找到 {name}，请先安装：ffmpeg")
            return 2

    log(f"读取视频信息：{args.video}")
    media = probe_media(args.video)
    duration = media["duration"]
    log(
        f"  {media['width']}×{media['height']}，时长 {timecode(duration, duration >= 3600)}"
        f"（{duration:.1f}s）"
    )

    per_second = args.per_second
    if per_second is not None and per_second <= 0:
        log("--per-second 的像素宽度必须大于 0")
        return 2
    if per_second is not None and args.style == "frame":
        log("--per-second 不支持 --style frame，请用 gradient 或 dominant")
        return 2

    # ---- 0. 片源哈希与分镜缓存
    cache_dir = args.cache_dir or "."
    shots_dir = args.shots_dir or "."
    source_stem = sanitize_name(os.path.splitext(os.path.basename(args.video))[0])
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(shots_dir, exist_ok=True)
    fps = media["fps"]
    frame_count = max(1, int(round(duration * fps)))
    if per_second is not None:
        det_params = {"mode": "per_second", "interval": 1.0}
    else:
        det_params = {
            "method": args.method,
            "detector": args.detector,
            "scene_threshold": round(args.scene_threshold, 4),
            "min_shot": round(args.min_shot, 4),
            "max_shot": round(args.max_shot, 4),
            "shot_length": round(args.shot_length, 4),
        }
    source_hash = None
    cache_json = cache_npz = None
    cached = None
    traces_loaded = False

    if not args.no_cache:
        log(f"0/4 计算片源哈希（{os.path.getsize(args.video) / 2**20:.0f} MB）…")
        source_hash = file_sha256(args.video, args.quiet)
        log(f"  {source_hash[: 7 + 16]}…", args.quiet)
        suffix = ".persecond" if per_second is not None else ""
        cache_json = os.path.join(cache_dir, f"{source_stem}{suffix}.shots.json")
        cache_npz = os.path.join(cache_dir, f"{source_stem}{suffix}.traces.npz")
        if args.refresh_shots:
            log("  按 --refresh-shots 忽略已有缓存", args.quiet)
        else:
            cached, reason = load_shot_cache(cache_json, source_hash, det_params)
            if cached is None:
                log(f"  未使用缓存：{reason}", args.quiet)
            else:
                log(f"  命中缓存，跳过检测：{cache_json}", args.quiet)

    if cached is not None:
        ranges = [(int(s["start"]), int(s["end"])) for s in cached["shots"]]
        shots = [Shot(a / fps, b / fps) for a, b in ranges]
        log(f"  缓存中有 {len(shots)} 个分镜", args.quiet)
    elif per_second is not None:
        n = max(1, int(math.floor(duration + 1e-9)))
        shots = [Shot(float(i), float(i + 1)) for i in range(n)]
        ranges = shot_frame_ranges(shots, fps, frame_count)
        shots = [Shot(a / fps, b / fps) for a, b in ranges]
        log(f"  每秒抽 1 帧：共 {len(shots)} 帧", args.quiet)
    else:
        # ---- 1. 分镜检测
        log("1/4 检测分镜 …")
        if args.method == "uniform":
            step = max(0.1, args.shot_length)
            cuts = [step * i for i in range(1, math.ceil(duration / step))]
            log(f"  按固定 {step:g}s 切分", args.quiet)
        else:
            use_pyscene = args.method in ("auto", "scenedetect")
            if use_pyscene:
                try:
                    cuts = detect_cuts_scenedetect(
                        args.video, args.scene_threshold, args.quiet, args.detector,
                        frame_count
                    )
                except RuntimeError as exc:
                    if args.method == "scenedetect":
                        log(f"  {exc}")
                        return 1
                    log(f"  {exc}；改用 ffmpeg scdet")
                    use_pyscene = False
            if not use_pyscene:
                cuts = detect_cuts_scdet(args.video, args.scene_threshold * 25.0, args.quiet)

        shots = normalize_shots(cuts, duration, args.min_shot, args.max_shot)
        sizes = [s.dur for s in shots]
        log(
            f"  共 {len(shots)} 个分镜，时长 {min(sizes):.2f}s ~ {max(sizes):.2f}s，"
            f"平均 {sum(sizes) / len(sizes):.2f}s"
        )
        ranges = shot_frame_ranges(shots, fps, frame_count)
        shots = [Shot(a / fps, b / fps) for a, b in ranges]

    # ---- 2. 宽度 / 采样参数
    shot_total = sum(s.dur for s in shots)
    if str(args.px_per_second).strip().lower() == "auto":
        pps = min(120.0, max(0.5, args.target_width / max(shot_total, 1e-9)))
    else:
        try:
            pps = float(args.px_per_second)
        except ValueError:
            log("--px-per-second 需要是数字或 auto")
            return 2
        if pps <= 0:
            log("--px-per-second 必须大于 0")
            return 2

    sample_w = args.sample_width or (480 if args.style == "frame" else 160)
    sw, sh = sample_dimensions(media, max(16, sample_w))
    sample_fps = args.sample_fps
    if per_second is not None:
        sample_fps = 1.0
    else:
        if sample_fps <= 0:
            sample_fps = min(8.0, max(0.5, 8000.0 / max(duration, 1.0)))
        short = min(s.dur for s in shots)
        sample_fps = max(sample_fps, min(12.0, max(0.5, 2.0 / max(short, 0.05))))

    if cached is not None and load_cached_traces(cache_npz, shots):
        traces_loaded = True
        log("  复用缓存色带，跳过采样", args.quiet)
        if per_second is not None:
            fixed = repair_isolated_black(shots)
            if fixed:
                log(f"  修复旧缓存中 {fixed} 个缺失的秒采样", args.quiet)
    else:
        log(
            f"2/4 采样（{sw}×{sh}，约 {sample_fps:.2f} fps）…",
            args.quiet,
        )
        collect_shot_data(shots, args.video, sample_fps, sw, sh, args.style, quiet=args.quiet)

    thumbs_added = False
    if args.style == "frame" and any(s.thumb is None for s in shots):
        log("  提取每个分镜的中间帧 …", args.quiet)
        collect_thumbnails(shots, args.video, media, quiet=args.quiet)
        thumbs_added = True

    edges_added = False
    if (per_second is None and args.px_per_shot > 0
            and args.shot_samples == "fml" and any(s.edges is None for s in shots)):
        log("  提取每个分镜的首帧/中间帧/末帧 …")
        collect_shot_edges(shots, args.video, sample_fps, sw, sh, quiet=args.quiet)
        edges_added = True

    if cache_json and (not traces_loaded or thumbs_added or edges_added):
        save_cached_traces(cache_npz, shots)
        write_shot_cache(cache_json, cache_npz, args.video, source_hash, fps,
                         frame_count, duration, det_params, ranges, shots)

    if per_second is None and args.merge_similar > 0:
        before = len(shots)
        shots = merge_similar_shots(shots, args.merge_similar)
        log(f"  合并相近分镜：{before} → {len(shots)}", args.quiet)

    if per_second is not None:
        total_w = compute_widths(shots, pps, max(1, args.min_shot_px),
                                 px_per_shot=per_second)
        pps = total_w / max(shot_total, 1e-9)
        log(
            f"3/4 排版：每秒 1 帧 × {per_second}px，共 {len(shots)} 列，总宽 {total_w}px",
            args.quiet,
        )
    elif args.equal_width:
        total_w = compute_equal_widths(shots, args.target_width)
        pps = total_w / max(shot_total, 1e-9)
        widths = [s.width for s in shots]
        log(
            f"3/4 排版：等宽 {min(widths)}~{max(widths)}px/分镜，总宽 {total_w}px",
            args.quiet,
        )
    elif args.px_per_shot > 0:
        per_shot = args.px_per_shot * (3 if args.shot_samples == "fml" else 1)
        total_w = compute_widths(shots, pps, max(1, args.min_shot_px),
                                 px_per_shot=per_shot)
        pps = total_w / max(shot_total, 1e-9)
        tag = "首/中/末帧各 1 列" if args.shot_samples == "fml" else "整镜平均"
        log(f"3/4 排版：每个分镜 {per_shot}px（{tag}），总宽 {total_w}px", args.quiet)
    else:
        total_w = compute_widths(shots, pps, max(1, args.min_shot_px))
        log(f"3/4 排版：总宽 {total_w}px，{pps:g} px/s", args.quiet)

    # ---- 3. 输出路径
    out = args.output or (os.path.splitext(os.path.basename(args.video))[0] + "_strip.jpg")
    root, ext = os.path.splitext(out)
    if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        root, ext = out, ".jpg"
    out_dir = os.path.dirname(os.path.abspath(root))
    os.makedirs(out_dir, exist_ok=True)
    if args.max_width > JPEG_MAX_DIM - 100:
        log(f"  提示：--max-width 超过 JPEG 上限，已按 {JPEG_MAX_DIM - 100} 处理")
    seg_max = min(max(1000, args.max_width), JPEG_MAX_DIM - 100)
    segments = split_segments([s.width for s in shots], seg_max)

    # ---- 4. 渲染
    log(f"4/4 渲染 {len(segments)} 段长图 …", args.quiet)
    render_bar = ProgressBar(total_w, "渲染长图", quiet=args.quiet)
    rendered = 0
    outputs: list[str] = []
    for si, (a, b) in enumerate(segments):
        part = shots[a:b]
        width = sum(s.width for s in part)
        canvas = Image.new("RGB", (max(1, width), args.height), (0, 0, 0))
        x = 0
        for s in part:
            if per_second is None and args.px_per_shot > 0 and args.shot_samples == "fml":
                strip = fml_strip(s, args.px_per_shot, args.height)
            else:
                strip = shot_strip(s, args.height, args.style)
            canvas.paste(strip, (x, 0))
            x += s.width
            del strip
            rendered += s.width
            render_bar.update(rendered)
        if args.blur > 0 and args.px_per_shot != 1 and per_second != 1:
            canvas = canvas.filter(ImageFilter.GaussianBlur(args.blur))
        if args.median >= 3 and args.px_per_shot != 1:
            canvas = canvas.filter(ImageFilter.MedianFilter(args.median))
        if args.ruler:
            if width >= 300:
                if args.px_per_shot > 0 or args.equal_width:
                    canvas = with_mapped_ruler(canvas, part)
                else:
                    canvas = with_ruler(canvas, part[0].start, part[-1].end)
            elif si == 0:
                log("  图太窄，已跳过时间刻度", args.quiet)
        if len(segments) == 1:
            path = root + ext
        else:
            path = f"{root}_{si + 1:02d}{ext}"
        save_image(canvas, path, args.quality)
        outputs.append(path)
        log(f"  {os.path.basename(path)}  {canvas.size[0]}×{canvas.size[1]}")
        del canvas

    # ---- 预览
    render_bar.finish()
    preview_path = None
    if not args.no_preview and args.preview_width > 0:
        scale = preview_scale(total_w, args.height, args.preview_width, args.preview_height)
        if scale < 1.0:
            preview_path = root + "_preview.jpg"
            prev = render_preview(shots, total_w, args.height, args.style,
                                  max_width=args.preview_width,
                                  max_height=args.preview_height,
                                  px_per_shot=args.px_per_shot,
                                  shot_samples=args.shot_samples)
            save_image(prev, preview_path, 90)
            log(
                f"  预览：{os.path.basename(preview_path)}  {prev.size[0]}×{prev.size[1]}",
                args.quiet,
            )

    # ---- 分镜清单
    manifest = None
    if not args.no_json:
        suffix = "_persecond" if per_second is not None else ""
        manifest = os.path.join(shots_dir, f"{source_stem}{suffix}_shots.json")
        final_ranges = shot_frame_ranges(shots, fps, frame_count)
        data = {
            "video": os.path.abspath(args.video),
            "source_hash": source_hash,
            "time_unit": "frames",
            "fps": round(fps, 6),
            "frame_count": frame_count,
            "duration_seconds": round(duration, 3),
            "px_per_second": round(pps, 4),
            "image_height": args.height,
            "style": args.style,
            "method": args.method,
            "detector": args.detector,
            "scene_threshold": args.scene_threshold,
            "min_shot": args.min_shot,
            "merge_similar": args.merge_similar,
            "px_per_shot": args.px_per_shot,
            "shot_samples": args.shot_samples,
            "equal_width": args.equal_width,
            "mode": "per_second" if per_second is not None else "shots",
            "per_second_px": per_second,
            "cached_shots": cached is not None,
            "cached_traces": traces_loaded,
            "total_width": total_w,
            "outputs": outputs,
            "shots": [
                {
                    "index": i,
                    "start": st,
                    "end": en,
                    "duration": en - st,
                    "color": "#%02x%02x%02x" % tuple(int(c) for c in s.color),
                    "width_px": s.width,
                }
                for i, (s, (st, en)) in enumerate(zip(shots, final_ranges))
            ],
        }
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    log("完成。")
    print("\n".join(outputs))
    if preview_path:
        print(preview_path)
    if manifest:
        print(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
