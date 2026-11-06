#!/usr/bin/env bash
# 生成一个 34.5 秒、12 个不同颜色/时长的测试视频，用来快速验证工具。
# 需要 ffmpeg。
set -euo pipefail
cd "$(dirname "$0")"

ffmpeg -y -hide_banner -loglevel error \
  -f lavfi -i "gradients=s=960x540:d=3:c0=0x081c33:c1=0x2e8b9a:speed=0.12:r=24" \
  -f lavfi -i "gradients=s=960x540:d=1:c0=0x5a0505:c1=0xff4a3a:speed=0.30:r=24" \
  -f lavfi -i "gradients=s=960x540:d=4:c0=0x3a1a06:c1=0xf2b26b:speed=0.08:r=24" \
  -f lavfi -i "gradients=s=960x540:d=2:c0=0x061a0c:c1=0x3f9a52:speed=0.20:r=24" \
  -f lavfi -i "gradients=s=960x540:d=6:c0=0x1a0626:c1=0xc06bd8:speed=0.05:r=24" \
  -f lavfi -i "gradients=s=960x540:d=2:c0=0xdfe9f2:c1=0x9ab6d6:speed=0.18:r=24" \
  -f lavfi -i "gradients=s=960x540:d=2.5:c0=0x050a18:c1=0x233d7a:speed=0.10:r=24" \
  -f lavfi -i "gradients=s=960x540:d=1.5:c0=0x4a3a05:c1=0xffd95a:speed=0.25:r=24" \
  -f lavfi -i "gradients=s=960x540:d=5:c0=0x3a0630:c1=0xd85ab0:speed=0.07:r=24" \
  -f lavfi -i "gradients=s=960x540:d=1:c0=0xf2f2f2:c1=0xb0b0b0:speed=0.15:r=24" \
  -f lavfi -i "gradients=s=960x540:d=4:c0=0x071a10:c1=0x2f7a4a:speed=0.09:r=24" \
  -f lavfi -i "gradients=s=960x540:d=3:c0=0x4a1505:c1=0xff8a4a:speed=0.16:r=24" \
  -filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v][6:v][7:v][8:v][9:v][10:v][11:v]concat=n=12:v=1:a=0,format=yuv420p[v]" \
  -map "[v]" -c:v libx264 -preset veryfast -crf 20 demo.mp4

echo "已生成 examples/demo.mp4（34.5 秒，12 个镜头）"
echo "python film_shot_strip.py examples/demo.mp4 -o out/demo.jpg --target-width 4000 --height 720"
