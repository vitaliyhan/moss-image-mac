#!/usr/bin/env python3
"""Detailed scene composition with overlapping bounding boxes.

MOSS-Image layout prompts accept boxes that overlap: nested and layered boxes compose a scene
in depth (a sign on a facade, a cat on a bench, a person in front of a bicycle). This script
renders examples/overlapping_scene.json and draws the boxes over the result.

    ./start_server.sh &                                   # or let the MCP server start it
    .venv/bin/python examples/overlapping_scene.py                    # standard, ~1 min
    .venv/bin/python examples/overlapping_scene.py --quality high     # 50 steps, ~21 min
    .venv/bin/python examples/overlapping_scene.py --spec my_scene.json --seed 7

Spec format: {"width", "height", "caption", "items": [{"label", "caption", "box": [x0, y0, x1, y1]}]}
with boxes in output pixels, top-left origin (the same format as the MCP tool).
"""
import argparse
import json
import os
import sys
import time

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import moss_client  # noqa: E402

COLORS = ["#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93", "#ff924c",
          "#52a675", "#4267ac", "#c5ca30", "#b5179e", "#f15bb5", "#00bbf9"]


def draw_boxes(image_path, items, out_path):
    """Overlay numbered, semi-transparent boxes so the composition is easy to read."""
    base = Image.open(image_path).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except OSError:
        font = ImageFont.load_default()
    for i, it in enumerate(items):
        c = COLORS[i % len(COLORS)]
        rgb = tuple(int(c[k:k + 2], 16) for k in (1, 3, 5))
        x0, y0, x1, y1 = it["box"]
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=rgb + (255,), width=3, fill=rgb + (28,))
        tag = str(i + 1)
        tw = d.textlength(tag, font=font)
        d.rectangle([x0 + 3, y0 + 3, x0 + 11 + tw, y0 + 27], fill=rgb + (230,))
        d.text((x0 + 7, y0 + 5), tag, fill=(0, 0, 0, 255), font=font)
    Image.alpha_composite(base, layer).convert("RGB").save(out_path, quality=90)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default=os.path.join(ROOT, "examples", "overlapping_scene.json"))
    p.add_argument("--quality", choices=["draft", "standard", "high"], default="standard")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default=os.path.join(ROOT, "outputs"))
    a = p.parse_args()

    spec = json.load(open(a.spec))
    w, h = moss_client.validate_size(spec["width"], spec["height"])
    prompt = moss_client.layout_prompt(spec["caption"], spec["items"], w, h)
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, f"overlap_{a.quality}_{a.seed}")

    def progress(step, total, msg):
        print(f"\r  {msg:<44}", end="", file=sys.stderr, flush=True)

    t0 = time.time()
    path, _ = moss_client.generate(prompt, w, h, seed=a.seed, out=stem + ".png",
                                   steps=moss_client.QUALITY_STEPS[a.quality], on_progress=progress)
    print(file=sys.stderr)
    boxes = draw_boxes(path, spec["items"], stem + "_boxes.jpg")
    print(f"{a.quality}: {path} ({w}x{h}, {len(spec['items'])} boxes) in {time.time() - t0:.0f}s")
    print(f"box overlay: {boxes}")


if __name__ == "__main__":
    try:
        main()
    except moss_client.MossError as e:
        sys.exit(f"error: {e}")
