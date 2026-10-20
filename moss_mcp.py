#!/usr/bin/env python3
"""MOSS-Image MCP server: local INT4 text-to-image, layout (bounding boxes),
single-object cutouts, image-to-image and inpainting for any MCP-capable app.

Runs over stdio. On first use it launches ./start_server.sh (sd-server with the models kept
loaded) unless a MOSS backend is already listening on MOSS_PORT; several MCP clients share one
backend. A backend this process started is stopped when it exits (MOSS_KEEP_SERVER=1 keeps it).
"""
import asyncio
import atexit
import functools
import fcntl
import io
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from PIL import Image as PILImage, ImageDraw, ImageFilter, UnidentifiedImageError

from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import moss_client  # noqa: E402
from moss_client import BackendUnavailable, MossError  # noqa: E402

OUT_DIR = Path(os.path.expanduser(os.environ.get("MOSS_OUTPUT_DIR", str(ROOT / "outputs"))))
TMP_DIR = OUT_DIR / ".tmp"
LOG_DIR = ROOT / "logs"
REQUIRED_FILES = [
    "bin/sd-server",
    "models/diffusion_models/MOSS-Image-SFT-Q4_0.gguf",
    "models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf",
    "models/vae/flux2_ae_small.safetensors",
    "models/tokenizers/tokenizer_qwen3.5.json",
    "models/loras/moss_4step_f16.safetensors",
    "models/loras/moss_3step_f16.safetensors",
]

# Canvas presets per aspect ratio (multiples of 16). Standard-quality times on a base M4:
# "fast" ~1300 latent tokens (~30-40 s), "balanced" ~1550 (~35-50 s), "large" ~2300 (~50-65 s).
SIZES = {
    "fast": {
        "1:1": (576, 576), "16:9": (768, 432), "9:16": (432, 768), "4:3": (672, 496),
        "3:4": (496, 672), "3:2": (704, 464), "2:3": (464, 704), "21:9": (832, 352),
    },
    "balanced": {
        "1:1": (624, 624), "16:9": (832, 464), "9:16": (464, 832), "4:3": (720, 544),
        "3:4": (544, 720), "3:2": (768, 512), "2:3": (512, 768), "21:9": (896, 384),
    },
    "large": {
        "1:1": (768, 768), "16:9": (1024, 576), "9:16": (576, 1024), "4:3": (880, 656),
        "3:4": (656, 880), "3:2": (944, 624), "2:3": (624, 944), "21:9": (1120, 480),
    },
}
Aspect = Literal["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"]
Size = Literal["fast", "balanced", "large"]
Quality = Literal["draft", "standard", "best", "high"]

_server_proc = None
_start_lock = threading.Lock()


def _stop_own_server():
    if _server_proc and _server_proc.poll() is None and not os.environ.get("MOSS_KEEP_SERVER"):
        try:
            os.killpg(_server_proc.pid, signal.SIGTERM)
        except OSError:
            pass


atexit.register(_stop_own_server)


def _ensure_server():
    """Make sure a MOSS backend is reachable on MOSS_PORT, starting one if needed.

    A file lock serialises the check-and-start across processes, so two apps launching at
    once never load two copies of the models (which would exhaust GPU memory).
    """
    global _server_proc
    if moss_client.server_identity():
        return
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
    if missing:
        raise MossError(f"model files missing ({', '.join(missing)}). Run {ROOT / 'setup.sh'} first.")
    LOG_DIR.mkdir(exist_ok=True)
    with _start_lock, open(LOG_DIR / ".server.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if moss_client.server_identity():
            return
        if _server_proc is None or _server_proc.poll() is not None:
            log = open(LOG_DIR / "sd-server.log", "ab")
            env = {**os.environ, "MOSS_PORT": moss_client.SERVER.rsplit(":", 1)[-1].strip("/")}
            _server_proc = subprocess.Popen([str(ROOT / "start_server.sh")], cwd=ROOT, env=env,
                                            stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.time() + 120
        while time.time() < deadline:
            if moss_client.server_identity():
                return
            if _server_proc.poll() is not None:
                raise BackendUnavailable(f"backend exited during start-up; see {LOG_DIR / 'sd-server.log'}")
            time.sleep(0.5)
    raise BackendUnavailable(f"backend did not start within 120 s; see {LOG_DIR / 'sd-server.log'}")


def _new_path(tag):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(OUT_DIR / f"moss_{tag}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png")


def _result(path, secs, extra=""):
    img = PILImage.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)  # inline preview; the full PNG stays on disk
    text = f"Saved {path} ({img.width}x{img.height}) in {secs:.1f}s.{(' ' + extra) if extra else ''}"
    return [Image(data=buf.getvalue(), format="jpeg"), text]


def _resolve_size(aspect_ratio, size, width, height):
    if width or height:
        if not (width and height):
            raise MossError("give both width and height, or neither (to use aspect_ratio + size)")
        return moss_client.validate_size(width, height)
    return SIZES[size][aspect_ratio]


def _steps(quality):
    """draft = 3-step turbo, standard/best = 4-step turbo, high = official 50 steps at CFG 3."""
    return moss_client.QUALITY_STEPS[quality]


async def _run(fn, *args, ctx: Context | None = None, **kw):
    """Run a blocking generation in a worker thread.

    - reports queue position / sampling step to the client every few seconds (clients that reset
      their timeout on progress stay alive through 10-30 min "high" renders);
    - cancels the backend job if the client cancels the call or disconnects;
    - if the shared backend went away mid-job (e.g. the app that started it quit), restarts it
      and retries once.
    """
    for attempt in (1, 2):
        await asyncio.to_thread(_ensure_server)
        state = {"step": 0, "total": None, "msg": "starting"}
        cancel = threading.Event()

        def on_progress(step, total, msg):
            state.update(step=step, total=total, msg=msg)

        task = asyncio.ensure_future(asyncio.to_thread(fn, *args, on_progress=on_progress,
                                                       cancel_event=cancel, **kw))
        t0 = time.time()
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=3)
                if done:
                    return task.result()
                if ctx is not None:
                    try:
                        await ctx.report_progress(state["step"], state["total"],
                                                  f"{state['msg']} ({int(time.time() - t0)} s)")
                    except Exception:
                        pass
        except BackendUnavailable:
            if attempt == 2:
                raise
        except asyncio.CancelledError:
            cancel.set()
            raise
        finally:
            cancel.set() if not task.done() else None


def _open_image(path, what="image"):
    p = Path(os.path.expanduser(str(path)))
    if not p.is_file():
        raise MossError(f"{what} not found: {p}")
    try:
        return PILImage.open(p)
    except (UnidentifiedImageError, OSError) as e:
        raise MossError(f"cannot read {what} {p}: {e} (use PNG, JPEG or WebP)") from None


mcp = MCPServer(
    "moss-image",
    instructions=(
        "Local MOSS-Image-SFT (5B, INT4) image generator on Apple Silicon. Strong at legible text in "
        "images (English and Chinese), posters, product shots, isolated single-object RGBA cutouts, and "
        "photoreal scenes. Each call takes ~30-60 s at the default quality='standard'; quality='high' "
        "(official 50-step settings, finest detail) takes ~10-30 min, so only use it when the user asks for "
        "maximum quality and can wait. Use generate_image for prompts, generate_layout_image "
        "to place text/objects in bounding boxes, generate_cutout_object for single isolated objects on "
        "a plain background with clean transparent RGBA extraction, edit_image to inpaint a region (mask or "
        "boxes), transform_image for img2img. Put any text that must appear in the image in quotes inside the prompt."
    ),
)


def _tool(fn):
    """Register an MCP tool whose anticipated failures reach the user as readable messages.

    MCP SDK 2.x hides the text of unexpected exceptions ("Error executing tool <name>"), so
    MossError and bad-input errors are re-raised as ToolError, which is shown verbatim.
    """
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except MossError as e:
            raise ToolError(str(e)) from None
        except (ValueError, KeyError) as e:
            raise ToolError(f"invalid input: {e}") from None
    return mcp.tool()(wrapper)


@_tool
async def generate_image(prompt: str, aspect_ratio: Aspect = "16:9", size: Size = "fast",
                         quality: Quality = "standard", width: int = 0, height: int = 0,
                         seed: int = -1, upscale_2x: bool = False, ctx: Context = None):
    """Generate an image from a text prompt.

    aspect_ratio + size pick a canvas (fast ~30-35 s, balanced ~35-45 s, large ~45-65 s on M4).
    width/height (multiples of 16, max ~1024) override the preset.
    quality: "standard" (default, 4-step turbo, ~30-60 s), "draft" (3-step, ~25% faster),
    "high" (official 50 steps at CFG 3: finest detail/texture, ~10-15 min at fast sizes and
    ~25-30 min at large sizes on a base M4; same memory). "best" = "standard".
    seed -1 = random. upscale_2x does a quick Lanczos 2x resize.
    """
    w, h = _resolve_size(aspect_ratio, size, width, height)
    out = _new_path("gen")
    path, secs = await _run(moss_client.generate, prompt, w, h, seed, out, _steps(quality), ctx=ctx)
    if upscale_2x:
        _, up = moss_client.upscale(path)
        secs += up
    return _result(path, secs)


@_tool
async def generate_layout_image(caption: str, items: list[dict], aspect_ratio: Aspect = "16:9",
                                size: Size = "fast", quality: Quality = "standard",
                                width: int = 0, height: int = 0, seed: int = -1,
                                ctx: Context = None):
    """Generate an image with elements placed in bounding boxes (MOSS layout control).

    caption: overall scene / style description.
    items: list of {"label": "text" | "object" | ..., "caption": what goes in the box
           (for text put the exact words in quotes), "box": [x0, y0, x1, y1]} where the box is in
           pixels of the output canvas with (0,0) at the top-left. Use the same canvas you request;
           the result text reports the canvas size.
    quality: "standard" (default), "draft" or "high" (50-step reference, ~10-30 min).
    """
    w, h = _resolve_size(aspect_ratio, size, width, height)
    if not items:
        raise MossError("items must contain at least one {label, caption, box}")
    prompt = moss_client.layout_prompt(caption, items, w, h)
    out = _new_path("layout")
    path, secs = await _run(moss_client.generate, prompt, w, h, seed, out, _steps(quality), ctx=ctx)
    return _result(path, secs, f"Layout canvas {w}x{h}.")


@_tool
async def generate_cutout_object(object_prompt: str, bg_color: str = "magenta",
                                 transparent: bool = True, use_layout_box: bool = True,
                                 aspect_ratio: Aspect = "1:1", size: Size = "balanced",
                                 quality: Quality = "standard", width: int = 0, height: int = 0,
                                 seed: int = -1, tolerance: int = 28,
                                 upscale_2x: bool = False, ctx: Context = None):
    """Generate a single isolated object on a plain solid background and extract a clean RGBA cutout.

    Uses MOSS layout bounding-box isolation (10% outer border margin) + flat studio lighting
    directives to prevent edge clipping and floor shadows, followed by exterior + enclosed-hole
    extraction, continuous alpha matting, and luminance-preserving chroma spill suppression.
    bg_color: "magenta" (default chroma key) | "green" | "white" | "gray" | "black" | "blue" | "#RRGGBB".
    transparent: when True (default), also saves a transparent RGBA PNG (<stem>_rgba.png) with
    interior apertures (e.g. rings/handles) removed and zero color spill on edges.
    """
    w, h = _resolve_size(aspect_ratio, size, width, height)
    out = _new_path("cutout")
    plain_path, rgba_path, secs = await _run(
        moss_client.generate_cutout,
        object_prompt,
        w,
        h,
        seed,
        out,
        _steps(quality),
        bg_color,
        transparent,
        use_layout_box,
        tolerance,
        ctx=ctx,
    )
    if upscale_2x:
        _, up = moss_client.upscale(plain_path)
        secs += up
        if rgba_path:
            _, up_rgba = moss_client.upscale(rgba_path)
            secs += up_rgba
    extra = f"Transparent RGBA cutout: {rgba_path}" if rgba_path else f"Solid {bg_color} background only."
    return _result(plain_path, secs, extra)


def _prep_init(image_path, max_side=1024):
    """Load the source image, cap it to the model's comfortable size, snap to /16."""
    img = _open_image(image_path).convert("RGB")
    max_side = max(256, min(int(max_side), 1024))
    scale = min(1.0, max_side / max(img.size))
    w = moss_client._snap16(img.width * scale)
    h = moss_client._snap16(img.height * scale)
    w, h = moss_client.validate_size(w, h)
    if (w, h) != img.size:
        img = img.resize((w, h), PILImage.LANCZOS)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    p = TMP_DIR / f"init_{time.time_ns()}.png"
    img.save(p)
    return str(p), img.size, scale


@_tool
async def edit_image(image_path: str, prompt: str, mask_path: str = "",
                     mask_boxes: list[list[int]] | None = None, strength: float = 1.0,
                     quality: Quality = "standard", seed: int = -1, max_side: int = 832,
                     ctx: Context = None):
    """Inpaint: repaint only part of an existing image.

    Give either mask_path (white = repaint, black = keep; any size, it is resized) or
    mask_boxes [[x0, y0, x1, y1], ...] in the ORIGINAL image's pixels. prompt should describe the
    whole desired image, including what should appear in the masked area. strength 1.0 fully
    repaints the masked area; 0.5-0.8 keeps some of the original there. The image is processed at
    up to max_side px on its longest edge (larger = slower).
    """
    if not mask_path and not mask_boxes:
        raise MossError("provide mask_path or mask_boxes")
    for b in mask_boxes or []:
        if len(b) != 4:
            raise MossError(f"mask box {b} must be [x0, y0, x1, y1]")
    init, (w, h), scale = _prep_init(image_path, max_side)
    if mask_path:
        mask = _open_image(mask_path, "mask").convert("L").resize((w, h), PILImage.NEAREST)
    else:
        mask = PILImage.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        for b in mask_boxes:
            x0, y0, x1, y1 = (v * scale for v in b)
            d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=255)
    if mask.getbbox() is None:
        raise MossError("the mask is empty (nothing to repaint); white marks the area to change")
    mask = mask.filter(ImageFilter.GaussianBlur(4))  # soften seams
    mpath = TMP_DIR / f"mask_{time.time_ns()}.png"
    mask.save(mpath)
    out = _new_path("edit")
    try:
        path, secs = await _run(moss_client.generate, prompt, w, h, seed, out, _steps(quality),
                                init, str(mpath), strength, ctx=ctx)
    finally:
        for p in (init, mpath):
            Path(p).unlink(missing_ok=True)
    return _result(path, secs)


@_tool
async def transform_image(image_path: str, prompt: str, strength: float = 0.6,
                          quality: Quality = "standard", seed: int = -1, max_side: int = 832,
                          ctx: Context = None):
    """Image-to-image: re-render a whole image guided by a prompt (style change, variation, cleanup).

    strength 0.3 = subtle, 0.6 = moderate, 0.85+ = mostly new image keeping rough composition.
    """
    init, (w, h), _ = _prep_init(image_path, max_side)
    out = _new_path("i2i")
    try:
        path, secs = await _run(moss_client.generate, prompt, w, h, seed, out, _steps(quality),
                                init, None, strength, ctx=ctx)
    finally:
        Path(init).unlink(missing_ok=True)
    return _result(path, secs)


@_tool
async def image_server_status() -> str:
    """Check (and start if needed) the local MOSS-Image backend."""
    t0 = time.time()
    await asyncio.to_thread(_ensure_server)
    return f"MOSS-Image backend up at {moss_client.SERVER} (checked in {time.time() - t0:.1f}s). Outputs: {OUT_DIR}"


if __name__ == "__main__":
    mcp.run("stdio")
