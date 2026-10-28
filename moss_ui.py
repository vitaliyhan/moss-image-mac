#!/usr/bin/env python3
"""Web UI for the local MOSS-Image generator: prompt, queue, live progress, gallery.

    ./start_ui.sh            # starts the MOSS backend if needed, then the UI
    open http://127.0.0.1:7877

The UI keeps its own job queue (one render at a time - the GPU is shared with nothing
else) and talks to the MOSS backend through moss_client, so the backend, CLI and MCP
server all stay interoperable. State is in memory only: restarting the UI clears the
queue but the finished images stay in outputs/.
"""
import base64
import glob
import json
import os
import random
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import moss_client
import qwen_mlx
from moss_client import (MossError, QUALITY_STEPS, validate_size)

# Qwen MLX base-model quality ladder (turbo is fixed at 8 steps); "high" = custom.
MLX_STEPS = {"fast": 10, "draft": 20, "standard": 40}


def _ref_aspect_size(width, height, ref_paths):
    """Fit width/height to the first reference's aspect ratio at the same pixel area.

    Snapped to /32 (Qwen-Image-2.1 requires it) and clamped to the UI's 256-1536
    range; returns the inputs unchanged when there is no readable reference."""
    if not ref_paths:
        return width, height
    try:
        from PIL import Image
        with Image.open(ref_paths[0]) as im:
            ratio = im.width / max(1, im.height)
    except Exception:
        return width, height
    if not 0.2 < ratio < 5.0:  # degenerate ratio would push one side out of bounds
        return width, height
    area = width * height
    w = max(256, min(1536, int(round((area * ratio) ** 0.5 / 32)) * 32))
    h = max(256, min(1536, int(round((area / ratio) ** 0.5 / 32)) * 32))
    return w, h

UI_PORT = int(os.environ.get("MOSS_UI_PORT", "7877"))
OUTPUTS = os.path.join(ROOT, "outputs")
UPLOADS = os.path.join(OUTPUTS, ".uploads")  # uploaded init/mask images; hidden from the gallery
METAS = os.path.join(OUTPUTS, ".meta")       # per-image generation metadata (<name>.png.json)
BACKEND_START_TIMEOUT = 180  # seconds for the ~15 s model load on a cold start

# Largest canvas we accept for an uploaded image (matches the model's limits: the turbo
# LoRA is trained at <=1024^2 and bigger canvases exhaust GPU memory).
MAX_UPLOAD_SIDE = 1024

# Gallery shows outputs/ by default; list more folders in MOSS_EXTRA_GALLERY_DIRS
# (colon-separated) to browse images from elsewhere. Only files under these roots can be
# served; delete/upscale apply to outputs/ files only.
GALLERY_DIRS = [OUTPUTS]
GALLERY_DIRS += [os.path.abspath(p) for p in
                 os.environ.get("MOSS_EXTRA_GALLERY_DIRS", "").split(":") if p]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")

UI_HTML_PATH = os.path.join(ROOT, "ui", "index.html")

_lock = threading.Lock()
_jobs = []            # newest last; dicts: id, prompt, params, state, step, total, message,
                      # path, error, enqueued, started, finished
_next_id = [1]
_cancel_events = {}   # job id -> threading.Event, for the job being rendered
_backend = {"state": "down", "detail": ""}  # down | starting | up | error
_backend_proc = None
_backend_mode = None  # which server config is running: "moss" | "qwen"


def _gguf_files():
    """Downloaded Qwen-Image-2.1 GGUF files (base names), newest first by mtime."""
    pat = os.path.join(ROOT, "models", "diffusion_models", "qwen-image-2.1*.gguf")
    files = [(os.path.basename(p), os.path.getmtime(p)) for p in glob.glob(pat)]
    files.sort(key=lambda t: -t[1])
    return [n for n, _ in files]


def _qwen_gguf_path(variant, quant):
    stem = f"qwen-image-2.1-UC-{quant}" if variant == "uc" else f"qwen-image-2.1-{quant}"
    return os.path.join(ROOT, "models", "diffusion_models", stem + ".gguf")


def _scan_gallery():
    out = []
    for root in GALLERY_DIRS:
        try:
            names = os.listdir(root)
        except FileNotFoundError:
            continue
        for n in names:
            if not n.lower().endswith(IMG_EXTS):
                continue
            p = os.path.join(root, n)
            try:
                st = os.stat(p)
            except FileNotFoundError:
                continue
            out.append({
                "name": n,
                "mtime": int(st.st_mtime),
                "size": st.st_size,
                # token the browser hands back to fetch the file; also our delete key
                "src": base64.urlsafe_b64encode(p.encode()).decode(),
                "managed": os.path.dirname(p) == OUTPUTS,
            })
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out


def _resolve_src(token):
    """Absolute path for a gallery token, or None if it is not under a gallery root."""
    try:
        path = os.path.realpath(base64.urlsafe_b64decode(token).decode())
    except Exception:
        return None
    for root in GALLERY_DIRS:
        if path == os.path.realpath(root) or path.startswith(os.path.realpath(root) + os.sep):
            return path if os.path.isfile(path) else None
    return None


def _public_state():
    with _lock:
        jobs = [{k: v for k, v in j.items() if k != "cancel"} for j in _jobs]
        backend = dict(_backend)
    return {"backend": backend, "jobs": jobs, "gallery": _scan_gallery(),
            "mlx": {"installed": qwen_mlx.mlx_available(),
                    "models": qwen_mlx.installed_variants()},
            "ggufs": _gguf_files()}


# ------------------------------------------------------------------ generation metadata
def _meta_path(img_path):
    return os.path.join(METAS, os.path.basename(img_path) + ".json")


def _write_meta(img_path, meta):
    try:
        os.makedirs(METAS, exist_ok=True)
        with open(_meta_path(img_path), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
    except OSError:
        pass  # metadata is a nice-to-have; never fail a finished render over it


def _read_meta(img_path):
    try:
        with open(_meta_path(img_path)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ backend management
def _try_capabilities():
    """True if a MOSS backend is up. Updates _backend on a foreign/port-conflict error.

    A refused connection (backend not bound yet, models still loading) is NOT an error:
    the caller keeps polling until the start timeout.
    """
    try:
        caps = moss_client.server_identity()
    except moss_client.BackendUnavailable:
        return False
    except MossError as e:
        with _lock:
            _backend.update(state="error", detail=str(e))
        return False
    if not caps:  # nothing is listening on the backend port
        return False
    name = (caps.get("model") or {}).get("name") or ""
    with _lock:
        _backend["model"] = name
        global _backend_mode
        _backend_mode = "qwen" if "qwen" in name.lower() else "moss"
    return True


def _ensure_backend(mode="moss", params=None):
    """Make sure the sd-server is reachable AND running the model this job needs.

    If the server is up with the other model (MOSS vs Qwen-Image-2.1 GGUF), it is stopped
    and restarted with the right one - the reload takes a minute or two, the job just waits.
    """
    global _backend_mode, _backend_proc
    if _backend["state"] == "up" and _try_capabilities():
        if _backend_mode == mode:
            return True
        with _lock:
            _backend.update(state="down", detail="switching model")
        _dbg(f"server runs {_backend_mode}, job needs {mode} - restarting")
        _stop_backend()
    if _backend["state"] == "error":
        return False
    env = dict(os.environ)
    if mode == "qwen":
        env["MOSS_SERVER_MODEL"] = "qwen2.1"
        env["QWEN_VARIANT"] = (params or {}).get("variant") or "uc"
        env["QWEN_QUANT"] = (params or {}).get("quant") or "Q4_K_M"
        if not os.path.isfile(_qwen_gguf_path(env["QWEN_VARIANT"], env["QWEN_QUANT"])):
            raise MossError(f"GGUF for {env['QWEN_VARIANT']}/{env['QWEN_QUANT']} is not downloaded - "
                            f"run: ./setup_qwen.sh gguf-{'uc' if env['QWEN_VARIANT']=='uc' else 'base'}-"
                            f"{env['QWEN_QUANT'].lower()} gguf-enc gguf-vae gguf-tok")
    with _lock:
        _backend.update(state="starting", detail="launching start_server.sh")
    for _spawn in range(3):  # retries handle an old server with the other model still up
        if _backend_proc is None or _backend_proc.poll() is not None:
            log = open(os.path.join(ROOT, "logs", "ui_backend.log"), "ab")
            _backend_proc = subprocess.Popen(
                [os.path.join(ROOT, "start_server.sh")], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT)
            _dbg(f"spawned start_server.sh ({mode})")
        deadline = time.time() + BACKEND_START_TIMEOUT
        while time.time() < deadline:
            if _try_capabilities():
                if _backend_mode == mode:
                    with _lock:
                        _backend.update(state="up", detail="")
                    _dbg(f"capabilities ok -> backend up ({mode})")
                    return True
                # a still-running server with the OTHER model answered; stop it and respawn
                _dbg(f"old {_backend_mode} server still up - stopping it")
                with _lock:
                    _backend.update(state="down", detail="")
                _stop_backend()
                break
            if _backend["state"] == "error":
                return False
            if _backend_proc.poll() is not None:
                # start_server.sh exited - either a crash or "already running" (old server)
                if _try_capabilities() and _backend_mode != mode:
                    with _lock:
                        _backend.update(state="down", detail="")
                    _stop_backend()
                    break
                with _lock:
                    _backend.update(state="error",
                                    detail="start_server.sh exited (see logs/ui_backend.log)")
                return False
            with _lock:
                _backend["detail"] = f"loading {'Qwen-Image-2.1' if mode == 'qwen' else 'MOSS'} model"
            time.sleep(1)
    with _lock:
        _backend.update(state="error", detail=f"backend did not come up in {BACKEND_START_TIMEOUT}s")
    return False


# ------------------------------------------------------------------ uploads (edit tab)
def _save_dataurl(dataurl):
    """Decode a data:image/...;base64 URL, fit it to model limits, save under UPLOADS.

    Returns (name, width, height). Raises ValueError on bad input.
    """
    import base64 as b64
    import io
    import re
    import uuid

    from PIL import Image

    m = re.match(r"data:image/(png|jpeg|jpg|webp);base64,(.+)", dataurl or "", re.S)
    if not m:
        raise ValueError("expected a data:image/...;base64 URL")
    img = Image.open(io.BytesIO(b64.b64decode(m.group(2))))
    img.load()
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_UPLOAD_SIDE / max(w, h), (1024 * 1024 + 64 * 1024) / (w * h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    os.makedirs(UPLOADS, exist_ok=True)
    name = f"up_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
    img.save(os.path.join(UPLOADS, name), format="PNG")
    return name, img.width, img.height


def _upload_path(name):
    name = os.path.basename(str(name or ""))
    p = os.path.join(UPLOADS, name)
    return p if name.startswith("up_") and os.path.isfile(p) else None


# ------------------------------------------------------------------ worker
def _dbg(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _worker():
    while True:
        with _lock:
            job = next((j for j in _jobs if j["state"] == "queued"), None)
        if job is None:
            time.sleep(0.3)
            continue
        try:
            _run_job(job)
        except Exception as e:  # never let one job kill the worker
            with _lock:
                job.update(state="failed", finished=time.time(),
                           error=f"internal error: {e!r}", message="failed")


def _run_job(job):
    ev = threading.Event()
    with _lock:
        _cancel_events[job["id"]] = ev
        job.update(state="running", started=time.time(), step=0, total=None,
                   message="waiting for backend")
    p = job["params"]
    if p.get("runtime") == "mlx":
        _run_job_mlx(job, ev)
        return
    mode = "qwen" if p.get("runtime") == "qwen-gguf" else "moss"
    try:
        if not _ensure_backend(mode, p):
            with _lock:
                job.update(state="failed", finished=time.time(),
                           error=_backend.get("detail") or "backend unavailable")
            _cancel_events.pop(job["id"], None)
            return
    except MossError as e:
        with _lock:
            job.update(state="failed", finished=time.time(), error=str(e), message="failed")
        _cancel_events.pop(job["id"], None)
        return

    prefix = "qwen" if p.get("runtime") == "qwen-gguf" else "moss"
    out = os.path.join(OUTPUTS, f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{job['id']}.png")
    seed = p["seed"]
    if seed == -1:  # roll the random seed here so the metadata can record the real one
        seed = random.randrange(2**31)

    def on_progress(step, total, message):
        with _lock:
            job.update(step=step, total=total, message=message)

    def run_generate():
        init_path = _upload_path(p["init"]) if p.get("init") else None
        mask_path = _upload_path(p["mask"]) if p.get("mask") else None
        if p.get("init") and not init_path:
            raise MossError("uploaded image is gone - pick the file again")
        ref_paths = []
        for name in p.get("refs") or []:
            rp = _upload_path(name)
            if not rp:
                raise MossError("uploaded reference image is gone - pick the files again")
            ref_paths.append(rp)
        width, height = p["width"], p["height"]
        if p.get("keep_aspect"):
            width, height = _ref_aspect_size(width, height, ref_paths)
        if p.get("runtime") == "qwen-gguf":
            model_name = _backend.get("model") or ""
            if "qwen" not in model_name.lower():
                raise MossError("the sd-server switched models unexpectedly - submit the job again")
            return moss_client.generate(
                p["prompt"], width=width, height=height, seed=seed,
                out=out, steps=p["steps"], strength=1.0, ref_images=ref_paths,
                txt_cfg=p.get("txt_cfg"), img_cfg=p.get("img_cfg"),
                on_progress=on_progress, cancel_event=ev, preview=True)
        return moss_client.generate(
            p["prompt"], width=p["width"], height=p["height"], seed=seed,
            out=out, steps=p["steps"], strength=p["strength"],
            init_image=init_path, mask_image=mask_path,
            on_progress=on_progress, cancel_event=ev, preview=True)

    try:
        try:
            _dbg("generate attempt 1")
            path, secs = run_generate()
        except moss_client.BackendUnavailable as e:
            # The backend may still be binding its port when the first job lands; re-ensure once.
            _dbg(f"attempt 1 unavailable: {e}")
            with _lock:
                _backend.update(state="down", detail="")
            time.sleep(2)
            if not _ensure_backend(mode, p):
                raise
            _dbg("generate attempt 2")
            path, secs = run_generate()
        _write_meta(out, {"mode": job["mode"], "prompt": p["prompt"],
                          "width": p["width"], "height": p["height"], "seed": seed,
                          "steps": p["steps"], "strength": p["strength"],
                          "upscale": p["upscale"], "secs": round(secs, 1),
                          "runtime": p.get("runtime") or "sdcpp",
                          "refs": len(p.get("refs") or [])})
        if p["upscale"]:
            with _lock:
                job["message"] = "upscaling 2x"
            moss_client.upscale(path)
        with _lock:
            job.update(state="done", finished=time.time(), path=os.path.basename(path),
                       secs=round(secs, 1), message="done")
    except MossError as e:
        with _lock:
            cancelled = ev.is_set()
            job.update(state="cancelled" if cancelled else "failed",
                       finished=time.time(), error=None if cancelled else str(e),
                       message="cancelled" if cancelled else "failed")
    finally:
        _cancel_events.pop(job["id"], None)


def _run_job_mlx(job, ev):
    """Qwen-Image-2.1 via the local mflux MLX runtime - no sd-server involved."""
    p = job["params"]
    out = os.path.join(OUTPUTS, f"qwen_{time.strftime('%Y%m%d_%H%M%S')}_{job['id']}.png")
    seed = p["seed"]
    if seed == -1:
        seed = random.randrange(2**31)

    def on_progress(step, total, message):
        with _lock:
            job.update(step=step, total=total, message=message)

    try:
        ref_paths = []
        for name in p.get("refs") or []:
            rp = _upload_path(name)
            if not rp:
                raise MossError("uploaded reference image is gone - pick the files again")
            ref_paths.append(rp)
        width, height = p["width"], p["height"]
        if p.get("keep_aspect"):
            width, height = _ref_aspect_size(width, height, ref_paths)
            if (width, height) != (p["width"], p["height"]):
                with _lock:
                    job["message"] = f"matching reference aspect ({width}x{height})"
        with _lock:
            job["message"] = "starting MLX render"
        turbo = p.get("variant") == "turbo"
        path, secs = qwen_mlx.generate(
            p["prompt"], out=out, model_dir=p.get("qmodel"),
            turbo=turbo, quant=int(p.get("quant") or 4),
            width=width, height=height, seed=seed,
            steps=p["steps"], guidance=p.get("guidance"),
            ref_images=ref_paths, step_cache=bool(p.get("step_cache")),
            small_text_encoder=bool(p.get("small_encoder")),
            loras=p.get("loras") or (),
            on_progress=on_progress, cancel_event=ev)
        _write_meta(out, {"mode": job["mode"], "prompt": p["prompt"],
                          "width": width, "height": height, "seed": seed,
                          "steps": p["steps"], "strength": None,
                          "upscale": p["upscale"], "secs": round(secs, 1),
                          "runtime": "mlx", "model": p.get("qmodel"),
                          "small_encoder": bool(p.get("small_encoder")),
                          "loras": [os.path.basename(x) for x in (p.get("loras") or ())],
                          "refs": len(ref_paths)})
        if p["upscale"]:
            with _lock:
                job["message"] = "upscaling 2x"
            moss_client.upscale(path)
        with _lock:
            job.update(state="done", finished=time.time(), path=os.path.basename(path),
                       secs=round(secs, 1), message="done")
    except MossError as e:
        with _lock:
            cancelled = ev.is_set()
            job.update(state="cancelled" if cancelled else "failed",
                       finished=time.time(), error=None if cancelled else str(e),
                       message="cancelled" if cancelled else "failed")
    finally:
        _cancel_events.pop(job["id"], None)


def _enqueue(payload):
    """Validate a generate request and queue it. Returns (job_dict_or_None, error_str)."""
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return None, "prompt is empty"
    try:
        width, height = validate_size(payload.get("width", 1024), payload.get("height", 1024))
    except MossError as e:
        return None, str(e)
    runtime = payload.get("runtime", "sdcpp")
    if runtime not in ("sdcpp", "mlx", "qwen-gguf"):
        return None, f"runtime must be sdcpp, mlx or qwen-gguf (got {runtime!r})"
    qwen = runtime in ("mlx", "qwen-gguf")
    try:
        seed = int(payload.get("seed", -1))
        batch = max(1, min(int(payload.get("batch", 1)), 8))
        strength = float(payload.get("strength", 0.75))
    except (TypeError, ValueError):
        return None, "seed, batch or strength is not a number"
    if not 0.0 < strength <= 1.0:
        return None, "strength must be in (0, 1]"

    # reference images (Qwen 2.1 tab): 0-8 uploads or data URLs -> saved names
    raw_refs = payload.get("refs") or []
    if not isinstance(raw_refs, list) or len(raw_refs) > 8:
        return None, "refs must be a list of at most 8 images"
    refs = []
    try:
        for r in raw_refs:
            if isinstance(r, str) and r.startswith("data:"):
                r, _, _ = _save_dataurl(r)
            refs.append(str(r))
    except ValueError as e:
        return None, f"bad reference image: {e}"

    qmodel = variant = quant = txt_cfg = img_cfg = guidance = None
    step_cache = bool(payload.get("step_cache"))
    small_encoder = bool(payload.get("small_encoder"))
    keep_aspect = bool(payload.get("keep_aspect"))
    if qwen:
        # Qwen quality ladder: turbo runs its fixed 8-step schedule; the base model
        # uses draft 20 / standard 40 / custom steps.
        if runtime == "mlx":
            variant = payload.get("variant", "base")
            if variant not in ("base", "turbo"):
                return None, "variant must be base or turbo"
            try:
                quant = int(payload.get("quant", 4))
            except (TypeError, ValueError):
                return None, "quant is not a number"
            if quant not in (4, 6, 8):
                return None, "quant must be 4, 6 or 8"
            qmodel = payload.get("qmodel") or (f"mlx-turbo-q{quant}" if variant == "turbo"
                                               else f"mlx-q{quant}")
            if qmodel not in qwen_mlx.VARIANTS:
                return None, f"unknown MLX model {qmodel!r} - run ./setup_qwen.sh {qmodel}"
            if variant == "turbo":
                steps = qwen_mlx.TURBO_STEPS
            else:
                steps = qwen_mlx.BASE_STEPS
        else:  # qwen-gguf
            variant = payload.get("variant", "uc")
            if variant not in ("uc", "base"):
                return None, "variant must be uc or base"
            try:
                quant = str(payload.get("quant", "Q4_K_M"))
            except (TypeError, ValueError):
                return None, "quant is not a string"
            steps = 40
        quality = payload.get("quality", "fast" if runtime == "mlx" else "standard")
        lora_dmd2 = runtime == "mlx" and bool(payload.get("lora_dmd2"))
        if runtime == "mlx":
            if lora_dmd2:
                # DMD2 is trained on the base model without CFG; 3 steps give a
                # finished image (2 is the trained minimum, 3 refines it).
                if variant == "turbo":
                    return None, "the DMD2 LoRA works with the base model, not turbo"
                if not os.path.isfile(qwen_mlx.DMD2_LORA):
                    return None, "DMD2 LoRA is not downloaded - run: ./setup_qwen.sh lora-dmd2"
                steps = 3
            elif quality not in MLX_STEPS and quality != "high":
                return None, f"quality must be one of {sorted(MLX_STEPS)} or high"
            elif variant != "turbo":
                steps = MLX_STEPS.get(quality, steps)
        elif quality not in QUALITY_STEPS:
            return None, f"quality must be one of {sorted(QUALITY_STEPS)}"
        if quality == "high" and not (runtime == "mlx" and variant == "turbo"):
            try:
                steps = int(payload.get("steps", steps))
            except (TypeError, ValueError):
                return None, "steps is not a number"
            if not 1 <= steps <= 100:
                return None, "steps must be 1-100"
        try:
            guidance = payload.get("guidance", None)
            guidance = float(guidance) if guidance not in (None, "") else None
            txt_cfg = float(payload.get("txt_cfg", 2.5))
            img_cfg = float(payload.get("img_cfg", 1.0))
        except (TypeError, ValueError):
            return None, "guidance/txt_cfg/img_cfg is not a number"
        if lora_dmd2:
            guidance = 1.0  # DMD2 runs without CFG (teacher guidance is baked in)
    else:
        quality = payload.get("quality", "standard")
        if quality not in QUALITY_STEPS:
            return None, f"quality must be one of {sorted(QUALITY_STEPS)}"
        steps = QUALITY_STEPS[quality]
        if quality == "high":  # turbo presets have fixed LoRA schedules; only high is tunable
            try:
                steps = int(payload.get("steps", steps))
            except (TypeError, ValueError):
                return None, "steps is not a number"
            if not 1 <= steps <= 100:
                return None, "steps must be 1-100"

    jobs = []
    init_name = payload.get("init")
    mask_name = payload.get("mask")
    try:
        if isinstance(mask_name, str) and mask_name.startswith("data:"):
            mask_name, _, _ = _save_dataurl(mask_name)
        if isinstance(init_name, str) and init_name.startswith("data:"):
            init_name, _, _ = _save_dataurl(init_name)
    except ValueError as e:
        return None, f"bad mask/image: {e}"
    mode = payload.get("tag") or ("inpaint" if (init_name and mask_name)
                                  else ("restyle" if init_name else "txt2img"))
    if qwen:
        mode = "qwen-edit" if refs else "qwen"
    with _lock:
        for _ in range(batch):
            job = {
                "id": _next_id[0], "state": "queued", "prompt": prompt[:200],
                "mode": mode,
                "params": {"prompt": prompt, "width": width, "height": height,
                           "seed": seed, "steps": steps, "strength": strength,
                           "upscale": bool(payload.get("upscale")),
                           "init": init_name, "mask": mask_name,
                           "runtime": runtime, "refs": refs, "qmodel": qmodel,
                           "variant": variant, "quant": quant,
                           "txt_cfg": txt_cfg, "img_cfg": img_cfg, "guidance": guidance,
                           "step_cache": step_cache, "small_encoder": small_encoder,
                           "keep_aspect": keep_aspect,
                           "loras": ([qwen_mlx.DMD2_LORA] if lora_dmd2 else [])},
                "step": 0, "total": None, "message": "queued",
                "enqueued": time.time(), "started": None, "finished": None,
            }
            _next_id[0] += 1
            _jobs.append(job)
            jobs.append(job)
        if len(_jobs) > 200:  # keep only the last 200 finished/pending entries in memory
            del _jobs[:len(_jobs) - 200]
    os.makedirs(OUTPUTS, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    return jobs, None


# ------------------------------------------------------------------ http
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype, cache=None):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except (FileNotFoundError, IsADirectoryError):
            self._json({"error": "not found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # Images are immutable content and cache well; the UI page itself must always be
        # re-fetched, or an updated interface looks like "the shortcut opens something old".
        self.send_header("Cache-Control", cache or "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._file(UI_HTML_PATH, "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(_public_state())
        elif path == "/media":
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            fp = _resolve_src((qs.get("src") or [""])[0])
            if not fp:
                self._json({"error": "not found"}, 404)
                return
            ext = os.path.splitext(fp)[1].lower()
            ctype = "image/png" if ext == ".png" else "image/jpeg"
            self._file(fp, ctype, cache="max-age=3600")
        elif path.startswith("/outputs/"):
            name = os.path.basename(path[len("/outputs/"):])
            ctype = "image/png" if name.endswith(".png") else "image/jpeg"
            self._file(os.path.join(OUTPUTS, name), ctype, cache="max-age=3600")
        elif path == "/api/meta":
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            fp = _resolve_src((qs.get("src") or [""])[0])
            meta = _read_meta(fp) if fp and os.path.dirname(fp) == os.path.realpath(OUTPUTS) else None
            if meta is None:
                self._json({"error": "no metadata"}, 404)
            else:
                self._json(meta)
        elif path == "/api/gallery-zip":
            self._json({"error": "not implemented"}, 404)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json({"error": "invalid JSON body"}, 400)
            return
        if path == "/api/upload":
            try:
                name, w, h = _save_dataurl(payload.get("data"))
                self._json({"ok": True, "name": name, "width": w, "height": h})
            except Exception as e:
                self._json({"error": f"bad image: {e}"}, 400)
        elif path == "/api/generate":
            job, err = _enqueue(payload)
            self._json({"error": err} if err else {"ok": True, "job": job})
        elif path == "/api/cancel":
            job_id = payload.get("id")
            with _lock:
                job = next((j for j in _jobs if j["id"] == job_id), None)
                if job and job["state"] == "running":
                    _cancel_events.get(job_id, threading.Event()).set()
                elif job:
                    job.update(state="cancelled", finished=time.time(), message="cancelled")
            self._json({"ok": True})
        elif path == "/api/delete":
            fp = _resolve_src(str(payload.get("src", "")))
            if not fp or os.path.dirname(fp) != os.path.realpath(OUTPUTS):
                self._json({"error": "only files in outputs/ can be deleted"}, 400)
                return
            try:
                os.remove(fp)
                try:
                    os.remove(_meta_path(fp))
                except FileNotFoundError:
                    pass
                self._json({"ok": True})
            except FileNotFoundError:
                self._json({"error": "not found"}, 404)
        elif path == "/api/stop-backend":
            _stop_backend()
            self._json({"ok": True})
        elif path == "/api/shutdown":
            _stop_backend()
            self._json({"ok": True})
            threading.Timer(0.3, os._exit, args=(0,)).start()  # let the reply flush first
        elif path == "/api/upscale":
            fp = _resolve_src(str(payload.get("src", "")))
            if not fp:
                self._json({"error": "not found"}, 404)
                return
            stem, ext = os.path.splitext(fp)
            out, _ = moss_client.upscale(fp, f"{stem}_2x{ext}")
            self._json({"ok": True, "name": os.path.basename(out)})
        else:
            self._json({"error": "not found"}, 404)


# ------------------------------------------------------------------ stop controls
def _stop_backend():
    """Stop the MOSS backend (ours or one started by start_server.sh / the MCP server)."""
    global _backend_proc
    if _backend_proc is not None and _backend_proc.poll() is None:
        _backend_proc.terminate()
    else:  # backend was started outside this UI process
        subprocess.run(["pkill", "-f", "bin/sd-server"], capture_output=True)
    _backend_proc = None
    with _lock:
        _backend.update(state="down", detail="")


def main():
    threading.Thread(target=_worker, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", UI_PORT), Handler)
    print(f"MOSS Image UI:  http://127.0.0.1:{UI_PORT}   (backend: {moss_client.SERVER})")

    def _serve():
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass

    threading.Thread(target=_serve, daemon=True).start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
