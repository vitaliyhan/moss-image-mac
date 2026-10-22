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
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import moss_client
from moss_client import (MossError, QUALITY_STEPS, validate_size)

UI_PORT = int(os.environ.get("MOSS_UI_PORT", "7877"))
OUTPUTS = os.path.join(ROOT, "outputs")
UPLOADS = os.path.join(OUTPUTS, ".uploads")  # uploaded init/mask images; hidden from the gallery
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
    return {"backend": _backend, "jobs": jobs, "gallery": _scan_gallery()}


# ------------------------------------------------------------------ backend management
def _try_capabilities():
    """True if a MOSS backend is up. Updates _backend on failure."""
    try:
        moss_client.server_identity()
        return True
    except MossError as e:
        with _lock:
            _backend.update(state="error", detail=str(e))
        return False


def _ensure_backend():
    """Make sure the MOSS backend is reachable, starting start_server.sh if it is not."""
    if _backend["state"] == "up" and _try_capabilities():
        return True
    if _backend["state"] == "error":
        return False
    global _backend_proc
    with _lock:
        _backend.update(state="starting", detail="launching start_server.sh")
    if _backend_proc is None or _backend_proc.poll() is not None:
        log = open(os.path.join(ROOT, "logs", "ui_backend.log"), "ab")
        _backend_proc = subprocess.Popen(
            [os.path.join(ROOT, "start_server.sh")], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + BACKEND_START_TIMEOUT
    while time.time() < deadline:
        if _try_capabilities():
            with _lock:
                _backend.update(state="up", detail="")
            return True
        if _backend["state"] == "error":
            return False
        if _backend_proc.poll() is not None:
            with _lock:
                _backend.update(state="error",
                                detail="start_server.sh exited (see logs/ui_backend.log)")
            return False
        with _lock:
            _backend["detail"] = "loading models"
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
    if not _ensure_backend():
        with _lock:
            job.update(state="failed", finished=time.time(),
                       error=_backend.get("detail") or "backend unavailable")
        _cancel_events.pop(job["id"], None)
        return

    p = job["params"]
    out = os.path.join(OUTPUTS, f"moss_{time.strftime('%Y%m%d_%H%M%S')}_{job['id']}.png")

    def on_progress(step, total, message):
        with _lock:
            job.update(step=step, total=total, message=message)

    def run_generate():
        init_path = _upload_path(p["init"]) if p.get("init") else None
        mask_path = _upload_path(p["mask"]) if p.get("mask") else None
        if p.get("init") and not init_path:
            raise MossError("uploaded image is gone - pick the file again")
        return moss_client.generate(
            p["prompt"], width=p["width"], height=p["height"], seed=p["seed"],
            out=out, steps=p["steps"], strength=p["strength"],
            init_image=init_path, mask_image=mask_path,
            on_progress=on_progress, cancel_event=ev, preview=True)

    try:
        try:
            path, secs = run_generate()
        except moss_client.BackendUnavailable:
            # The backend may still be binding its port when the first job lands; re-ensure once.
            with _lock:
                _backend.update(state="down", detail="")
            time.sleep(2)
            if not _ensure_backend():
                raise
            path, secs = run_generate()
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
    quality = payload.get("quality", "standard")
    if quality not in QUALITY_STEPS:
        return None, f"quality must be one of {sorted(QUALITY_STEPS)}"
    try:
        seed = int(payload.get("seed", -1))
        batch = max(1, min(int(payload.get("batch", 1)), 8))
        strength = float(payload.get("strength", 0.75))
    except (TypeError, ValueError):
        return None, "seed, batch or strength is not a number"
    if not 0.0 < strength <= 1.0:
        return None, "strength must be in (0, 1]"
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
    mode = "inpaint" if (init_name and mask_name) else ("restyle" if init_name else "txt2img")
    with _lock:
        for _ in range(batch):
            job = {
                "id": _next_id[0], "state": "queued", "prompt": prompt[:200],
                "mode": mode,
                "params": {"prompt": prompt, "width": width, "height": height,
                           "seed": seed, "steps": QUALITY_STEPS[quality], "strength": strength,
                           "upscale": bool(payload.get("upscale")),
                           "init": init_name, "mask": mask_name},
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
