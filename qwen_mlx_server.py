#!/usr/bin/env python3
"""Persistent MLX worker for Qwen-Image-2.1 - mflux loaded once, reused per render.

The mflux CLI reloads ~4 GB of weights, re-quantizes and re-traces mx.compile for
every image. This worker keeps the model resident: the first render pays the load,
every later render starts sampling immediately, and the transformer's text/image
prefix K/V caches survive across renders (repeated edits of the same reference
reuse their attention prefixes).

Run under .venv/bin/python (mflux deps live there):

    .venv/bin/python qwen_mlx_server.py --parent-pid $$

All MLX work (model load and renders) runs on one dedicated thread - MLX streams
are thread-local, so mixing threads strands lazy arrays on dead threads' streams.
HTTP handlers only submit tasks and report the shared state.

Serves 127.0.0.1:<ephemeral> and records the port in logs/mlx_worker.json:

    GET  /status   -> {phase, step, total, job_id, loaded, variant, bits, small_text_encoder, error, result}
    POST /load     {model_dir, base_model, bits, small_text_encoder}
    POST /generate {job_id, out, prompt, negative_prompt, steps, seed, width, height,
                    guidance, ref_images, step_cache, step_cache_threshold, strength}
    POST /cancel   {}
    POST /shutdown {}

Generation runs asynchronously: /generate replies immediately and clients poll
/status (progress arrives through an mflux in-loop callback; cancel raises
KeyboardInterrupt at the next step boundary, which mflux turns into its clean
StopImageGenerationException).
"""
import argparse
import gc
import json
import os
import queue
import signal
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(ROOT, "logs", "mlx_worker.json")

# Resolutions above this pixel area tile the VAE decode (512 px tiles) to keep the
# decode from spiking into swap; at or below it tiling only costs time.
VAE_TILE_ABOVE_AREA = 1024 * 1024


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class _GenState:
    """Mutable slot the progress callback reads while a render runs."""

    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "idle"
        self.step = 0
        self.total = None
        self.job_id = None
        self.error = None
        self.cancel = threading.Event()


_state = _GenState()
_model = None
_model_info = {"variant": None, "bits": None, "small_text_encoder": None}
_last_result = {}


class _WorkerCallback:
    """Reports per-step progress and turns /cancel into mflux's stop exception."""

    def call_before_loop(self, seed, prompt, latents, config, **kwargs):
        with _state.lock:
            _state.phase = "sampling"
            _state.step = 0
            _state.total = config.num_inference_steps

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps=None):
        if _state.cancel.is_set():
            raise KeyboardInterrupt  # mflux converts this to StopImageGenerationException
        with _state.lock:
            _state.step = t + 1

    def call_after_loop(self, seed, prompt, latents, config):
        with _state.lock:
            _state.phase = "decoding"


# Phases that mean a render occupies the worker.
BUSY_PHASES = ("encoding", "sampling", "decoding")


def _load_model(model_dir, base_model, bits, small_text_encoder, lora_paths):
    """Load (or reload) the model. Mirrors the mflux edit-CLI's construction."""
    import mlx.core as mx

    from mflux.models.common.compute_precision import ComputePrecision
    from mflux.models.common.resolution.config_resolution import ConfigResolution
    from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

    # Same architecture as qwen-image-2.1 (the turbo checkpoint brings its own
    # schedule); mirrors FAMILY_MODELS in mflux's qwen21 edit CLI.
    model_config = ConfigResolution.resolve_restricted(
        model_dir,
        "qwen-image-2.1",
        model_path=model_dir,
        extra_keys=("qwen-image-2.1-turbo",),
        base_model=base_model,
    )
    model = QwenImage21Edit(
        quantize=bits,
        model_path=model_dir,
        model_config=model_config,
        compute_precision=ComputePrecision.dtype_for(None),
        small_text_encoder=small_text_encoder,
        lora_paths=list(lora_paths) or None,
        lora_scales=[1.0] * len(lora_paths) or None,
    )
    model.callbacks.register(_WorkerCallback())
    gc.collect()
    mx.clear_cache()
    return model


def _generate(params):
    import mlx.core as mx

    from mflux.models.common.vae.tiling_config import TilingConfig
    from mflux.utils.exceptions import StopImageGenerationException

    model = _model
    out = params["out"]
    width = max(32, int(params["width"]) // 32 * 32)
    height = max(32, int(params["height"]) // 32 * 32)
    guidance = float(params.get("guidance") or 1.0)
    refs = [os.path.abspath(p) for p in params.get("ref_images") or []]

    if width * height > VAE_TILE_ABOVE_AREA:
        if getattr(model, "tiling_config", None) is None:
            model.tiling_config = TilingConfig(vae_decode_tile_size=512)
    elif getattr(model, "tiling_config", None) is not None:
        model.tiling_config = None  # back to untiled decode for small renders
    t0 = time.time()
    try:
        image = model.generate_image(
            seed=int(params["seed"]),
            prompt=str(params["prompt"]),
            negative_prompt=str(params.get("negative_prompt") or ""),
            num_inference_steps=int(params["steps"]),
            guidance=guidance,
            width=width,
            height=height,
            image_paths=refs or None,
            use_kv_cache=True,
            use_step_cache=bool(params.get("step_cache")),
            step_cache_threshold=float(params.get("step_cache_threshold") or 0.12),
            strength=float(params.get("strength") or 1.0),
            scheduler="linear",
        )
        image.save(os.path.abspath(out), export_json_metadata=False)
    except StopImageGenerationException:
        raise CancelledError("cancelled")
    finally:
        # Release scratch buffers but keep every weight resident: MLX accumulates
        # freed tensor buffers across denoising passes otherwise.
        gc.collect()
        mx.clear_cache()
    if not os.path.isfile(out):
        raise RuntimeError("generation finished but produced no output file")
    return {"out": os.path.abspath(out), "secs": round(time.time() - t0, 1)}


class CancelledError(Exception):
    pass


def _run_generate(params):
    job_id = params.get("job_id")
    try:
        result = _generate(params)
        _last_result.clear()
        _last_result.update(result)
        _last_result["job_id"] = job_id
        with _state.lock:
            _state.phase = "done"
            _state.job_id = job_id
        _log(f"job {job_id} done in {result['secs']}s -> {result['out']}")
    except CancelledError:
        with _state.lock:
            _state.phase = "cancelled"
            _state.job_id = job_id
        _log(f"job {job_id} cancelled")
    except Exception as e:
        _log(f"job {job_id} failed: {e}\n{traceback.format_exc()}")
        with _state.lock:
            _state.phase = "failed"
            _state.job_id = job_id
            _state.error = f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# dedicated MLX thread: handlers submit tasks, never touch mx.* themselves
# ---------------------------------------------------------------------------

_mlx_tasks = queue.Queue()


def _mlx_loop():
    while True:
        fn, args, out_q = _mlx_tasks.get()
        try:
            out_q.put((True, fn(*args)))
        except BaseException as e:  # propagate everything, incl. KeyboardInterrupt
            out_q.put((False, e))


def _run_on_mlx_thread(fn, args=()):
    """Run fn on the MLX thread and block until it finishes (or raises)."""
    out_q = queue.Queue()
    _mlx_tasks.put((fn, args, out_q))
    ok, val = out_q.get()
    if not ok:
        raise val
    return val


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # default logging is per-request noise
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        if self.path != "/status":
            return self._json({"error": "not found"}, 404)
        with _state.lock:
            st = {
                "phase": _state.phase,
                "step": _state.step,
                "total": _state.total,
                "job_id": _state.job_id,
                "error": _state.error,
                "result": dict(_last_result) if _state.phase == "done" else None,
                "ok": True,
                **{k: v for k, v in _model_info.items() if k != "_key"},
                "loaded": _model is not None,
            }
        return self._json(st)

    def do_POST(self):
        global _model
        if self.path == "/shutdown":
            self._json({"ok": True})
            _log("shutdown requested")
            os.kill(os.getpid(), signal.SIGTERM)
            return
        if self.path == "/cancel":
            _state.cancel.set()
            return self._json({"ok": True})
        if self.path == "/load":
            if _state.phase in BUSY_PHASES:
                return self._json({"error": "a render is running; cancel it first"}, 409)
            p = self._body()
            loras = tuple(sorted(os.path.abspath(x) for x in p.get("lora_paths") or []))
            key = (p.get("model_dir"), p.get("base_model"), int(p.get("bits") or 4),
                   bool(p.get("small_text_encoder")), loras)
            if _model is not None and _model_info.get("_key") == key:
                return self._json({"ok": True, "cached": True, **_model_info})
            with _state.lock:
                _state.phase = "loading"
                _state.step, _state.total, _state.job_id = 0, None, None
                _state.error = None
            try:
                model = _run_on_mlx_thread(_load_model, key)
            except Exception as e:
                _log(f"load failed: {e}\n{traceback.format_exc()}")
                with _state.lock:
                    _state.phase = "load-failed"
                    _state.error = f"{type(e).__name__}: {e}"
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            _model = model
            _model_info.clear()
            _model_info.update({"variant": os.path.basename(key[0]), "bits": key[2],
                                "small_text_encoder": key[3], "loras": list(loras),
                                "_key": key})
            with _state.lock:
                _state.phase = "idle"
                _state.error = None
            _log(f"loaded {key[0]} bits={key[2]} small_encoder={key[3]} loras={len(loras)}")
            return self._json({"ok": True, "cached": False, **_model_info})
        if self.path == "/generate":
            if _model is None:
                return self._json({"error": "no model loaded - POST /load first"}, 409)
            if _state.phase in BUSY_PHASES:
                return self._json({"error": "a render is already running"}, 409)
            p = self._body()
            _state.cancel.clear()
            with _state.lock:
                _state.phase = "encoding"
                _state.step, _state.total = 0, int(p.get("steps") or 0)
                _state.job_id = p.get("job_id")
                _state.error = None
            _mlx_tasks.put((_run_generate, (dict(p),), queue.Queue()))
            return self._json({"ok": True, "started": True, "job_id": p.get("job_id")})
        return self._json({"error": "not found"}, 404)


def _write_state(port):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"port": port, "pid": os.getpid()}, f)
    os.replace(tmp, STATE_PATH)


def _watchdog(parent_pid):
    while True:
        time.sleep(5)
        if os.getppid() != parent_pid:
            _log("parent gone - exiting")
            os._exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-pid", type=int, default=os.getppid())
    args = ap.parse_args()
    threading.Thread(target=_watchdog, args=(args.parent_pid,), daemon=True).start()
    threading.Thread(target=_mlx_loop, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    _write_state(port)
    _log(f"mlx worker listening on 127.0.0.1:{port} (pid {os.getpid()})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
