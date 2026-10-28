#!/usr/bin/env python3
"""Qwen-Image-2.1 via the mflux MLX runtime (Apple-Silicon native).

Renders run in a persistent worker process (qwen_mlx_server.py) that keeps the
model resident: weights, quantization and mx.compile are paid once, not per
image, and the transformer's text/image prefix K/V caches survive across renders.
The worker is spawned on the first MLX job, recorded in logs/mlx_worker.json,
and exits when this process (its parent) goes away. Set MOSS_MLX_SUBPROCESS=1
to bypass the worker and shell out to the mflux CLI per render, the old way.

Weights are local mflux-format dirs downloaded by setup_qwen.sh into
models/mlx/<name> (transformer/, text_encoder/, vae/, processor/ - all
components included, nothing is fetched at render time).

    from qwen_mlx import mlx_available, generate
    generate("a fox", out="fox.png", model_dir="models/mlx/mlx-q4",
             turbo=False, steps=40, seed=42)
"""
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

from moss_client import MossError

ROOT = os.path.dirname(os.path.abspath(__file__))
MLX_DIR = os.path.join(ROOT, "models", "mlx")
LOGS_DIR = os.path.join(ROOT, "logs")
WORKER_STATE = os.path.join(LOGS_DIR, "mlx_worker.json")
WORKER_LOG = os.path.join(LOGS_DIR, "mlx_worker.log")
DMD2_LORA = os.path.join(ROOT, "models", "lora", "qwen-image-2.1-dmd2-2steps.safetensors")
BASE_MODEL_TURBO = "qwen-image-2.1-turbo"
BASE_MODEL = "qwen-image-2.1"
TURBO_STEPS = 8
BASE_STEPS = 40

# models/mlx/<dir> -> (is_turbo, quantization bits); built by setup_qwen.sh
VARIANTS = {
    "mlx-q4": (False, 4), "mlx-q6": (False, 6), "mlx-q8": (False, 8),
    "mlx-turbo-q4": (True, 4), "mlx-turbo-q6": (True, 6), "mlx-turbo-q8": (True, 8),
}


class MfluxUnavailable(MossError):
    """mflux is not installed (run setup_qwen.sh) or the model dir is missing."""


def installed_variants():
    """{name: "base|turbo qN-bit"} for every downloaded variant directory."""
    out = {}
    for name, (turbo, q) in VARIANTS.items():
        if os.path.isfile(os.path.join(MLX_DIR, name, "model_index.json")):
            out[name] = f"{'turbo' if turbo else 'base'} {q}-bit"
    return out


def mlx_available():
    """True if the mflux CLI exists in .venv."""
    return os.path.isfile(os.path.join(ROOT, ".venv", "bin", "mflux-generate-qwen-2.1-edit"))


def _mflux_bin(name):
    p = os.path.join(ROOT, ".venv", "bin", name)
    if not os.path.isfile(p):
        raise MfluxUnavailable("mflux is not installed - run: ./setup_qwen.sh mlx-q4")
    return p


def _model_dir(name):
    d = os.path.join(MLX_DIR, name)
    if not os.path.isfile(os.path.join(d, "model_index.json")):
        raise MfluxUnavailable(f"model '{name}' is not downloaded - run: ./setup_qwen.sh {name}")
    return d


def default_steps(turbo):
    return TURBO_STEPS if turbo else BASE_STEPS


# ---------------------------------------------------------------------------
# persistent worker (qwen_mlx_server.py)
# ---------------------------------------------------------------------------

class _WorkerDown(Exception):
    """The worker is unreachable or its protocol broke - caller may fall back."""


def _venv_python():
    p = os.path.join(ROOT, ".venv", "bin", "python")
    if not os.path.isfile(p):
        raise MfluxUnavailable(".venv python not found - run: ./setup_qwen.sh mlx-q4")
    return p


def _http(method, port, path, payload=None, timeout=10):
    """(status_code, parsed_body); raises _WorkerDown only when the worker is
    unreachable - HTTP error statuses (409 busy, 500 failed) come back as data."""
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read() or b"{}")
        except (json.JSONDecodeError, OSError):
            body = {"error": f"HTTP {e.code}"}
        return e.code, body
    except (urllib.error.URLError, OSError) as e:
        raise _WorkerDown(str(e)) from e


def _worker_status(timeout=2):
    """(port, status_dict) if a live worker is recorded, else None."""
    try:
        with open(WORKER_STATE) as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        _, st = _http("GET", info["port"], "/status", timeout=timeout)
    except _WorkerDown:
        return None
    return info["port"], st


def _spawn_worker():
    """Start the worker and wait until it serves /status. Returns its port."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log = open(WORKER_LOG, "ab")
    subprocess.Popen(
        [_venv_python(), os.path.join(ROOT, "qwen_mlx_server.py"),
         "--parent-pid", str(os.getpid())],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=False,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        got = _worker_status(timeout=1)
        if got:
            return got[0]
        time.sleep(0.3)
    raise _WorkerDown("worker did not start - see logs/mlx_worker.log")


def _ensure_worker(model_dir, base_model, bits, small_text_encoder, loras,
                   on_progress=None, steps=None):
    """Worker serving /status with this exact model (+LoRAs) loaded. Returns its port.

    The model stays resident across renders; only a changed key (model, bits,
    encoder, LoRA set) triggers a reload."""
    got = _worker_status()
    if not got:
        got = _spawn_worker(), None
    port, st = got
    want_loras = tuple(sorted(os.path.abspath(p) for p in loras))
    loaded = st and st.get("loaded")
    same = loaded and (st.get("variant") == os.path.basename(model_dir)
                       and st.get("bits") == bits
                       and bool(st.get("small_text_encoder")) == bool(small_text_encoder)
                       and tuple(sorted(st.get("loras") or ())) == want_loras)
    if same:
        if on_progress:
            on_progress(0, steps, "model resident - encoding prompt")
        return port
    if on_progress:
        on_progress(0, steps, "loading model into worker (once per change)")
    code, resp = _http("POST", port, "/load",
                       {"model_dir": model_dir, "base_model": base_model,
                        "bits": bits, "small_text_encoder": bool(small_text_encoder),
                        "lora_paths": list(want_loras)},
                       timeout=1800)
    if code != 200:
        raise MossError(f"mflux load failed: {resp.get('error', code)}")
    # Re-verify: an older worker process silently ignores fields it doesn't know.
    _, st = _http("GET", port, "/status", timeout=10)
    if not (st.get("loaded") and st.get("variant") == os.path.basename(model_dir)
            and st.get("bits") == bits
            and bool(st.get("small_text_encoder")) == bool(small_text_encoder)
            and tuple(sorted(st.get("loras") or ())) == want_loras):
        raise _WorkerDown("worker did not load the requested model/LoRA combination "
                          "(stale worker? kill it and retry)")
    return port


def _generate_worker(prompt, out, model_dir, base_model, bits, steps, seed, width,
                     height, guidance, ref_images, step_cache, small_text_encoder,
                     loras, on_progress, cancel_event, timeout):
    t0 = time.time()
    port = _ensure_worker(model_dir, base_model, bits, small_text_encoder, loras,
                          on_progress=on_progress, steps=steps)
    job_id = f"mlx-{int(t0)}"
    code, resp = _http("POST", port, "/generate", {
        "job_id": job_id, "out": os.path.abspath(out), "prompt": str(prompt),
        "steps": int(steps), "seed": int(seed), "width": int(width), "height": int(height),
        "guidance": guidance, "ref_images": list(ref_images),
        "step_cache": bool(step_cache),
    }, timeout=30)
    if code != 200:
        raise _WorkerDown(f"/generate rejected: {resp.get('error', code)}")
    if on_progress:
        on_progress(0, steps, "encoding prompt" if not ref_images else "encoding prompt + references")

    deadline = time.time() + timeout
    reported_step = -1
    while True:
        if cancel_event is not None and cancel_event.is_set():
            try:
                _http("POST", port, "/cancel", {}, timeout=5)
            except _WorkerDown:
                pass
        if time.time() > deadline:
            try:
                _http("POST", port, "/cancel", {}, timeout=5)
            except _WorkerDown:
                pass
            raise MossError("render timed out")
        _, st = _http("GET", port, "/status", timeout=10)
        phase = st.get("phase")
        if phase == "sampling" and on_progress and st.get("step") != reported_step:
            reported_step = st.get("step")
            on_progress(min(reported_step, steps), steps, "sampling")
        elif phase == "loading" and on_progress and reported_step < 0:
            on_progress(0, steps, "loading model")
        elif phase == "decoding" and on_progress and reported_step != steps:
            reported_step = steps
            on_progress(steps, steps, "decoding")
        if phase in ("done", "cancelled", "failed"):
            break
        time.sleep(0.4)

    if phase == "cancelled":
        raise MossError("cancelled")
    if phase == "failed":
        raise MossError(st.get("error") or "render failed inside the MLX worker")
    result = st.get("result") or {}
    if not os.path.isfile(out):
        raise _WorkerDown("worker reported done but the output file is missing")
    if on_progress:
        on_progress(steps, steps, "done")
    return os.path.abspath(out), time.time() - t0


def _generate_subprocess(prompt, out, model_dir, base_model, bits, steps, seed, width,
                         height, guidance, ref_images, step_cache=False, loras=(),
                         on_progress=None, cancel_event=None):
    """Old path: one mflux CLI process per render (fallback / MOSS_MLX_SUBPROCESS=1)."""
    cmd = [_mflux_bin("mflux-generate-qwen-2.1-edit"),
           "--model", model_dir,
           "--base-model", base_model,
           "--prompt", str(prompt),
           "--steps", str(int(steps)),
           "--seed", str(int(seed)),
           "--width", str(max(32, int(width) // 32 * 32)),
           "--height", str(max(32, int(height) // 32 * 32)),
           "--quantize", str(bits),
           "--output", os.path.abspath(out)]
    if guidance is not None:
        cmd += ["--guidance", str(guidance)]
    if step_cache:
        cmd += ["--use-step-cache"]
    if loras:
        for lora in loras:
            cmd += ["--lora", os.path.abspath(lora)]
    if ref_images:
        cmd += ["--image-paths", *ref_images]

    env = dict(os.environ, HF_ENDPOINT=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        # mflux reports per-step progress as "NN%" lines; fall back to a time estimate.
        pat = re.compile(r"(\d+)%")
        for line in proc.stdout:
            m = pat.search(line)
            if m and on_progress:
                pct = min(99, int(m.group(1)))
                on_progress(pct * steps // 100, steps, f"sampling ~{pct}%")
            if cancel_event is not None and cancel_event.is_set():
                os.killpg(proc.pid, signal.SIGTERM)
                raise MossError("cancelled")
        code = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        raise MossError("mflux did not exit after finishing")
    except MossError:
        proc.wait(timeout=60)
        raise
    except Exception:
        os.killpg(proc.pid, signal.SIGKILL)
        raise
    if cancel_event is not None and cancel_event.is_set():
        raise MossError("cancelled")
    if code != 0:
        raise MossError(f"mflux exited with code {code} - see logs/ui_backend.log")
    if not os.path.isfile(out):
        raise MossError("mflux finished but produced no output file")
    if on_progress:
        on_progress(steps, steps, "done")
    return out, time.time() - t0


def generate(prompt, out="out.png", model_dir=None, turbo=False, quant=4,
             width=1024, height=1024, seed=-1, steps=None, guidance=None,
             ref_images=(), step_cache=False, small_text_encoder=False,
             loras=(), on_progress=None, cancel_event=None, timeout=3600):
    """Generate/edit one image with mflux. ref_images: 0-8 local file paths.

    step_cache: skip unchanged transformer blocks on similar steps - faster, slightly
    different output. small_text_encoder: swap the 8B prompt encoder for a 0.8B one
    (experimental: English prompts only, numerals in rendered text may be off).
    loras: LoRA safetensors paths applied at full strength (e.g. the DMD2 2-step
    distill; pair it with steps=2 and guidance=1).
    on_progress(step, total, message); setting cancel_event stops the render.
    Returns (out_path, elapsed_seconds). Raises MossError on failure.
    """
    if model_dir is None:
        model_dir = f"mlx-turbo-q{quant}" if turbo else f"mlx-q{quant}"
    model_dir = _model_dir(model_dir)
    turbo, quant = VARIANTS.get(os.path.basename(model_dir), (turbo, quant))
    base_model = BASE_MODEL_TURBO if turbo else BASE_MODEL
    if turbo and steps is None:
        steps = TURBO_STEPS
    if steps is None:
        steps = BASE_STEPS
    if not str(prompt or "").strip():
        raise MossError("prompt is empty")
    refs = [os.path.abspath(p) for p in ref_images][:8]
    for p in refs:
        if not os.path.isfile(p):
            raise MossError(f"reference image not found: {p}")
    loras = [os.path.abspath(p) for p in loras]
    for p in loras:
        if not os.path.isfile(p):
            raise MossError(f"LoRA file not found: {p} - run: ./setup_qwen.sh lora-dmd2")
    if seed == -1:
        seed = int.from_bytes(os.urandom(4), "little") % (2**31)

    kwargs = dict(model_dir=model_dir, base_model=base_model, bits=quant, steps=steps,
                  seed=seed, width=width, height=height, guidance=guidance,
                  ref_images=refs, loras=loras)
    if os.environ.get("MOSS_MLX_SUBPROCESS"):
        return _generate_subprocess(prompt, out, step_cache=step_cache,
                                    on_progress=on_progress, cancel_event=cancel_event,
                                    **kwargs)
    try:
        return _generate_worker(prompt, out, step_cache=step_cache,
                                small_text_encoder=small_text_encoder,
                                on_progress=on_progress, cancel_event=cancel_event,
                                timeout=timeout, **kwargs)
    except _WorkerDown as e:
        print(f"qwen_mlx: worker unavailable ({e}) - falling back to mflux CLI",
              flush=True)
    except MossError as e:
        cancelled = str(e) == "cancelled" or (cancel_event is not None and cancel_event.is_set())
        if cancelled or str(e) == "render timed out":
            raise  # nothing failed - don't redo the whole render on the other path
        print(f"qwen_mlx: worker render failed ({e}) - falling back to mflux CLI",
              flush=True)
    return _generate_subprocess(prompt, out, step_cache=step_cache,
                                on_progress=on_progress, cancel_event=cancel_event,
                                **kwargs)


if __name__ == "__main__":  # quick manual test: python3 qwen_mlx.py "a fox" fox.png
    import sys
    p, o = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "out.png"
    path, secs = generate(p, out=o, on_progress=lambda s, t, m: print(f"\r{s}/{t} {m}", end="", flush=True))
    print(f"\nsaved {path} in {secs:.1f}s")
