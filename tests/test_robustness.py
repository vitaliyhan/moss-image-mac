"""Edge-case tests for the client/MCP layer against a fake sd-server (no GPU, runs in seconds).

    PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -t .
"""
import asyncio
import base64
import io
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from PIL import Image, ImageDraw

import moss_client
from moss_client import BackendUnavailable, MossError


def _png(blank=False, size=(64, 64)):
    im = Image.new("RGB", size, (118, 114, 106))
    if not blank:
        ImageDraw.Draw(im).ellipse([8, 8, 56, 56], fill=(250, 40, 40))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeBackend:
    """Minimal emulation of the sd-server endpoints the client uses."""

    def __init__(self, model="MOSS-Image-SFT-Q4_0.gguf", steps_to_finish=3, blank=False,
                 fail=None, never_finish=False):
        self.model, self.steps_to_finish, self.blank = model, steps_to_finish, blank
        self.fail, self.never_finish = fail, never_finish
        self.jobs, self.cancelled, self.bodies = {}, [], []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/sdcpp/v1/capabilities":
                    return self._send(200, {"model": {"name": fake.model}})
                if self.path.startswith("/sdcpp/v1/jobs/"):
                    jid = self.path.rsplit("/", 1)[-1]
                    j = fake.jobs.get(jid)
                    if not j:
                        return self._send(404, {"error": "job not found"})
                    j["polls"] += 1
                    if j["cancelled"]:
                        return self._send(200, {"id": jid, "status": "cancelled"})
                    if fake.fail:
                        return self._send(200, {"id": jid, "status": "failed", "error": {"message": fake.fail}})
                    if fake.never_finish or j["polls"] <= fake.steps_to_finish:
                        st = "queued" if j["polls"] == 1 else "generating"
                        pv = {"step": j["polls"], "total_steps": 4} if st == "generating" else None
                        return self._send(200, {"id": jid, "status": st, "queue_position": 1, "preview": pv})
                    return self._send(200, {"id": jid, "status": "completed",
                                            "result": {"images": [{"b64_json": _png(fake.blank)}]}})
                self._send(404, {})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/sdcpp/v1/img_gen":
                    if body.get("width", 0) > 4096:
                        return self._send(400, {"error": "too big"})
                    fake.bodies.append(body)
                    jid = f"job_{len(fake.jobs)}"
                    fake.jobs[jid] = {"polls": 0, "cancelled": False}
                    return self._send(202, {"id": jid, "status": "queued", "poll_url": f"/sdcpp/v1/jobs/{jid}"})
                if self.path.endswith("/cancel"):
                    jid = self.path.split("/")[-2]
                    fake.cancelled.append(jid)
                    if jid in fake.jobs:
                        fake.jobs[jid]["cancelled"] = True
                    return self._send(200, {"id": jid, "status": "cancelled"})
                self._send(404, {})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class ClientEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "o.png")
        self.backends = []
        self._server = moss_client.SERVER
        moss_client.POLL_INTERVAL = 0.01  # fast polling in tests

    def tearDown(self):
        moss_client.SERVER = self._server
        moss_client.POLL_INTERVAL = 0.5
        for b in self.backends:
            b.close()
        self.tmp.cleanup()

    def use(self, **kw):
        b = FakeBackend(**kw)
        self.backends.append(b)
        moss_client.SERVER = b.url
        return b

    def test_turbo_request_and_progress(self):
        b = self.use()
        seen = []
        path, _ = moss_client.generate("a cat", 768, 432, out=self.out,
                                       on_progress=lambda s, t, m: seen.append((s, t, m)))
        self.assertTrue(os.path.exists(path))
        body = b.bodies[0]
        self.assertEqual(body["sample_params"]["custom_sigmas"], [1.0, 0.91, 0.717, 0.024, 0.0])
        self.assertEqual(body["sample_params"]["guidance"]["txt_cfg"], 1.0)
        self.assertEqual(body["lora"][0]["path"], "moss_4step_f16.safetensors")
        self.assertNotIn("preview", body)  # turbo: no preview overhead
        self.assertTrue(any("queued" in m for _, _, m in seen))

    def test_high_quality_uses_full_model(self):
        b = self.use()
        moss_client.generate("a cat", 768, 432, out=self.out, steps=moss_client.QUALITY_STEPS["high"])
        sp = b.bodies[0]["sample_params"]
        self.assertEqual(sp["sample_steps"], 50)
        self.assertEqual(sp["guidance"]["txt_cfg"], 3.0)
        self.assertNotIn("custom_sigmas", sp)
        self.assertEqual(b.bodies[0]["lora"], [])
        self.assertNotIn("preview", b.bodies[0])  # no progress callback -> no preview overhead
        seen = []
        moss_client.generate("a cat", 768, 432, out=self.out, steps=50,
                             on_progress=lambda s, t, m: seen.append(t))
        self.assertEqual(b.bodies[1]["preview"], "proj")
        self.assertIn(4, seen)  # step-level progress for long renders

    def test_blank_image_is_reported_not_saved(self):
        self.use(blank=True)
        with self.assertRaisesRegex(MossError, "ran out of"):
            moss_client.generate("a cat", 512, 512, out=self.out)
        self.assertFalse(os.path.exists(self.out))
        with patch.dict(os.environ, {"MOSS_ALLOW_FLAT": "1"}):
            moss_client.generate("a cat", 512, 512, out=self.out)
        self.assertTrue(os.path.exists(self.out))

    def test_backend_failure_message(self):
        self.use(fail="out of memory")
        with self.assertRaisesRegex(MossError, "out of memory"):
            moss_client.generate("a cat", 512, 512, out=self.out)

    def test_timeout_cancels_job(self):
        b = self.use(never_finish=True)
        body = {"prompt": "x", "width": 512, "height": 512}
        with self.assertRaisesRegex(MossError, "timed out"):
            moss_client._run(body, self.out, timeout=0.05)
        self.assertEqual(b.cancelled, ["job_0"])

    def test_cancel_event_cancels_job(self):
        b = self.use(never_finish=True)
        ev = threading.Event()
        threading.Timer(0.05, ev.set).start()
        with self.assertRaisesRegex(MossError, "cancelled"):
            moss_client.generate("a cat", 512, 512, out=self.out, cancel_event=ev)
        self.assertEqual(b.cancelled, ["job_0"])

    def test_foreign_service_on_port(self):
        self.use(model="sd_xl_base_1.0.safetensors")
        with self.assertRaisesRegex(MossError, "another service"):
            moss_client.server_identity()

    def test_backend_down(self):
        moss_client.SERVER = "http://127.0.0.1:9"  # discard port: nothing listens
        self.assertIsNone(moss_client.server_identity())
        with self.assertRaises(BackendUnavailable):
            moss_client.generate("a cat", 512, 512, out=self.out)

    def test_input_validation(self):
        self.use()
        for w, h in [(4096, 512), (128, 512), (1536, 1536)]:
            with self.assertRaisesRegex(MossError, "out of range"):
                moss_client.generate("a cat", w, h, out=self.out)
        self.assertEqual(moss_client.validate_size(1000, 563), (992, 560))
        with self.assertRaisesRegex(MossError, "empty"):
            moss_client.generate("   ", 512, 512, out=self.out)
        with self.assertRaisesRegex(MossError, "strength"):
            moss_client.generate("a cat", 512, 512, out=self.out, strength=1.5)
        with self.assertRaisesRegex(MossError, "not found"):
            moss_client.generate("a cat", 512, 512, out=self.out, init_image="/nope/x.png")

    def test_layout_boxes_are_clamped_and_validated(self):
        p = json.loads(moss_client.layout_prompt("s", [{"caption": "t", "box": [700, 400, -50, 900]}], 768, 432))
        self.assertEqual(p["items"][0]["bbox"], [184, -384, 216, 316])  # yxyx, centre origin, clamped
        with self.assertRaisesRegex(MossError, "empty or outside"):
            moss_client.layout_prompt("s", [{"caption": "t", "box": [900, 0, 1000, 10]}], 768, 432)
        with self.assertRaisesRegex(MossError, "needs box"):
            moss_client.layout_prompt("s", [{"caption": "t"}], 768, 432)


class McpEdgeCases(unittest.TestCase):
    def setUp(self):
        import moss_mcp
        self.mcp = moss_mcp
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_models_message(self):
        with patch.object(self.mcp.moss_client, "server_identity", return_value=None), \
                patch.object(self.mcp, "ROOT", self.mcp.Path(self.tmp.name)):
            with self.assertRaisesRegex(MossError, "setup.sh"):
                self.mcp._ensure_server()

    def test_retry_once_when_backend_lost(self):
        calls = []

        def flaky(*a, on_progress=None, cancel_event=None):
            calls.append(1)
            if len(calls) == 1:
                raise BackendUnavailable("gone")
            return ("ok.png", 1.0)

        with patch.object(self.mcp, "_ensure_server", lambda: None):
            self.assertEqual(asyncio.run(self.mcp._run(flaky)), ("ok.png", 1.0))
        self.assertEqual(len(calls), 2)

    def test_client_cancel_sets_cancel_event(self):
        seen = {}

        def slow(*a, on_progress=None, cancel_event=None):
            seen["ev"] = cancel_event
            cancel_event.wait(5)
            return ("x", 0)

        async def go():
            t = asyncio.ensure_future(self.mcp._run(slow))
            await asyncio.sleep(0.2)
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t

        with patch.object(self.mcp, "_ensure_server", lambda: None):
            asyncio.run(go())
        self.assertTrue(seen["ev"].is_set())

    def test_size_and_input_errors(self):
        with self.assertRaisesRegex(MossError, "both width and height"):
            self.mcp._resolve_size("16:9", "fast", 768, 0)
        self.assertEqual(self.mcp._resolve_size("16:9", "fast", 0, 0), (768, 432))
        with self.assertRaisesRegex(MossError, "not found"):
            self.mcp._open_image("~/definitely/missing.png")
        bad = os.path.join(self.tmp.name, "bad.png")
        with open(bad, "w") as f:
            f.write("not an image")
        with self.assertRaisesRegex(MossError, "cannot read"):
            self.mcp._open_image(bad)

    def test_all_presets_are_valid_canvases(self):
        for size, table in self.mcp.SIZES.items():
            for aspect, (w, h) in table.items():
                self.assertEqual(moss_client.validate_size(w, h), (w, h), (size, aspect))


if __name__ == "__main__":
    unittest.main()
