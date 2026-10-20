# Troubleshooting

Backend log: `logs/sd-server.log`. Check whether a backend is up:

```bash
curl -s http://127.0.0.1:18765/sdcpp/v1/capabilities | head -c 200
```

| Symptom | Cause | Fix |
|---|---|---|
| `model files missing … Run setup.sh` | Setup not run, or interrupted | `./setup.sh` (it resumes and skips finished files) |
| `checksum mismatch` during setup | Corrupted or partial download | Re-run `./setup.sh`; the bad file was deleted |
| `bin/sd-cli does not run on this Mac` | macOS older than 26 with the prebuilt binary | `./setup.sh --build-from-source` (needs `brew install cmake` and `xcode-select --install`) |
| `port … is used by another service` | Something else (often a Stable Diffusion WebUI) owns the port | Set `MOSS_PORT` to a free port, in your shell or the MCP config's `env` |
| `blank image … GPU ran out of memory` | Another large model, game or a second backend is using GPU memory | Quit it and retry. Never run two backends; `start_server.sh` refuses to start a second one |
| `backend did not start within 120 s` | Very slow disk, or the backend crashed at start | Read `logs/sd-server.log`; check free memory |
| First image is ~15 s slower | Models load on the first request | Expected. Call `image_server_status` (MCP) or start `./start_server.sh` ahead of time |
| `high` call cancelled by the app | The client caps tool-call duration | Use `standard`, or run `./generate.sh "prompt" 768 432 -1 high` in a terminal |
| Text in the image is misspelled | Prompt didn't quote it, or the text is too small for the canvas | Put exact words in quotes; use a larger size or a layout box for the text |
| Words from the layout caption appear in the image | MOSS renders caption text | Keep the layout `caption` to style and scene; put all text in items |
| `canvas … out of range` | Sides outside 256-1536 px or over ~1.1 MP | Use a preset `size` or a smaller custom size |
| `image not found` / `cannot read image` | Path not on this Mac, or not PNG/JPEG/WebP | Use an absolute local path; convert HEIC to PNG/JPEG first |

Reset everything except the downloaded models:

```bash
pkill -f "bin/sd-server"; rm -rf .venv logs; ./setup.sh
```
