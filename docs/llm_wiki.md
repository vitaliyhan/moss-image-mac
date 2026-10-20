# LLM Wiki — MOSS-Image-SFT Local Server & Cutout Engine

## Quick Orientation for AI Agents

- **Repository Root**: the directory containing this `docs/` folder
- **Core Modules**:
  - [`moss_client.py`](../moss_client.py): Python HTTP client, layout prompt builder (`layout_prompt`), single-object isolation prompt generator (`cutout_prompt`), border-flood-fill RGBA alpha-matting engine (`extract_cutout_rgba`), end-to-end cutout generator (`generate_cutout`), and CLI entrypoint.
  - [`moss_mcp.py`](../moss_mcp.py): MCP (SDK 2.x `MCPServer`) stdio server exposing 6 tools: `generate_image`, `generate_layout_image`, `generate_cutout_object`, `edit_image`, `transform_image`, and `image_server_status`.
  - [`tests/test_cutout.py`](../tests/test_cutout.py): Unit test suite covering color resolution, layout-box cutout prompt generation, interior-highlight preservation during border flood-fill RGBA extraction, and MCP tool invocation.

## Critical Architectural Rules & Invariants

1. **Turbo LoRA + Custom Sigmas Pairing**:
   - `steps=4` (`quality="standard"`, alias `"best"`): pairs `moss_4step_f16.safetensors` with `custom_sigmas=[1.0, 0.91, 0.717, 0.024, 0.0]` and `txt_cfg=1.0`.
   - `steps=3` (`quality="draft"`): pairs `moss_3step_f16.safetensors` with `custom_sigmas=[1.0, 0.835, 0.024, 0.0]` and `txt_cfg=1.0`.
   - `steps=50` (`quality="high"`): no LoRA, no custom sigmas, `txt_cfg=3.0` (official settings; pixel-identical to the reference pipeline).
   - Each LoRA is trained for exactly its own step count: never pair a turbo LoRA with other step counts or CFG > 1.
2. **Layout Prompt (`abs_centralized` `yxyx`) Rules**:
   - MOSS-Image-SFT uses center-origin `[y0 - cy, x0 - cx, y1 - cy, x1 - cx]` coordinates (`bbox_coord_space="abs_centralized"`, `bbox_format="yxyx"`).
   - Never put title-like phrases in `summary.caption` unless you want them rendered as text; keep `summary.caption` purely descriptive of scene/backdrop style and put all visible text inside dedicated `"label": "text"` items.
3. **Single-Object Cutout Generation (`generate_cutout` / `generate_cutout_object`)**:
   - Combines a 10%-padded central layout bounding box (`margin=0.10`) with flat-matte shadowless lighting directives and border-connected flood-fill alpha matting (`extract_cutout_rgba`) so objects never clip the frame border and internal white details remain 100% opaque.
4. **Errors**: raise `moss_client.MossError` for anything the user should read; `moss_mcp._tool` converts it to an MCP `ToolError`. Do not raise bare exceptions from tools (their text is hidden by the SDK).
5. **One backend only**: never start a second `sd-server`; `_ensure_server` holds `logs/.server.lock` while starting. Two loaded copies exhaust GPU memory and yield blank images (detected by `_looks_blank`).
6. **Default port is 18765** (`MOSS_PORT`), verified by model identity; 7860 commonly belongs to other web UIs.
7. **Tests**: `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -t .` (fake backend, no GPU).
