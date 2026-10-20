# HTTP API

`./start_server.sh` runs stable-diffusion.cpp's `sd-server` on `http://127.0.0.1:18765` (`MOSS_PORT`) with the models
kept loaded. It has **no authentication**, so keep `MOSS_HOST=127.0.0.1` unless you add your own proxy.

The simplest integration is the Python client, which applies the right settings for you:

```python
from moss_client import generate, layout_prompt, generate_cutout, MossError

path, secs = generate("prompt", width=768, height=432, seed=-1, out="a.png")           # standard
path, secs = generate("prompt", 768, 432, steps=3, out="b.png")                         # draft
path, secs = generate("prompt", 768, 432, steps=50, out="c.png",                        # high
                      on_progress=lambda step, total, msg: print(msg))
path, secs = generate("prompt", 768, 432, init_image="in.png", mask_image="mask.png",   # inpaint
                      strength=1.0, out="d.png")
```

`generate` raises `MossError` with a readable message on bad input, backend errors, timeouts and blank (out-of-memory)
output. Interrupting it (Ctrl-C, or setting a `cancel_event`) cancels the backend job.

## Calling the backend directly

If you call the endpoints yourself, send the settings that match each quality mode. Without the LoRA and sigmas, a
4-step request produces mush.

**Native async API:** `POST /sdcpp/v1/img_gen`, then poll `GET /sdcpp/v1/jobs/{id}`:

```json
{
  "prompt": "a lighthouse at dawn, watercolor",
  "width": 768, "height": 432, "seed": -1,
  "sample_params": {
    "sample_method": "euler",
    "sample_steps": 4,
    "custom_sigmas": [1.0, 0.91, 0.717, 0.024, 0.0],
    "guidance": { "txt_cfg": 1.0 }
  },
  "lora": [{ "path": "moss_4step_f16.safetensors", "multiplier": 1.0 }],
  "output_format": "png"
}
```

| Mode | `sample_steps` | `custom_sigmas` | `txt_cfg` | `lora` |
|---|---|---|---|---|
| draft | 3 | `[1.0, 0.835, 0.024, 0.0]` | 1.0 | `moss_3step_f16.safetensors` |
| standard | 4 | `[1.0, 0.91, 0.717, 0.024, 0.0]` | 1.0 | `moss_4step_f16.safetensors` |
| high | 50 | *(omit: server default schedule, shift 5)* | 3.0 | *(none)* |

Inpainting / img2img: add `init_image` (base64 PNG), optional `mask_image` (base64, white = repaint), and `strength`.
Cancel a job with `POST /sdcpp/v1/jobs/{id}/cancel`. Add `"preview": "proj"` to get `preview.step` /
`preview.total_steps` in job polls.

The Automatic1111-style (`/sdapi/v1/txt2img`, `/sdapi/v1/img2img`) and OpenAI-style (`/v1/images/generations`) endpoints
accept the same settings embedded in the prompt as `<sd_cpp_extra_args>{...}</sd_cpp_extra_args>`; see the
[stable-diffusion.cpp server docs](https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/api.md).

## Layout prompts

MOSS-Image reads a JSON layout as its prompt. `layout_prompt(caption, items, width, height)` builds it from top-left
pixel boxes. The native format uses `[y_min, x_min, y_max, x_max]` relative to the canvas centre:

```json
{"summary": {"caption": "scene / style only"},
 "items": [{"label": "text", "caption": "title 'HELLO'", "bbox": [-180, -320, -80, 320]}],
 "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
```

See `examples/layout_poster.json`.
