# MOSS-Image-SFT (INT4) — System Architecture (HLD & LLD)

## 1. High-Level Design (HLD)

The local MOSS-Image-SFT system exposes a 5B-parameter Diffusion Transformer (`MOSS-Image-SFT-Q4_0.gguf`) paired with `Qwen3.5-9B-Q4_K_M.gguf` text encoding, `flux2_ae_small.safetensors` VAE decoding, and 3-step/4-step DMD Turbo LoRAs over Apple Silicon Metal via `stable-diffusion.cpp` (`sd-server`), wrapped by a Python client (`moss_client.py`) and an MCP stdio server (`moss_mcp.py`).

Quality modes map to sampler settings in `moss_client.generate`: `draft` = 3-step LoRA, `standard` = 4-step LoRA (both CFG 1 with the release scheduler's snapped sigmas), `high` = no LoRA, 50 steps, CFG 3 (official settings).

```mermaid
flowchart LR
    subgraph Clients["MCP & CLI Clients"]
        MCPApp["Claude / Gemini / Cursor (MCP)"]
        CLI["Shell / Python Pipeline"]
    end

    subgraph Middleware["MOSS Python Layer"]
        MCP["moss_mcp.py (6 MCP Tools)"]
        Client["moss_client.py (HTTP + Cutout Matting)"]
    end

    subgraph Backend["Apple Silicon Metal Backend (127.0.0.1:18765)"]
        SDServer["sd-server (stable-diffusion.cpp)"]
        Qwen["Qwen3.5-9B-Q4_K_M Text Encoder"]
        DiT["MOSS-Image-SFT-Q4_0 (5B DiT)"]
        LoRA["3-Step / 4-Step DMD Turbo LoRA (FP16, unused in high mode)"]
        VAE["FLUX.2 Small VAE"]
    end

    MCPApp -->|"stdio JSON-RPC"| MCP
    CLI --> Client
    MCP --> Client
    Client -->|"POST /sdcpp/v1/img_gen"| SDServer
    SDServer --> Qwen
    Qwen --> DiT
    LoRA -.->|"Runtime Unmerged"| DiT
    DiT --> VAE
    VAE -->|"Base64 PNG"| Client
```

---

## 2. Low-Level Design (LLD) — Two-Stage Single-Object Cutout Pipeline

Standard text-to-image prompts produce floor shadows, studio cyclorama gradients, or canvas edge clipping, and naive global thresholding erases white highlights inside the object. Our two-stage pipeline solves both failure modes:

1. **Stage 1 (`cutout_prompt`)**: Wraps the single object inside MOSS's native `abs_centralized` `yxyx` bounding box with a `10%` outer margin (`[0.10*W, 0.10*H, 0.90*W, 0.90*H]`) and explicit flat-matte lighting directives (`zero cast shadows, no floor, no horizon line, no vignette`). This guarantees the outer 10% frame perimeter is pure solid background and prevents any extremity from clipping the canvas edge.
2. **Stage 2 (`extract_cutout_rgba`)**: Samples the median RGB color along the 1px outer perimeter, computes a C-accelerated Chebyshev distance map (`ImageChops.difference` + `ImageChops.lighter`), and seeds `ImageDraw.floodfill` exclusively from perimeter pixels. Interior matching colors (e.g. white feathers, porcelain dials, specular glints) remain completely unreachable from the border and keep `alpha = 255`. Finally, a 2px morphological boundary band (`MaxFilter(5) - ext_bg`) receives continuous alpha ramping and foreground color decontamination:
   $$C_{\text{fg}} = \text{clip}\left(\frac{C_{\text{obs}} - (1 - \alpha) C_{\text{bg}}}{\max(\alpha, 0.20)}, 0, 255\right)$$

```mermaid
sequenceDiagram
    participant Caller as Caller / MCP Tool
    participant Client as moss_client.py
    participant Server as sd-server (Metal)
    participant Matte as extract_cutout_rgba (Pillow C)

    Caller->>Client: generate_cutout(object_prompt, bg_color, tolerance)
    Client->>Client: cutout_prompt() -> abs_centralized yxyx box [10%..90%]
    Client->>Server: POST /sdcpp/v1/img_gen (4-step LoRA, euler, custom_sigmas)
    Server-->>Client: Plain solid-background PNG
    Client->>Matte: extract_cutout_rgba(plain_path, tolerance=28, feather=2)
    Matte->>Matte: 1. Sample 1px border median (R_bg, G_bg, B_bg)
    Matte->>Matte: 2. Chebyshev distance map & border-seeded floodfill
    Matte->>Matte: 3. Preserve enclosed interior white/highlight pixels (alpha=255)
    Matte->>Matte: 4. 2px boundary band soft-alpha ramp + color unmixing
    Matte-->>Client: Clean RGBA PNG (<stem>_rgba.png)
    Client-->>Caller: (plain_path, rgba_path, total_secs)
```

---

## 3. Backend lifecycle and failure handling

```mermaid
sequenceDiagram
    participant App as MCP app
    participant MCP as moss_mcp.py
    participant Lock as logs/.server.lock
    participant SD as sd-server :18765

    App->>MCP: tools/call generate_image
    MCP->>SD: GET /sdcpp/v1/capabilities (identity: model == MOSS-Image-SFT-Q4_0.gguf)
    alt nothing listening
        MCP->>Lock: flock (serialises start across processes)
        MCP->>SD: start_server.sh (refuses if a MOSS backend or another program owns the port)
        MCP->>SD: wait for identity (<=120 s)
    else another service on the port
        MCP-->>App: ToolError "port used by another service, set MOSS_PORT"
    end
    MCP->>SD: POST /sdcpp/v1/img_gen
    loop every 0.5 s
        MCP->>SD: GET /sdcpp/v1/jobs/{id}
        MCP-->>App: progress (queued / step n of 50 in high mode)
    end
    alt app cancels / timeout / error
        MCP->>SD: POST /sdcpp/v1/jobs/{id}/cancel
    else backend vanished mid-job
        MCP->>MCP: restart backend, retry once
    end
    SD-->>MCP: PNG (rejected if near-uniform: GPU OOM signature)
    MCP-->>App: inline JPEG preview + saved PNG path
```

- Expected failures are raised as `MossError` in the client and converted to MCP `ToolError` in the server, so the
  user sees the actual message (MCP SDK 2.x hides the text of any other exception).
- A backend started by an MCP process runs in its own session and is stopped when that process exits, unless
  `MOSS_KEEP_SERVER=1`. Other MCP processes using it recover through the retry above.
- Step previews (`preview: proj`) are requested only for full-model renders; they cost ~1% and turbo jobs don't need them.
