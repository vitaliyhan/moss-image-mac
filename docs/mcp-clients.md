# Connecting MCP clients

`moss_mcp.py` is a stdio MCP server. Every client needs the same two things:

- **command:** `/absolute/path/to/moss-image-mac/.venv/bin/python`
- **args:** `["/absolute/path/to/moss-image-mac/moss_mcp.py"]`

Use absolute paths. Apps launched from the Dock don't inherit your shell's working directory or PATH.

## `mcpServers` format

Used by Claude Desktop, Cursor, Windsurf, LM Studio and most other clients:

```json
{
  "mcpServers": {
    "moss-image": {
      "command": "/absolute/path/to/moss-image-mac/.venv/bin/python",
      "args": ["/absolute/path/to/moss-image-mac/moss_mcp.py"],
      "env": { "MOSS_OUTPUT_DIR": "/Users/you/Pictures/moss" }
    }
  }
}
```

| Client | Config file |
|---|---|
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` (restart the app) |
| Cursor | `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project) |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` |
| LM Studio | `~/.lmstudio/mcp.json` |

`env` is optional; see the configuration table in the README.

## VS Code

`.vscode/mcp.json` in a workspace (or the user-level MCP config) uses `servers`:

```json
{
  "servers": {
    "moss-image": {
      "type": "stdio",
      "command": "/absolute/path/to/moss-image-mac/.venv/bin/python",
      "args": ["/absolute/path/to/moss-image-mac/moss_mcp.py"]
    }
  }
}
```

## Claude Code

```bash
claude mcp add moss-image -s user -- /absolute/path/to/moss-image-mac/.venv/bin/python /absolute/path/to/moss-image-mac/moss_mcp.py
```

## Behaviour to know

- **First call** starts the backend; expect ~15 s extra while the models load. Use `image_server_status` to warm it up ahead of time.
- **Several apps** share one backend. Requests queue; long `high` renders report their queue position.
- **Closing the app** that started the backend stops it (frees ~9 GB). Set `MOSS_KEEP_SERVER=1` to keep it warm.
- **Long calls:** `standard` takes ~30-60 s. `high` takes 10-30 min and sends progress notifications every few seconds.
  Clients that reset their timeout on progress handle that. Clients with a hard per-call limit may cancel it, which
  also cancels the backend job; use `generate.sh … high` from a terminal in that case.
- **Images** are returned inline (JPEG preview) and saved as PNG; the text result includes the file path.
- **File paths** for `edit_image` / `transform_image` must be on this Mac and readable by the app (`~` is expanded).
