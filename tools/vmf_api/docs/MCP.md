# MCP server

`vmf_cli.py mcp` serves the API commands as [Model Context Protocol](https://modelcontextprotocol.io) tools
over stdio, so that an AI app or agent can drive VisoMaster. It is a standard MCP server: it works with any
client that supports MCP, and nothing in it is tied to one assistant. It is a client of the [HTTP API](API.md)
like the CLI, and runs in `vmf_api\.venv` (the official `mcp` SDK, installed by `install.bat`), not in
VisoMaster's Python. Every tool: [REFERENCE.md](REFERENCE.md#mcp-tools).

## Setup

In your MCP client, add a **local server** (also called *stdio* or *command* server) with:

| Field | Value |
|---|---|
| name | `visomaster` (any name) |
| command | `<vmf_api>\.venv\Scripts\python.exe` |
| arguments | `<vmf_api>\vmf_cli.py` `mcp` |

`<vmf_api>` is the full path of the `VisoMaster-Fusion\tools\vmf_api` folder. `install.bat` prints the
command and the JSON below with your real paths, ready to paste. No environment variable, port or token to set:
the server finds VisoMaster's API by itself.

Most clients read a JSON file with an `mcpServers` section (desktop AI apps, IDE assistants, agent CLIs; the file
name and place depend on the client, see its MCP documentation):
```json
{
  "mcpServers": {
    "visomaster": {
      "command": "C:\\VisoMaster\\VisoMaster-Fusion\\tools\\vmf_api\\.venv\\Scripts\\python.exe",
      "args": ["C:\\VisoMaster\\VisoMaster-Fusion\\tools\\vmf_api\\vmf_cli.py", "mcp"]
    }
  }
}
```

Other formats you may meet, same command and arguments:

- a `servers` section with `"type": "stdio"` (e.g. VS Code's `mcp.json`);
- TOML, `[mcp_servers.visomaster]` with `command = "…"` and `args = ["…", "mcp"]` (e.g. Codex CLI);
- a command line, e.g. Claude Code (`--scope user`: in every project; without it, only in the folder where the
  command was typed):
  `claude mcp add --scope user visomaster -- <vmf_api>\.venv\Scripts\python.exe <vmf_api>\vmf_cli.py mcp`.

The client must run on the same Windows machine as VisoMaster: the server talks to it on `127.0.0.1`. A client
that only accepts remote servers (a URL) needs a stdio-to-HTTP MCP bridge in front of this command; keep that
bridge on the local machine, since whoever reaches it controls VisoMaster.

The server starts without VisoMaster: `launch_visomaster` starts it (with `hidden=true`, no window at all),
and every other tool answers "VisoMaster is not running with the API" (or "API unreachable" when a
`server.json` is left over) until it runs with the API. Restart the MCP server after
updating `vmf_cli.py`; changes to `commands.py` / `server.py` need a VisoMaster restart instead.

## Tools

Mostly one tool per API command, same name (`set` is `set_value`, `quit` is `quit_visomaster`). On top of them:

| Tool | |
|---|---|
| `launch_visomaster`, `quit_visomaster` | start VisoMaster with the API and wait until it is idle; close it and wait until the process exits |
| `wait_ready`, `wait_recording`, `wait_tensorrt_build` | one call that returns when the work is done, with MCP progress notifications meanwhile: the model never polls |
| `start_recording` | waits until VisoMaster is idle (models loaded) before starting, so the timing does not include a model load |
| `snapshot` | returns the PNG as an image the model can see (`include_image=false`: size only) |
| `apply_offline` | merge a config into `last_workspace.json` while VisoMaster is closed |

Paths are made absolute by the MCP process before they are sent. `diff` and `apply_config` also take `config`, the
JSON object itself (e.g. settings read from screenshots), in place of `path`: no file to write first.

## Rules given to the model

The server's `instructions` (sent to the client at connection) tell the model:

- personal settings are never overwritten by `apply_config`; run `diff` before `apply_config`;
- write tools are refused during a recording, a scan, a job queue or a TensorRT build;
- embeddings: only list them and select one; creating, editing, deleting, loading or saving them is reserved to the user;
- never poll in a loop: use the `wait_*` tools; send nothing but `wait_recording` while a recording runs;
- a map of which tool does what (state, settings, import, backups, swap, recording, batches, jobs, TensorRT, workspace).

Destructive actions answer with a warning and a confirmation code first: the model is told to show it to the
user and to call again only after their approval. Nothing in the protocol enforces that a human agreed, so the
guard rails that matter are on the server side (refusals listed in [DEVELOPING.md](DEVELOPING.md#security-model)).

## Errors

A refused or failed command becomes an MCP tool error carrying the API's message (e.g. `refused: recording in
progress`, `not ready: loading input faces (230 so far). Call wait_ready …`). The messages name the next step
for every client (MCP tool, CLI subcommand, plain HTTP).
