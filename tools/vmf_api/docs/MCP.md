# MCP server

`vmf_cli.py mcp` serves the API commands as [Model Context Protocol](https://modelcontextprotocol.io) tools
over stdio, so that an AI assistant can drive VisoMaster. It is a client of the [HTTP API](API.md) like the
CLI, and runs in `vmf_api\.venv` (the official `mcp` SDK, installed by `install.bat`), not in VisoMaster's Python.
Every tool: [REFERENCE.md](REFERENCE.md#mcp-tools).

## Setup

Claude Code:
```
claude mcp add visomaster -- <vmf_api>\.venv\Scripts\python.exe <vmf_api>\vmf_cli.py mcp
```

Other clients (JSON configuration):
```json
{
  "mcpServers": {
    "visomaster": {
      "command": "<vmf_api>\\.venv\\Scripts\\python.exe",
      "args": ["<vmf_api>\\vmf_cli.py", "mcp"]
    }
  }
}
```

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

Paths are made absolute by the MCP process before they are sent.

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
