# vmf_api — local API, CLI and MCP server for VisoMaster Fusion

Drive a running VisoMaster Fusion from a script, the command line or an AI assistant, **without changing any
file of VisoMaster**: the add-on hooks into the app at start-up and stays inert unless it is asked for.

- **HTTP API** inside VisoMaster: `POST http://127.0.0.1:<port>/api/<command>`, JSON in and out, token required.
- **CLI** `vmf-cli`: one subcommand per API command, plus waits (`ready-wait`, `recording-wait`…), `launch` and `quit`.
- **MCP server** (`vmf_cli.py mcp`, stdio): the same commands as tools for any AI app or agent that supports MCP.

All three run the same code (`commands.py`): a command does the same thing whichever door it comes through.

Main uses:
- import someone else's preset or `last_workspace.json` **without overwriting machine-specific settings**
  (paths, GPU, threads, appearance… see `personal_keys.json`), with a diff first and a backup to undo;
- record a video, run VisoMaster's image batches or one frame × every input face, and get the timings;
- build the TensorRT engine cache ahead of time;
- run VisoMaster with no window (`launch --hidden`) for unattended jobs.

Tested with VisoMaster Fusion 3.17.2 on Windows (portable install).

## Install

For the Windows portable install of VisoMaster Fusion (the one started with `Start_Portable.bat`). The add-on is
already there, in `VisoMaster-Fusion\tools\vmf_api`.

1. Double-click `install.bat` in that folder, once. It prepares the portable Python and prints the lines to give to
   an MCP client. Run it again only if you reinstall the portable Python (`uninstall.bat` undoes it).
2. Start VisoMaster with the launcher as usual, tick **Enable API / MCP**, then **Launch**. The box is unticked
   each time the launcher opens, so the API only runs when you ask for it.

Check that it answers: open a terminal in `tools\vmf_api` and type `.\vmf-cli status`.

## Use it

**From the command line**: [docs/CLI.md](docs/CLI.md) starts with the everyday tasks. A few examples:

```bat
vmf-cli status
vmf-cli diff D:\received\their_workspace.json
vmf-cli apply-config D:\received\their_workspace.json
vmf-cli start-recording --from-start
vmf-cli recording-wait
```

**From an AI app or agent** (any MCP client): add a local MCP server that runs this command. `install.bat` prints
it with the full paths of your install, ready to paste; [docs/MCP.md](docs/MCP.md) shows where each kind of client
expects it.

```
<vmf_api>\.venv\Scripts\python.exe <vmf_api>\vmf_cli.py mcp
```

**From your own code**: plain HTTP + JSON, see [docs/API.md](docs/API.md).

## Documentation

| | |
|---|---|
| [docs/API.md](docs/API.md) | HTTP protocol, authentication, errors, asynchronous operations, examples (PowerShell, Python, curl) |
| [docs/CLI.md](docs/CLI.md) | `vmf-cli`: everyday tasks, output, exit codes, waits, scripting |
| [docs/MCP.md](docs/MCP.md) | MCP server: setup in any MCP client, tools, rules given to the model |
| [docs/DEVELOPING.md](docs/DEVELOPING.md) | architecture, how to add a command, security model, tests |
| [docs/REFERENCE.md](docs/REFERENCE.md) | every HTTP command, CLI subcommand and MCP tool (generated from the code) |

## Safety in short

- Listens on 127.0.0.1 only; random token per start in `runtime\server.json`; requests from a web page (`Origin`)
  or through another host name are refused.
- Write commands are refused while VisoMaster records, runs a batch, an issue scan, a job queue or a TensorRT build.
- Destructive actions need a confirmation code from a first call. Backups are written before `apply_config`,
  `restore_backup`, destructive actions, reset buttons, `load_workspace` and before `last_workspace.json` or an
  existing output file is overwritten.
- Never exposed: dialogs, creating / editing / deleting embeddings, turning on a webcam or the virtual camera,
  loading K/V map files from outside `model_assets`, extra environment variables other than `VISOMASTER_*` / `VMF_*`.
