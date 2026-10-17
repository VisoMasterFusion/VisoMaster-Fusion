# vmf_api — local API, CLI and MCP server for VisoMaster Fusion

Drive a running VisoMaster Fusion from a script, the command line or an AI assistant, **without changing any
file of VisoMaster**: the add-on hooks into the app at start-up and stays inert unless it is asked for.

- **HTTP API** inside VisoMaster: `POST http://127.0.0.1:<port>/api/<command>`, JSON in and out, token required.
- **CLI** `vmf-cli`: one subcommand per API command, plus waits (`ready-wait`, `recording-wait`…), `launch` and `quit`.
- **MCP server** (`vmf_cli.py mcp`, stdio): the same commands as tools for Claude Code or any MCP client.

All three run the same code (`commands.py`): a command does the same thing whichever door it comes through.

Main uses:
- import someone else's preset or `last_workspace.json` **without overwriting machine-specific settings**
  (paths, GPU, threads, appearance… see `personal_keys.json`), with a diff first and a backup to undo;
- record a video, run VisoMaster's image batches or one frame × every input face, and get the timings;
- build the TensorRT engine cache ahead of time;
- run VisoMaster with no window (`launch --hidden`) for unattended jobs.

Tested with VisoMaster Fusion 3.17.2 on Windows (portable install).

## Install

Two places are supported:

```
<BASE_DIR>\                        portable install (Start_Portable.bat)
├── portable-files\python\python.exe
├── VisoMaster-Fusion\             the git checkout
│   └── tools\vmf_api\             <- here (inside the repository) ...
└── vmf_api\                       <- ... or here (next to it; wins if both exist)
```

1. Run `vmf_api\install.bat` (idempotent). It checks the portable Python, copies `vmf_api_hook.pth` and
   `vmf_api_bootstrap.py` into its `site-packages`, creates `vmf_api\.venv` with the `mcp` SDK, and prints the
   `claude mcp add` command. Re-run it only if `portable-files\python` is reinstalled; `uninstall.bat` removes the two files.
2. Start VisoMaster as usual (launcher), tick **Enable API / MCP**, then *Launch*. The box is unticked each time the
   launcher opens: the API only runs when asked for. `vmf-cli launch [--hidden]` starts VisoMaster with the API
   without the launcher.

## Quick start

```bat
vmf-cli status
vmf-cli diff D:\received\their_workspace.json
vmf-cli apply-config D:\received\their_workspace.json --faces all
vmf-cli restore-backup
```

```bat
vmf-cli launch --hidden
vmf-cli load-target-media D:\images --wait
vmf-cli run-action batchallImageButton
vmf-cli recording-wait
vmf-cli quit --no-save
```

Claude Code:
```
claude mcp add visomaster -- <vmf_api>\.venv\Scripts\python.exe <vmf_api>\vmf_cli.py mcp
```

## Documentation

| | |
|---|---|
| [docs/API.md](docs/API.md) | HTTP protocol, authentication, errors, asynchronous operations, examples (PowerShell, Python, curl) |
| [docs/CLI.md](docs/CLI.md) | `vmf-cli`: output, exit codes, waits, scripting |
| [docs/MCP.md](docs/MCP.md) | MCP server: setup, tools, rules given to the model |
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
