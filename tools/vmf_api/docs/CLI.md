# vmf-cli

A client of the [HTTP API](API.md) for people and scripts. Standard library only: any Python 3.10+ runs it.

```
vmf_api\vmf-cli.bat <subcommand> [options]          (uses vmf_api\.venv, works from any folder)
<python> vmf_api\vmf_cli.py <subcommand> [options]
```

Add `vmf_api` to `PATH` to type `vmf-cli` anywhere. `vmf-cli --help` lists the subcommands,
`vmf-cli <subcommand> --help` their options. Every subcommand: [REFERENCE.md](REFERENCE.md#cli-subcommands).

## Names

Subcommands are the API commands with dashes: `apply_config` → `apply-config`. Some run in the CLI itself
(no HTTP endpoint): `launch`, `ready-wait`, `recording-wait`, `tensorrt-wait`, `apply-offline` (merge into
`last_workspace.json` while VisoMaster is closed), `api`, `mcp`. `quit` calls the API, then waits for the
process to exit.

## Output

- **stdout**: one JSON document, `{"ok": true, "result": …}` or `{"ok": false, "error": "…"}`; `--compact`
  (before the subcommand) prints it on one line.
- **stderr**: progress lines of the waits (`batch 19/500 input faces (3.8%), 3.7 per s, about 130 s left`).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | done |
| 1 | refused or failed (message in `error`), or the API is unreachable; `recording-wait`: ended without a file; `tensorrt-wait`: a model failed |
| 2 | invalid command line |
| 3 | nothing done: destructive action, run again with `--confirm <code>` |
| 4 | a wait (`*-wait`, `--wait`) gave up before the end (the work goes on); `quit`: VisoMaster still running; `launch`: started but not idle yet |

## Arguments

- Paths are made absolute from the folder the command is typed in (also `set <…Folder> <relative path>`).
- `set <key> <value>`: the value is sent as typed and typed by the server from the schema (`1` / `true` for a
  toggle, `2.50` for a selection, numbers within bounds). `nan` / `inf` are refused.
- `launch --env KEY=VALUE`: only `VISOMASTER_*` / `VMF_*` variables, except `VMF_API`, `VMF_API_PORT`,
  `VMF_API_HOME` and `VMF_PYTHON`. `--port`: 1024-65535.
- `launch` starts `portable-files\python\python.exe`; set `VMF_PYTHON` (in the CLI's own environment) to use
  another `python.exe`. If the API does not answer in time, a hidden VisoMaster is stopped; one with a window is
  left running (a dialog may wait for an answer on screen).
- `launch --hidden` needs a VisoMaster started once before (a `last_workspace.json`): on its first start
  VisoMaster asks for the execution provider in a dialog, which would stay invisible. It is refused otherwise.

## Asynchronous operations

`load-target-media` and `load-input-faces` return at once (cards arrive one by one): add `--wait`, or run
`ready-wait`. Recordings and batches return at once too: follow them with `recording-wait`, which prints the
progress and returns the result when the file is written. `start-recording` waits until VisoMaster is idle first.

## Scripts

Batch file:
```bat
call vmf-cli launch --hidden || exit /b 1
call vmf-cli set OutputMediaFolder D:\out
call vmf-cli load-input-faces D:\faces --wait || exit /b 1
call vmf-cli batch-input-faces || exit /b 1
call vmf-cli recording-wait
call vmf-cli quit --no-save
```

PowerShell, reading the result:
```powershell
$r = vmf-cli --compact recording-wait | ConvertFrom-Json
if (-not $r.ok) { throw $r.error }
$r.result.result.processing_time_s
```

Destructive action:
```
vmf-cli run-action clearTargetFacesButton          -> exit 3, "confirm": "a1b2c3"
vmf-cli run-action clearTargetFacesButton --confirm a1b2c3
```
