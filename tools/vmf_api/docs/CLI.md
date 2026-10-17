# vmf-cli

A client of the [HTTP API](API.md) for people and scripts. Standard library only: any Python 3.10+ runs it.

## Getting started

1. Run `install.bat` once and start VisoMaster with **Enable API / MCP** ticked ([README](../README.md#install)).
2. Open a terminal (cmd or PowerShell) in `VisoMaster-Fusion\tools\vmf_api` and type:

   ```
   .\vmf-cli status
   ```

   It prints VisoMaster's state as JSON: the media loaded, the target faces, whether the swap is on, the VRAM used.
   If it answers that VisoMaster is not running with the API, start it with the box ticked (or `vmf-cli launch`).

To type `vmf-cli` from any folder, add `tools\vmf_api` to your `PATH`. Without the `.bat`:
`<python> vmf_cli.py <subcommand>`.

`vmf-cli --help` lists the subcommands, `vmf-cli <subcommand> --help` their options. Every subcommand with its
arguments: [REFERENCE.md](REFERENCE.md#cli-subcommands).

## Everyday tasks

**Start and stop VisoMaster without the launcher**
```
vmf-cli launch                 start it with the API, wait until it is ready
vmf-cli launch --hidden        same, with no window at all (unattended jobs)
vmf-cli quit                   close it (saves the workspace, as the window's close button does)
vmf-cli quit --no-save         close it without saving
```

**Find a setting and change it**
```
vmf-cli schema                         list of the setting groups
vmf-cli schema restorer                every setting whose name contains "restorer": type, default, choices
vmf-cli get-config --keys FaceRestorerEnableToggle FaceRestorerTypeSelection
vmf-cli set FaceRestorerEnableToggle true
vmf-cli set FaceRestorerTypeSelection GFPGAN-v1.4
```
Per-face settings apply to the selected target face (`--face-id` for another one).

**Import someone else's settings without losing yours**
```
vmf-cli diff D:\received\their_workspace.json          what would change
vmf-cli apply-config D:\received\their_workspace.json  apply it (a backup is written first)
vmf-cli restore-backup                                 undo: back to the last backup
```
Paths, GPU, threads, appearance and other machine settings are never overwritten (`personal_keys.json`).
`--faces all` applies the face parameters to every target face, not only the selected one.

**Keep a known state and come back to it**
```
vmf-cli save-backup my-settings
vmf-cli list-backups
vmf-cli restore-backup my-settings
```

**Set up a swap**
```
vmf-cli load-target-media D:\videos\clip.mp4 --wait
vmf-cli select-target-media clip.mp4
vmf-cli run-action findTargetFacesButton
vmf-cli select-target-face                 list the target faces found
vmf-cli list-embeddings
vmf-cli select-embedding "My face"         use this embedding for the selected target face
```

**Record a video**
```
vmf-cli start-recording --from-start
vmf-cli recording-wait                     progress on screen, then the output file and its timings
```

**Run VisoMaster's image batch**
```
vmf-cli load-target-media D:\images --wait
vmf-cli run-action batchallImageButton
vmf-cli recording-wait
```

**Build the TensorRT engines ahead of time**
```
vmf-cli list-tensorrt-models
vmf-cli build-tensorrt-cache               the models the current settings use ("all" for every model)
vmf-cli tensorrt-wait
```

**See the current frame**
```
vmf-cli snapshot D:\out\frame.png
```

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
