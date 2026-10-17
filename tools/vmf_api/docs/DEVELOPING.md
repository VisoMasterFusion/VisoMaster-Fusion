# Developing vmf_api

## Architecture

```
python main.py  (VMF_API=1)                       vmf-cli / MCP server / any script
   │ site-packages\vmf_api_hook.pth                        │
   ▼                                                       │  POST /api/<command>
vmf_api_bootstrap.py ── inert unless launcher / app        │  Bearer token
   ▼                                                       ▼
vmf_hooks.py ── post-import hook on app.ui.main_ui ──► server.py (127.0.0.1, HTTP thread)
                                                          │ queued Qt signal + Future (timeout)
                                                          ▼
                                                     commands.py (Qt main thread) ──► VisoMaster objects
```

| File | Role |
|---|---|
| `vmf_api_hook.pth`, `vmf_api_bootstrap.py` | copied into the portable Python's `site-packages` by `install.bat`. Run at every interpreter start; inert (imports nothing) except for the launcher (`-m app.ui.launcher`, `launcher.py`) and the app (`main.py` with `VMF_API=1`) |
| `vmf_hooks.py` | import hooks (meta path finder, no timer): adds the *Enable API / MCP* box to the launcher (sets `VMF_API=1` for the app process when ticked), starts the server after `MainWindow` opens, stops it on quit. If a hook point is gone after an update, it logs `[VMF-API] … unavailable` and VisoMaster starts normally without the API |
| `server.py` | HTTP server (exclusive port bind), token / Host / Origin checks, dispatch to the Qt main thread |
| `commands.py` | every command (`@command`), busy / ready checks, recording and batch tracking, TensorRT cache builds |
| `schema.py` | settings schema read from VisoMaster's `*_layout_data.py` (types, bounds, options, labels, `depends_on`); testable without Qt |
| `merge.py` | config import: personal-key exclusions, validation, diff plan; testable without Qt |
| `personal_keys.json` | the exclusion list used by `apply_config` / `diff` (groups, keys with a reason, fnmatch patterns, `force_include`); re-read at each command |
| `vmf_cli.py` | CLI (`build_parser`, `run_cli`) and MCP server (`build_mcp`); waits, `launch`, `quit`, offline merge |
| `vmf_common.py` | paths for both install layouts, `config.json` (port), `runtime\server.json` |
| `winqos.py` | turns off Windows power throttling (EcoQoS) for VisoMaster's process, so a recording runs at the same speed with its window in the background |
| `gen_docs.py` | writes `docs/REFERENCE.md` from the code |
| `install.bat`, `uninstall.bat`, `install_helper.py` | install / remove the two site-packages files, create `.venv` with the `mcp` SDK |

Runtime files (not versioned): `config.json`, `runtime\` (server.json, launch log, diagnostics), `backups\`,
`exports\`, `snapshots\`, `.venv\`.

## Where the state lives in VisoMaster

| What | Where | Note |
|---|---|---|
| Settings tab (`SETTINGS_LAYOUT_DATA`), Denoiser (`DENOISER_LAYOUT_DATA`), `OutputMediaFolder` | `main_window.control` (global) | `data_type="control"` |
| Swapper, Face Editor, Face Restorer, expressions (`COMMON_LAYOUT_DATA`…) | `main_window.parameters[face_id]` (per target face) | `data_type="parameter"` |
| Panel shown with no face selected (template for new faces) | `main_window.current_widget_parameters` | |

Values are written like the UI does: the widget is updated and the setting's `exec_function` runs, so side effects
(model loading, mask models…) happen as after a click.

## VisoMaster internals the add-on relies on

The add-on calls these private functions and attributes; a change in VisoMaster may need a matching change here:

- hook points: `app.ui.launcher.launcher_window` (launcher window class), `app.ui.main_ui` (`MainWindow`);
- `save_load_actions.save_current_workspace` / `load_saved_workspace`;
- `list_view_actions.select_target_medias` / `select_input_face_images` / `get_all_merged_embeddings`;
- `video_control_actions.process_batch_images`, `is_issue_scan_active`, `resolve_output_folder`;
- `common_actions.create_and_show_messagebox` / `refresh_frame` / `set_widgets_values_using_face_id_parameters`,
  `widget_components.ProgressDialog` (batches run without dialogs);
- `video_processor` attributes: `recording`, `processing`, `frames_written`, `start_time`, `current_frame`,
  `process_current_frame(synchronous=True)`; `models_processor.load_model`, `get_gpu_memory`;
- the layout data modules (`*_layout_data.py`, `models_toggle_data.MODELS_TOGGLE_MAP`).

`tests/test_commands.py` covers the logic with fake objects; a real run (`vmf-cli launch --hidden`, a few
commands, `quit`) is the check after a VisoMaster update.

## Adding a command

1. **Server** (`commands.py`):
   ```python
   @command("my_command", write=True, timeout=60)
   def cmd_my_command(ctx, path: str, force: bool = False):
       """One line for the reference. Details after."""
       mw = ctx.mw
       ...
       return {"done": True}          # JSON-serializable
   ```
   - `write=True` for anything that changes VisoMaster: refused while busy, before the function runs.
   - `main_thread=False` only for plain reads that never touch Qt (they run in the HTTP thread, e.g. polled during a recording).
   - `timeout`: how long the HTTP thread waits for the main thread.
   - Refuse with `raise CommandError("message that says what to do next")`; never open a dialog or block on user input.
   - Paths: `_abs(path)`; files written where the caller says: `_out_path(path, ".ext")`; changes to the config: write a
     backup first (`_write_backup(mw, "before-my-command")`).
   - Long work: start it with `QtCore.QTimer.singleShot(0, …)`, return at once, report progress through a status command.
2. **CLI** (`vmf_cli.py`): a subparser in `build_parser()` (dashes in the name) and its call in `run_cli()`.
3. **MCP** (`vmf_cli.py`, `build_mcp()`): a `@mcp.tool()` function whose docstring tells the model when to use it;
   pass absolute paths (`_abs`).
4. **Docs**: `.venv\Scripts\python.exe gen_docs.py` (regenerates `docs/REFERENCE.md`; `tests/test_docs.py` fails otherwise).
5. **Tests**: `tests/test_commands.py` style, with `SimpleNamespace` fakes for the main window.

## Security model

The API runs with the user's rights inside VisoMaster: whoever can call it can do what the UI can do. The aim is
that only the local user's own programs can call it, and that a mistaken or malicious call cannot do more than the UI.

- **Who can call**: 127.0.0.1 only; token from `secrets.token_urlsafe(32)` at each start, compared in constant time;
  `Origin` header or foreign `Host` → 403 (a web page cannot reach the API, even through DNS rebinding); 20 MB body
  limit; 60 s socket timeout.
- **What is refused**:
  - code execution paths: `launch` environment limited to `VISOMASTER_*` / `VMF_*` (no `PYTHONSTARTUP`, `PATH`…);
    `load_workspace` refuses K/V map files outside `model_assets` (VisoMaster loads them with `torch.load(weights_only=False)`);
  - privacy: turning on a webcam or the virtual camera (`set`, `select_target_media`, `run_action`);
  - user-owned data: creating, editing, deleting, loading or saving embeddings and K/V maps;
  - dialogs, and actions whose slot may open one (found in the slot's source, two call levels deep, plus a manual list);
  - files written with an unexpected extension (`.json`, `.png` only where the caller chooses the path), or inside
    VisoMaster's / vmf_api's folders;
  - workspaces that would turn on a webcam / the virtual camera, or that are damaged (VisoMaster clears the session
    before it notices);
  - recordings that would open a message box (image target, one-frame video, invalid markers).
- **What is protected**: write commands while busy; destructive actions need a single-use confirmation code (a guard
  against mistakes, not proof of a human); backups before `apply_config`, `restore_backup`, destructive actions and reset buttons, before
  `load_workspace` (nothing is loaded if that backup fails), before overwriting `last_workspace.json` (`save_workspace`,
  `quit`) or an existing output file; personal settings never imported by `apply_config`.
- **What is not hidden**: the token file is readable by whoever can read the `vmf_api` folder; diagnostics
  (`VMF_DETECT_LOG=1`) write face boxes and landmarks to `runtime\` until deleted.

## Diagnostics

Environment variables for VisoMaster's process (`vmf-cli launch --env KEY=VALUE`):

| Variable | Effect |
|---|---|
| `VMF_PROFILE_DETECTOR=1` | samples the detector and swap threads every ~1 ms, writes the costliest functions to `runtime\detector_profile.txt` after each recording |
| `VMF_GIL_PROBE=1` | adds `gil_probe` (delay to get the GIL back) to recording results |
| `VMF_GIL_SWITCH_INTERVAL_MS` | changes Python's GIL switch interval (default 5 ms) |
| `VMF_DETECT_LOG=1` | writes per-frame detections and frame MD5s to `runtime\` to check that a VisoMaster change keeps the output identical |
| `VMF_ALLOW_POWER_THROTTLING=1` | leaves Windows power throttling (EcoQoS) as Windows sets it, instead of turning it off (`winqos.py`), to compare |

## Install layouts

`vmf_common._layout()` (and the same rule in `install_helper.py` and `vmf_api_bootstrap._vmf_dir`) accepts
`<BASE_DIR>\vmf_api` next to the `VisoMaster-Fusion` checkout, or `VisoMaster-Fusion\tools\vmf_api` inside it;
the first wins when both exist. `VMF_API_HOME` overrides the folder (used by the tests).

Only the portable install is supported for now: the `.pth` file goes into `portable-files\python`. For a conda /
venv install, copy `vmf_api_hook.pth` and `vmf_api_bootstrap.py` into that environment's `site-packages` and set
`VMF_API_HOME` to this folder.

## Tests

```
portable-files\python\python.exe -m unittest discover -s <vmf_api>\tests      (all, including the HTTP and server tests)
<vmf_api>\.venv\Scripts\python.exe -m unittest discover -s <vmf_api>\tests     (all but those needing PySide6; includes the MCP tests)
```

Run both before a pull request: neither one covers everything.

| File | Covers |
|---|---|
| `test_commands.py` | busy refusals, TensorRT builds and confirmations, destructive actions, recording tracking, schema search, readiness, launch environment |
| `test_security.py` | env allowlist, workspace and K/V map checks, output paths, backups, privacy refusals, NaN, HTTP 401 / 403 |
| `test_merge.py` | exclusions, patterns, validation, preset / workspace formats, offline merge (fixtures in `tests/fixtures`) |
| `test_cli.py` | progress lines, exit codes |
| `test_mcp.py` | every MCP tool through the real stdio server against a fake API |
| `test_bootstrap.py` | bootstrap inert for any other Python process, active for the launcher / `main.py`, launcher checkbox, both install layouts |
| `test_server.py` | a second instance cannot take the same port |
| `test_docs.py` | `docs/REFERENCE.md` is up to date |
