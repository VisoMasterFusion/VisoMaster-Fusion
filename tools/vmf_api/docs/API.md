# HTTP API

The API is a small HTTP server that runs **inside** VisoMaster's process. It is RPC over HTTP, not REST:
one endpoint per command, always `POST`, arguments and result in JSON. Every command, its arguments and its
description: [REFERENCE.md](REFERENCE.md#http-commands).

## Connecting

| | |
|---|---|
| Address | `http://127.0.0.1:<port>` only (never another interface). Port: 8765 by default, `vmf-cli api --port N` changes it for the next start |
| Discovery | `vmf_api\runtime\server.json`: `{"port", "token", "pid", "api_version", "started"}`. Written when the server starts, removed when it stops. **The token changes at every start (the port only with `vmf-cli api --port`): read the file each time** |
| Authentication | `Authorization: Bearer <token>` (or `X-VMF-Token: <token>`) on every request |
| Request | `POST /api/<command>`, body = JSON object of the arguments (`{}` or empty when there are none), at most 20 MB |
| Response | `{"ok": true, "result": …}` or `{"ok": false, "error": "…"}` |

## Status codes

| Code | When |
|---|---|
| 200 | the command ran **or was refused**: always test `ok` |
| 400 | body is not a JSON object |
| 401 | token missing or wrong |
| 403 | `Origin` header present, or `Host` other than `127.0.0.1:<port>` / `localhost:<port>` (browsers, DNS rebinding) |
| 404 | path outside `/api/` |
| 405 | `GET` instead of `POST` (other methods: 501) |
| 413 | body larger than 20 MB, or invalid `Content-Length` |

An unknown command answers 200 with `ok: false` and the list of commands in `result`.

## How a command runs

- Most commands run on VisoMaster's **Qt main thread**: the HTTP thread queues them and waits up to the
  command's timeout. If the main thread does not pick the command up in time (a modal dialog is open, a long
  operation runs), it is cancelled and the error says so. If it already started, it completes but its result is lost.
- Two reads, `ready_status` and `recording_status`, run in the HTTP thread and never touch Qt: they can be
  polled during a recording without disturbing it. The others (`tensorrt_build_status` included) are short
  main-thread commands: a client polling them retries when one times out while VisoMaster is busy.
- **Write commands are refused while VisoMaster is busy** (recording, batch, issue scan, job queue, TensorRT
  build): `ok: false`, error starting with `refused:`. Reads always work.
- Commands never open a dialog. Actions that would (file pickers, message boxes) are refused by `run_action`;
  the API has its own command instead (`load_target_media`, `save_workspace`…).

## Asynchronous operations

The HTTP API has no wait endpoint: start the operation, then poll a status command every few seconds.

| Start with | Poll | Done when |
|---|---|---|
| `load_target_media`, `load_input_faces` (cards arrive one by one) | `ready_status` | `ready` is true |
| any change that loads models lazily (swap on, a feature turned on, after Clear VRAM) | `ready_status` | `ready` is true |
| `start_recording`, `run_action batchallImageButton` / `batchImageButton`, `batch_input_faces` | `recording_status` | `active` is false; then read `result` |
| `build_tensorrt_cache` | `tensorrt_build_status` | `running` is false; then read `summary` |

`start_recording` and the batches refuse to start while VisoMaster is not ready (error starting with `not ready: `):
wait for `ready_status` first. The CLI and the MCP server do this for you (`ready-wait`, `recording-wait`,
`wait_ready`, `wait_recording`…).

## Confirmations

Destructive actions (`run_action clearTargetFacesButton`, a TensorRT rebuild…) do nothing on the first call and
answer `{"triggered": false, "confirmation_required": true, "warning": "…", "confirm": "<code>"}`. Call again
with `"confirm": "<code>"` to run it. The code is single-use, tied to that action, valid 5 minutes. It protects
against mistakes, not against a client that is determined to run the action: it is not a proof that a human agreed.

## Files and paths

- Paths are resolved by VisoMaster's process: send **absolute** paths (the CLI and the MCP server do it).
- Files the API writes where you ask must have the expected extension: `.json` (`export_config`,
  `save_workspace`), `.png` (`snapshot`), and be outside VisoMaster's and vmf_api's folders (except
  `last_workspace.json` for `save_workspace`). An existing file is copied to `backups\overwritten\` first
  (the 20 most recent copies are kept).
- Config backups (`save_backup`, and before `apply_config`, `restore_backup`, a destructive `run_action`, a
  reset `press_parameter_button`) go to `vmf_api\backups\`. Whole workspaces too: the current one before
  `load_workspace` (nothing is loaded if it fails), the previous `last_workspace.json` before `save_workspace`
  or `quit` overwrite it (10 of each kept).
- `load_workspace` checks the file first (a workspace, complete entries, K/V maps inside `model_assets`, no
  webcam / virtual camera turned on): VisoMaster clears the session before it notices a damaged file.

## Examples

PowerShell:
```powershell
$rt = Get-Content <vmf_api>\runtime\server.json | ConvertFrom-Json
$h  = @{ Authorization = "Bearer $($rt.token)" }
function Api($cmd, $body = @{}) {
    $r = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$($rt.port)/api/$cmd" -Headers $h `
         -ContentType 'application/json' -Body ($body | ConvertTo-Json -Compress)
    if (-not $r.ok) { throw $r.error }; $r.result
}
Api load_input_faces @{ path = 'D:\faces' }
do { Start-Sleep 1 } until ((Api ready_status).ready)
Api batch_input_faces
do { Start-Sleep 5; $b = Api recording_status; "$($b.progress.frames_written)/$($b.progress.total_frames)" } while ($b.active)
$b.result
```

Python (standard library only):
```python
import json, time, urllib.request
from pathlib import Path

rt = json.loads(Path(r"<vmf_api>\runtime\server.json").read_text(encoding="utf-8"))

def api(cmd, **args):
    req = urllib.request.Request(
        f"http://127.0.0.1:{rt['port']}/api/{cmd}", data=json.dumps(args).encode(), method="POST",
        headers={"Authorization": f"Bearer {rt['token']}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.load(r)
    if not res["ok"]:
        raise RuntimeError(res["error"])
    return res["result"]

print(api("diff", path=r"D:\received\preset.json")["control"]["counts"])
api("start_recording", from_start=True)
while (s := api("recording_status"))["active"]:
    time.sleep(5)
print(s["result"]["output_path"], s["result"]["processing_time_s"])
```

curl:
```
curl -s -X POST -H "Authorization: Bearer <token>" -H "Content-Type: application/json" -d "{}" http://127.0.0.1:8765/api/status
```
