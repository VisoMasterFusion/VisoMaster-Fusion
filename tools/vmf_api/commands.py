"""Commands of the VisoMaster API. They all run on the Qt main thread.

Each command takes (ctx, args: dict) and returns a JSON-serialisable result,
or raises CommandError with a readable message. VisoMaster internals are
reached through getattr/lazy imports: if something moved after an update,
the command fails with a clear error instead of crashing the app.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import merge
import schema as schema_mod
from vmf_common import (
    API_VERSION,
    APP_DIR,
    BACKUP_DIR,
    EXPORT_DIR,
    PERSONAL_KEYS_PATH,
    SNAPSHOT_DIR,
    VMF_DIR,
    write_json_atomic,
)


class CommandError(Exception):
    pass


@dataclass
class Command:
    fn: Callable
    write: bool = False  # refused while recording / scanning / job queue
    timeout: float = 30.0
    help: str = ""
    # False: runs in the HTTP thread (plain reads only, no Qt), e.g. polling during a recording
    main_thread: bool = True

    def run(self, ctx: "Context", args: dict):
        if self.write:
            reason = busy_reason(ctx.mw)
            if reason:
                raise CommandError(f"refused: {reason}")
        return self.fn(ctx, **args)


COMMANDS: dict[str, Command] = {}

# State of the background TensorRT cache build (see build_tensorrt_cache).
_TRT_JOB: dict | None = None


def command(name: str, write: bool = False, timeout: float = 30.0, main_thread: bool = True):
    def deco(fn):
        COMMANDS[name] = Command(fn, write=write, timeout=timeout, help=(fn.__doc__ or "").strip(),
                                 main_thread=main_thread)
        return fn

    return deco


class Context:
    def __init__(self, main_window, api_port: int | None = None):
        self.mw = main_window
        self.api_port = api_port
        self._schema: schema_mod.Schema | None = None

    @property
    def schema(self) -> schema_mod.Schema:
        if self._schema is None:
            self._schema = live_schema()
        return self._schema

    @property
    def rules(self) -> merge.PersonalRules:
        # Re-read every time: the file is meant to be edited by hand.
        return merge.PersonalRules.load(PERSONAL_KEYS_PATH)


def live_schema() -> schema_mod.Schema:
    try:
        from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
        from app.ui.widgets.denoiser_layout_data import DENOISER_LAYOUT_DATA
        from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
        from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA
        from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA

        return schema_mod.from_layouts(
            [
                ("SETTINGS_LAYOUT_DATA", SETTINGS_LAYOUT_DATA, "control"),
                ("DENOISER_LAYOUT_DATA", DENOISER_LAYOUT_DATA, "control"),
                ("COMMON_LAYOUT_DATA", COMMON_LAYOUT_DATA, "parameter"),
                ("SWAPPER_LAYOUT_DATA", SWAPPER_LAYOUT_DATA, "parameter"),
                ("FACE_EDITOR_LAYOUT_DATA", FACE_EDITOR_LAYOUT_DATA, "parameter"),
            ]
        )
    except Exception:
        return schema_mod.from_source(APP_DIR)


# ---------------------------------------------------------------------------
# VisoMaster state helpers
# ---------------------------------------------------------------------------


def _vca():
    from app.ui.widgets.actions import video_control_actions

    return video_control_actions


def _common():
    from app.ui.widgets.actions import common_actions

    return common_actions


def _job_running(mw) -> bool:
    jp = getattr(mw, "job_processor", None)
    try:
        return bool(jp is not None and jp.isRunning())
    except Exception:
        return False


_FINALIZING = {"on": False}  # VisoMaster is closing the encoder / merging audio / saving (see _wrap_finalize)


def _recording(mw) -> bool:
    # VisoMaster sets recording=False as soon as it closes the encoder, before the file is written:
    # the finalization still counts as recording.
    vp = getattr(mw, "video_processor", None)
    return bool(vp is not None and (getattr(vp, "recording", False) or getattr(vp, "is_processing_segments", False)
                                    or _FINALIZING["on"]))


def _vram(mw) -> dict | None:
    """Whole-GPU VRAM, same source as VisoMaster's VRAM bar (nvidia-smi, else torch)."""
    mp = getattr(mw, "models_processor", None)
    try:
        used, total = mp.get_gpu_memory()
    except Exception:
        return None
    if not total:
        return None
    return {"used_mb": int(used), "total_mb": int(total), "percent": round(100.0 * used / total, 1)}


def _swap_state(mw) -> dict:
    """Is the swap active, and does each target face have a source (input faces or embedding)?"""
    btn = getattr(mw, "swapfacesButton", None)
    faces = {}
    for fid, face in (getattr(mw, "target_faces", {}) or {}).items():
        inputs = len(getattr(face, "assigned_input_faces", {}) or {})
        embeds = len(getattr(face, "assigned_merged_embeddings", {}) or {})
        faces[fid] = {"input_faces": inputs, "embeddings": embeds, "has_source": bool(inputs or embeds)}
    enabled = bool(btn.isChecked()) if btn is not None else None
    state = {"enabled": enabled, "faces": faces}
    if enabled is False:
        state["note"] = "swap is off (Clear VRAM turns it off): run_action swapfacesButton to turn it on"
    elif faces and not any(f["has_source"] for f in faces.values()):
        state["note"] = "no target face has a source: select_embedding or assign input faces"
    return state


# ---------------------------------------------------------------------------
# Readiness: model loading, preview frame
# ---------------------------------------------------------------------------

_LOADS_LOCK = threading.Lock()
_LOADS: dict = {"inflight": {}, "recent": []}  # recent: {"model", "seconds", "at"}
# TensorRT work VisoMaster announces with its build dialog (engine build, lazy first-run build,
# shape inference): running = stack of {"step", "model", "t0"}, done = {"step", "model", "seconds", "at"}
_BUILDS: dict = {"running": [], "done": []}
_LOAD_STORES = {"load_model": "models", "load_dfm_model": "dfm_models"}
NOT_READY = "not ready: "  # prefix of the start_recording error, recognized by the CLI / MCP
# Hints name the command for every client: MCP tool, CLI subcommand, plain HTTP (no wait endpoint).
WAIT_THEN = "Call wait_ready (CLI: ready-wait; HTTP: poll ready_status until ready), then "
WAIT_REC = "wait_recording (CLI: recording-wait; HTTP: poll recording_status until active=false)"


def install_hooks(mw):
    """Hooks installed when the API starts (in memory only)."""
    _install_console_tap()
    _install_load_hooks(mw)
    _install_rec_hooks(mw)


# [WARN] / [ERROR] lines VisoMaster prints, with their time: reported with a recording's result.
_CONSOLE: list = []
_CONSOLE_MAX = 300


class _ConsoleTap:
    """Pass-through for sys.stdout / sys.stderr that keeps VisoMaster's warning and error lines."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, s):
        n = self._stream.write(s)
        try:
            if "[WARN" in s or "[ERROR" in s:
                for line in s.splitlines():
                    if "[WARN" in line or "[ERROR" in line:
                        _CONSOLE.append((time.time(), line.strip()[:300]))
                del _CONSOLE[:-_CONSOLE_MAX]
        except Exception:
            pass
        return n

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _install_console_tap():
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and not isinstance(stream, _ConsoleTap):
            setattr(sys, name, _ConsoleTap(stream))


def _console_since(t0: float, limit: int = 10) -> list:
    out: list = []
    for t, line in list(_CONSOLE):
        if t >= t0 and line not in out:
            out.append(line)
    return out[-limit:]


def _install_load_hooks(mw):
    """Track model loads: VisoMaster loads models lazily, on the first frame that needs them (and
    builds a missing TensorRT engine at that moment)."""
    mp = getattr(mw, "models_processor", None)
    if mp is None:
        return
    for name, store in _LOAD_STORES.items():
        fn = getattr(mp, name, None)
        if fn is not None and not getattr(fn, "_vmf_wrapped", False):
            setattr(mp, name, _wrap_load(mp, fn, store))
    if not getattr(mp, "_vmf_build_hooked", False):
        try:
            mp.show_build_dialog.connect(_build_started)
            mp.hide_build_dialog.connect(_build_finished)
            mp._vmf_build_hooked = True
        except Exception:
            pass


def _build_started(title, text=""):
    lines = [x.strip() for x in str(text).splitlines() if x.strip()]
    model = lines[1] if len(lines) > 1 else None
    with _LOADS_LOCK:
        _BUILDS["running"].append({"step": str(title), "model": model, "t0": time.time()})


def _build_finished(*a):
    with _LOADS_LOCK:
        if not _BUILDS["running"]:
            return
        b = _BUILDS["running"].pop()
        _BUILDS["done"].append({"step": b["step"], "model": b["model"],
                                "seconds": round(time.time() - b["t0"], 1), "at": time.time()})
        del _BUILDS["done"][:-30]


def _builds_between(t0: float, t1: float) -> list:
    with _LOADS_LOCK:
        return [{k: b[k] for k in ("step", "model", "seconds")} for b in _BUILDS["done"]
                if t0 <= b["at"] - b["seconds"] and b["at"] <= t1 + 1]


def _wrap_load(mp, fn, store):
    def load(*a, **k):
        model = a[0] if a else next(iter(k.values()), "?")
        model = ", ".join(map(str, model)) if isinstance(model, (list, tuple)) else str(model)
        try:
            cached = bool((getattr(mp, store, None) or {}).get(model))
        except Exception:
            cached = False
        if cached:  # plain lookup, called for every frame
            return fn(*a, **k)
        token, t0 = object(), time.time()
        with _LOADS_LOCK:
            _LOADS["inflight"][token] = (model, t0)
        try:
            return fn(*a, **k)
        finally:
            dt = time.time() - t0
            with _LOADS_LOCK:
                _LOADS["inflight"].pop(token, None)
                if dt >= 0.05:
                    _LOADS["recent"].append({"model": model, "seconds": round(dt, 2), "at": time.time()})
                    del _LOADS["recent"][:-30]

    load._vmf_wrapped = True
    load.__wrapped__ = fn
    return load


def _loads_between(t0: float, t1: float) -> list:
    with _LOADS_LOCK:
        return [{"model": r["model"], "seconds": r["seconds"]} for r in _LOADS["recent"]
                if t0 <= r["at"] - r["seconds"] and r["at"] <= t1 + 1]


def _lazy_pending(mw) -> list:
    """Models whose TensorRT engine VisoMaster will compile at their first real use (cannot be built
    in advance)."""
    try:
        return sorted(getattr(getattr(mw, "models_processor", None), "models_pending_build", None) or ())
    except Exception:
        return []


def _engines_since(t0: float) -> list:
    """TensorRT engine files written since t0 (VisoMaster compiles missing engines on first use)."""
    root = APP_DIR / "tensorrt-engines"
    out = []
    try:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in (".engine", ".onnx") and p.stat().st_mtime >= t0:
                out.append(p.name)
    except Exception:
        pass
    return sorted(out)


def _preview_busy(mw) -> bool:
    """A preview frame is waiting or being computed (after a seek or a settings change)."""
    wpm = getattr(getattr(mw, "video_processor", None), "worker_pool_manager", None)
    if wpm is None:
        return False
    if getattr(wpm, "pending_single_frame_request", None) is not None:
        return True
    w = getattr(wpm, "current_single_frame_worker", None)
    try:
        return bool(w is not None and w.is_alive())
    except Exception:
        return False


def _media_loading(mw) -> list:
    """Target media / input faces still being loaded by VisoMaster's loader threads (load_target_media
    and load_input_faces return at once; the cards arrive one by one)."""
    out = []
    for attr, what, store in (("video_loader_worker", "target media", "target_videos"),
                              ("input_faces_loader_worker", "input faces", "input_faces")):
        w = getattr(mw, attr, None)
        try:
            running = bool(w is not None and w.isRunning())
        except Exception:  # deleted Qt object
            running = False
        # Cards read by the loader but not added yet (VisoMaster adds them in batches on a timer).
        try:
            pending = len(getattr(mw, f"_pending_{'target_media' if store == 'target_videos' else 'input_face'}"
                                      "_thumbnails", None) or ())
        except Exception:
            pending = 0
        if running or pending:
            out.append(f"loading {what} ({len(getattr(mw, store, {}) or {})} so far"
                       + (f", {pending} waiting to be added" if pending else "") + ")")
    return out


def _readiness(mw) -> dict:
    """Is VisoMaster idle: no model loading, no media loading, no preview frame or playback in progress,
    nothing busy? Plain reads only (safe outside the UI thread)."""
    now = time.time()
    with _LOADS_LOCK:
        loading = [{"model": m, "for_s": round(now - t0, 1)} for m, t0 in _LOADS["inflight"].values()]
        recent = [dict(r) for r in _LOADS["recent"][-5:]]
        building = [{"step": b["step"], "model": b["model"], "for_s": round(now - b["t0"], 1)}
                    for b in _BUILDS["running"]]
    waiting = []
    for b in building:
        waiting.append(f"TensorRT: {b['step']} ({b['model']}, {b['for_s']} s)")
    if loading:
        waiting.append("loading models: " + ", ".join(f"{x['model']} ({x['for_s']} s)" for x in loading))
    waiting += _media_loading(mw)
    busy = busy_reason(mw)
    if busy:
        waiting.append(busy)
    vp = getattr(mw, "video_processor", None)
    if getattr(vp, "processing", False) and not _recording(mw):
        waiting.append("playback in progress")
    if _preview_busy(mw):
        waiting.append("computing the preview frame")
    res = {"ready": not waiting}
    if waiting:
        res["waiting_for"] = waiting
    if loading:
        res["loading_models"] = loading
    if building:
        res["tensorrt_building"] = building
    if recent:
        for r in recent:
            r["ago_s"] = round(now - r.pop("at"), 1)
        res["last_model_loads"] = recent
    lazy = _lazy_pending(mw)
    if lazy:
        res["tensorrt_compiled_on_first_use"] = lazy
    return res


# ---------------------------------------------------------------------------
# Recording tracking (progress, end, output file)
# ---------------------------------------------------------------------------

_REC: dict = {}  # current or last recording, see _rec_update
# Face detection calls of VisoMaster's sequential detector (one thread for all frames, before the
# workers): (start, seconds, cpu seconds, model calls) on the perf_counter clock, see _wrap_detector.
# model calls: ((model, seconds), ...) of the ONNX Runtime runs made inside it, see _wrap_ort_run
_DETECT: list = []
_DETECT_TLS = threading.local()
# Same for SequentialDetector.precompute, when VisoMaster has it: the stateless part of the detection
# (detector model + embeddings) run ahead on upcoming frames by a small pool of threads.
_PREFETCH: list = []
# VMF_DETECT_LOG=1 (launch_visomaster env): keep what the sequential detector returned for each frame
# (boxes, 5-point landmarks, dense landmarks) and write it to runtime/detections_<id>.jsonl at the end of
# each recording, to check that a change in VisoMaster leaves the detection identical.
# Also the MD5 of every processed frame handed to the display / encoder (store_frame_to_display), to tell
# an image change from an encoding change.
# And the MD5 of every frame written to the encoder (FFmpegEncoder.write_frame), in write order.
_DETECT_LOG: dict = {"on": os.environ.get("VMF_DETECT_LOG") == "1", "rows": [], "frames": [], "written": []}


def _wrap_write_frame(fn):
    def write_frame(frame, *a, **k):
        try:
            import hashlib

            _DETECT_LOG["written"].append(hashlib.md5(frame[::8, ::8].tobytes()).hexdigest())
        except Exception:
            pass
        return fn(frame, *a, **k)

    write_frame._vmf_wrapped = True
    write_frame.__wrapped__ = fn
    return write_frame


def _wrap_store_frame(fn):
    def store_frame_to_display(frame_number, frame, *a, **k):
        try:
            import hashlib

            _DETECT_LOG["frames"].append((frame_number, hashlib.md5(frame[::8, ::8].tobytes()).hexdigest()))
        except Exception:
            pass
        return fn(frame_number, frame, *a, **k)

    store_frame_to_display._vmf_wrapped = True
    store_frame_to_display.__wrapped__ = fn
    return store_frame_to_display


def _detect_log_add(k: dict, out) -> None:
    def plain(x):
        try:
            return x.tolist()
        except Exception:
            return None

    try:
        bboxes, kpss_5, kpss, kpss_203 = out
        _DETECT_LOG["rows"].append({"frame": k.get("frame_number"), "bboxes": plain(bboxes),
                                    "kps5": plain(kpss_5), "kps": plain(kpss), "kps203": plain(kpss_203)})
    except Exception:
        pass


def _detect_log_dump(rec_id) -> str | None:
    rows, _DETECT_LOG["rows"] = _DETECT_LOG["rows"], []
    frames, _DETECT_LOG["frames"] = _DETECT_LOG["frames"], []
    written, _DETECT_LOG["written"] = _DETECT_LOG["written"], []
    if written:
        out = Path(__file__).resolve().parent / "runtime" / f"written_{rec_id}.txt"
        out.write_text("".join(f"{i} {h}\n" for i, h in enumerate(written)), encoding="utf-8")
    if frames:
        out = Path(__file__).resolve().parent / "runtime" / f"frames_{rec_id}.txt"
        out.write_text("".join(f"{n} {h}\n" for n, h in sorted(frames, key=lambda x: (x[0] is None, x[0] or 0))),
                       encoding="utf-8")
    if not rows:
        return None
    out = Path(__file__).resolve().parent / "runtime" / f"detections_{rec_id}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: (r["frame"] is None, r["frame"] or 0)):
            f.write(json.dumps(r) + "\n")
    return str(out)
# Swap workers' frames: (start, seconds, cpu seconds), see _wrap_worker_frame
_WORKERS: list = []
# VMF_GIL_PROBE=1 (launch_visomaster env): a thread releases the GIL every ~2 ms (sleep) and records how
# late it gets it back: (time, extra seconds). Large delays while recording = Python threads queue on the GIL.
_GIL_PROBE: dict = {"on": os.environ.get("VMF_GIL_PROBE") == "1", "samples": [], "thread": None}


def _gil_probe_loop():
    nap = 0.002
    while True:
        t = time.perf_counter()
        time.sleep(nap)
        late = time.perf_counter() - t - nap
        _GIL_PROBE["samples"].append((t, late))
        if len(_GIL_PROBE["samples"]) > 200000:
            del _GIL_PROBE["samples"][:100000]


def _gil_probe_stats(t0: float, t1: float) -> dict | None:
    lates = sorted(d for t, d in list(_GIL_PROBE["samples"]) if t0 <= t <= t1)
    if not lates:
        return None
    n = len(lates)
    return {
        "samples": n,
        "avg_extra_ms": round(sum(lates) / n * 1000, 2),
        "p50_ms": round(lates[n // 2] * 1000, 2),
        "p95_ms": round(lates[int(n * 0.95)] * 1000, 2),
        "max_ms": round(lates[-1] * 1000, 2),
        "note": "how late a thread gets the GIL back after a 2 ms sleep (OS timer jitter included: "
                "compare with an idle VisoMaster)",
    }
# VMF_PROFILE_DETECTOR=1 (launch_visomaster env): sample the detector thread's stack every ~1 ms while it
# is inside run(), and write the hottest functions to runtime/detector_profile.txt at the end of each
# recording. (cProfile cannot do it: since Python 3.12 it profiles every thread.)
# Also samples the swap workers (inside process_and_emit_task), reported separately.
_DETECT_PROF: dict = {"on": os.environ.get("VMF_PROFILE_DETECTOR") == "1", "threads": {}, "data": {},
                      "sampler": None}


def _prof_enter(label: str):
    if not _DETECT_PROF["on"]:
        return
    _DETECT_PROF["threads"][threading.get_ident()] = label
    if _DETECT_PROF["sampler"] is None:
        _DETECT_PROF["sampler"] = threading.Thread(target=_detector_sampler, name="vmf-detector-sampler",
                                                   daemon=True)
        _DETECT_PROF["sampler"].start()


def _prof_exit():
    if _DETECT_PROF["on"]:
        _DETECT_PROF["threads"].pop(threading.get_ident(), None)


def _detector_sampler():
    while True:
        time.sleep(0.001)
        threads = dict(_DETECT_PROF["threads"])
        if not threads:
            continue
        frames = sys._current_frames()
        for tid, label in threads.items():
            frame = frames.get(tid)
            if frame is None:
                continue
            d = _DETECT_PROF["data"].setdefault(label, {"self": {}, "incl": {}, "samples": 0})
            seen = set()
            first = True
            while frame is not None:
                code = frame.f_code
                key = f"{Path(code.co_filename).name}:{code.co_firstlineno} {code.co_name}"
                if first:
                    k2 = f"{key} @line {frame.f_lineno}"
                    d["self"][k2] = d["self"].get(k2, 0) + 1
                    first = False
                if key not in seen:
                    seen.add(key)
                    d["incl"][key] = d["incl"].get(key, 0) + 1
                frame = frame.f_back
            d["samples"] += 1


def _detector_profile_dump() -> str | None:
    data, _DETECT_PROF["data"] = _DETECT_PROF["data"], {}
    if not data:
        return None
    lines = []
    for label, d in sorted(data.items()):
        n = d["samples"]
        lines += [f"===== {label}: {n} samples (~1 ms apart, all threads of this kind together)", "",
                  "INCLUSIVE (function on the stack):"]
        for k, v in sorted(d["incl"].items(), key=lambda x: -x[1])[:45]:
            lines.append(f"{100 * v / n:6.1f}%  {k}")
        lines += ["", "SELF (innermost Python frame, i.e. where the time is spent, C calls included):"]
        for k, v in sorted(d["self"].items(), key=lambda x: -x[1])[:40]:
            lines.append(f"{100 * v / n:6.1f}%  {k}")
        lines.append("")
    out = Path(__file__).resolve().parent / "runtime" / "detector_profile.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out)


_DETECT_MAX = 100000
_OUTPUTS: list = []  # (time, path) returned by VisoMaster's get_output_file_path


def _install_output_hook():
    """Remember the output paths VisoMaster computes (in memory only, like the other hooks):
    the final file name is never stored on the processor."""
    try:
        from app.helpers import miscellaneous as misc
    except Exception:
        return
    fn = getattr(misc, "get_output_file_path", None)
    if fn is None or getattr(fn, "_vmf_wrapped", False):
        return

    def get_output_file_path(*a, **k):
        path = fn(*a, **k)
        try:
            _OUTPUTS.append((time.time(), str(path)))
            del _OUTPUTS[:-20]
        except Exception:
            pass
        return path

    get_output_file_path._vmf_wrapped = True
    get_output_file_path.__wrapped__ = fn
    misc.get_output_file_path = get_output_file_path


def _rec_progress(vp, rec) -> dict:
    written = int(getattr(vp, "frames_written", 0) or 0)
    total = rec.get("total_frames") or 0
    elapsed = time.time() - rec["started_at"]
    out = {"frames_written": written, "total_frames": total or None, "elapsed_s": round(elapsed, 1)}
    if total:
        out["percent"] = round(min(100.0, 100.0 * written / total), 1)
    if written and elapsed > 0:
        rate = written / elapsed
        out["processing_fps"] = round(rate, 2)
        if total and rate > 0:
            out["eta_s"] = round(max(0, total - written) / rate)
    return out


def _rec_output(since: float) -> str | None:
    """Output file of the recording that started at `since` (incomplete suffix included)."""
    for t, p in reversed(_OUTPUTS):
        if t < since:
            break
        cand = Path(p)
        inc = cand.with_name(f"{cand.stem}_incomplete{cand.suffix}")
        if inc.is_file() and not cand.is_file():
            return str(inc)
        return p
    return None


def _install_rec_hooks(mw):
    """Output path hook + end detection right when VisoMaster stops processing, so a result is
    built even if nobody polls between two recordings."""
    _install_output_hook()
    _install_load_hooks(mw)
    vp = getattr(mw, "video_processor", None)
    if vp is None or getattr(vp, "_vmf_rec_hooked", False):
        return
    try:
        vp.processing_stopped_signal.connect(lambda *a: _rec_update_safe(mw))
        vp._vmf_rec_hooked = True
    except Exception:
        pass
    # Mark the end of frame processing: after it, VisoMaster closes the encoder, merges the audio and
    # saves the file (on the UI thread, which freezes the window meanwhile).
    for name in ("_finalize_default_style_recording", "finalize_segment_concatenation"):
        fn = getattr(vp, name, None)
        if fn is not None and not getattr(fn, "_vmf_wrapped", False):
            setattr(vp, name, _wrap_finalize(fn, mw))
    det = getattr(vp, "sequential_detector", None)
    fn = getattr(det, "run", None)
    if fn is not None and not getattr(fn, "_vmf_wrapped", False):
        det.run = _wrap_detector(fn)
    fn = getattr(det, "precompute", None)
    if fn is not None and not getattr(fn, "_vmf_wrapped", False):
        det.precompute = _wrap_detector(fn, _PREFETCH, "detector prefetch threads")
    mp = getattr(vp, "media_pipeline", None)
    fn = getattr(mp, "store_frame_to_display", None)
    if _DETECT_LOG["on"] and fn is not None and not getattr(fn, "_vmf_wrapped", False):
        mp.store_frame_to_display = _wrap_store_frame(fn)
    enc = getattr(vp, "encoder", None)
    fn = getattr(enc, "write_frame", None)
    if _DETECT_LOG["on"] and fn is not None and not getattr(fn, "_vmf_wrapped", False):
        enc.write_frame = _wrap_write_frame(fn)
    if _GIL_PROBE["on"] and _GIL_PROBE["thread"] is None:
        _GIL_PROBE["thread"] = threading.Thread(target=_gil_probe_loop, name="vmf-gil-probe", daemon=True)
        _GIL_PROBE["thread"].start()
    fwm = sys.modules.get("app.processors.workers.frame_worker")
    cls = getattr(fwm, "FrameWorker", None)
    fn = getattr(cls, "process_and_emit_task", None)
    if fn is not None and not getattr(fn, "_vmf_wrapped", False):
        cls.process_and_emit_task = _wrap_worker_frame(fn)
    fw = getattr(mw, "function_worker", None)
    fn = getattr(fw, "run_ort_with_iobinding", None)
    if fn is not None and not getattr(fn, "_vmf_wrapped", False):
        fw.run_ort_with_iobinding = _wrap_ort_run(fn, mw)
    try:
        # VisoMaster refreshes its VRAM bar every 5 s while processing: keep the peak.
        mw.gpu_memory_update_signal.connect(lambda used, total: _rec_vram_sample(used))
    except Exception:
        pass


def _wrap_finalize(fn, mw=None):
    def finalize(*a, **k):
        try:
            if _REC.get("active"):
                _REC["finalize_started"] = time.perf_counter()  # same clock as VisoMaster's start/end_time
        except Exception:
            pass
        _FINALIZING["on"] = True
        try:
            return fn(*a, **k)
        finally:
            _FINALIZING["on"] = False
            if mw is not None:
                # processing_stopped_signal fired while finalizing (still "recording"): build the result now.
                try:
                    _rec_update(mw, install_hooks=False)
                except Exception:
                    pass

    finalize._vmf_wrapped = True
    finalize.__wrapped__ = fn
    return finalize


def _wrap_detector(fn, store: list = _DETECT, label: str = "detector thread"):
    def run(*a, **k):
        t, c = time.perf_counter(), time.thread_time()
        _DETECT_TLS.models = []
        _prof_enter(label)
        try:
            out = fn(*a, **k)
            if _DETECT_LOG["on"] and store is _DETECT:
                _detect_log_add(k, out)
            return out
        finally:
            _prof_exit()
            models = tuple(_DETECT_TLS.models)
            _DETECT_TLS.models = None
            store.append((t, time.perf_counter() - t, time.thread_time() - c, models))
            if len(store) > _DETECT_MAX:
                del store[: len(store) - _DETECT_MAX]

    run._vmf_wrapped = True
    run.__wrapped__ = fn
    return run


def _wrap_worker_frame(fn):
    def process_and_emit_task(self, *a, **k):
        t, c = time.perf_counter(), time.thread_time()
        _prof_enter("swap workers")
        try:
            return fn(self, *a, **k)
        finally:
            _prof_exit()
            _WORKERS.append((t, time.perf_counter() - t, time.thread_time() - c))
            if len(_WORKERS) > _DETECT_MAX:
                del _WORKERS[: len(_WORKERS) - _DETECT_MAX]

    process_and_emit_task._vmf_wrapped = True
    process_and_emit_task.__wrapped__ = fn
    return process_and_emit_task


def _workers_stats(t0: float, t1: float, det: dict | None, frames_time: float | None) -> dict | None:
    entries = [e for e in list(_WORKERS) if t0 <= e[0] <= t1]
    if not entries:
        return None
    n = len(entries)
    wall = sum(e[1] for e in entries)
    cpu = sum(e[2] for e in entries)
    res = {"frames": n, "avg_ms": round(wall / n * 1000, 1), "thread_cpu_ms": round(cpu / n * 1000, 1)}
    if frames_time and det:
        det_cpu = det["breakdown_ms_per_frame"]["thread_cpu"] * det["frames"] / 1000
        # CPU time of the detector + all swap workers per second of frame processing: near 1.0 means
        # one core is busy all along, the signature of a pipeline limited by Python's GIL.
        res["detector_plus_workers_cpu_per_s"] = round((cpu + det_cpu) / frames_time, 2)
    return res


def _wrap_ort_run(fn, mw):
    """Times the ONNX Runtime runs made by the detector thread (GPU work and the waits around it:
    stream sync, session lock, output sync); other threads only pay one attribute lookup."""
    def run_ort(session, io_binding, *a, **k):
        models = getattr(_DETECT_TLS, "models", None)
        if models is None:
            return fn(session, io_binding, *a, **k)
        t = time.perf_counter()
        try:
            return fn(session, io_binding, *a, **k)
        finally:
            name = "?"
            try:
                for n, m in mw.models_processor.models.items():
                    if m is session:
                        name = n
                        break
            except Exception:
                pass
            models.append((name, time.perf_counter() - t))

    run_ort._vmf_wrapped = True
    run_ort.__wrapped__ = fn
    return run_ort


def _detector_stats(t0: float, t1: float, frames_time: float | None, store: list = _DETECT) -> dict | None:
    """Time VisoMaster's sequential detector spent between t0 and t1 (perf_counter clock)."""
    entries = [e for e in list(store) if t0 <= e[0] <= t1]
    if not entries:
        return None
    calls = [e[1] for e in entries]
    n = len(calls)
    busy = sum(calls)
    res = {
        "frames": n,
        "time_s": round(busy, 2),
        "avg_ms": round(busy / n * 1000, 1),
        "max_ms": round(max(calls) * 1000, 1),
    }
    if frames_time:
        res["share_of_frames_time"] = round(busy / frames_time, 2)
    cpu = sum(e[2] for e in entries)
    per_model: dict = {}
    for e in entries:
        for name, d in e[3]:
            m = per_model.setdefault(name, [0, 0.0])
            m[0] += 1
            m[1] += d
    onnx = sum(m[1] for m in per_model.values())
    # Per frame: onnx = model runs (GPU + the syncs and locks around them); python = the rest of
    # run() (pre/post-processing); cpu = time the thread really ran on a core. python - cpu (roughly)
    # = waiting for the GIL behind the swap workers.
    res["breakdown_ms_per_frame"] = {
        "onnx_runs": round(onnx / n * 1000, 1),
        "python": round((busy - onnx) / n * 1000, 1),
        "thread_cpu": round(cpu / n * 1000, 1),
        "not_on_cpu": round((busy - cpu) / n * 1000, 1),
    }
    if per_model:
        res["models"] = {
            k: {"calls": v[0], "avg_ms": round(v[1] / v[0] * 1000, 1)} for k, v in sorted(per_model.items())
        }
    res["note"] = (
        "face detection runs in ONE thread for every frame, before the swap workers: time_s is a floor "
        "for frames_time_s whatever the thread count (share near 1 = detection is the limit)"
    )
    if store is _PREFETCH:
        res["note"] = (
            "stateless part of the detection (detector model + embeddings), run ahead by several threads "
            "in parallel: time_s adds up all of them, so it is not a floor for frames_time_s"
        )
    return res


def _rec_vram_sample(used):
    try:
        if _REC.get("active"):
            _REC["vram_peak_mb"] = max(int(used), int(_REC.get("vram_peak_mb") or 0))
    except Exception:
        pass


def _rec_update_safe(mw):
    try:
        _rec_update(mw)
    except Exception:
        pass  # never break VisoMaster's own stop handling


_REC_LOCK = threading.RLock()


def _rec_update(mw, install_hooks: bool = True) -> dict | None:
    """Starts tracking a recording (even one started from the UI), detects its end and builds
    the final result. Returns the current/last record, or None if there never was one.
    install_hooks=False: called outside the Qt thread (plain reads only)."""
    with _REC_LOCK:
        return _rec_update_locked(mw, install_hooks)


def _rec_start_time(vp, written: int) -> float:
    """Wall-clock start of a recording first seen while it runs (started from VisoMaster's UI)."""
    t0 = getattr(vp, "start_time", None)  # VisoMaster's perf_counter() at the start of processing
    try:
        ago = time.perf_counter() - float(t0)
        if 0 <= ago < 7 * 86400:
            return time.time() - ago
    except (TypeError, ValueError):
        pass
    # Fallback, too late when processing is slower than real time: frames written at the media fps.
    return time.time() - (written / float(vp.fps) if written and getattr(vp, "fps", 0) else 0)


def _rec_update_locked(mw, install_hooks: bool) -> dict | None:
    if install_hooks:
        _install_rec_hooks(mw)
    vp = getattr(mw, "video_processor", None)
    if vp is None:
        return _REC or None
    active = _recording(mw)
    vp_start = getattr(vp, "start_time", None)
    if active and _REC.get("active") and _REC.get("vp_start") not in (None, vp_start):
        # A new recording started before the end of the previous one was seen: close the old record first
        # (with its last progress, frames_written already belongs to the new recording).
        _rec_finish(mw, vp, int((_REC.get("progress") or {}).get("frames_written", 0) or 0))
    if active and not _REC.get("active"):
        written = int(getattr(vp, "frames_written", 0) or 0)
        start = max(0, int(getattr(vp, "next_frame_to_display", 0) or 0) - written)
        total = max(0, int(getattr(vp, "max_frame_number", 0) or 0) + 1 - start)
        _REC.clear()
        _REC.update(
            active=True,
            id=int(time.time() * 1000),
            started_at=_rec_start_time(vp, written),
            started_by=_REC_STARTER.pop("by", "visomaster ui"),
            media=Path(str(getattr(vp, "media_path", "") or "")).name or None,
            start_frame=start,
            total_frames=total or None,
            fps=round(float(getattr(vp, "fps", 0) or 0), 3) or None,
            output_folder=str(mw.control.get("OutputMediaFolder", "") or "") or None,
            swap_enabled=_swap_state(mw)["enabled"],
            lazy_at_start=_lazy_pending(mw),
            vp_start=vp_start,
        )
        v = _vram(mw)
        if v:
            _REC["vram_start_mb"] = _REC["vram_peak_mb"] = v["used_mb"]
    if not _REC:
        return None
    if _REC.get("active"):
        if active:
            _REC["progress"] = _rec_progress(vp, _REC)
            if _FINALIZING["on"]:
                _REC["progress"]["phase"] = "finalizing: closing the encoder, audio merge, saving the file"
        else:
            _rec_finish(mw, vp)
    return _REC


def _rec_finish(mw, vp, written: int | None = None) -> None:
    """Result of the tracked recording, which has ended."""
    # Ended: VisoMaster finalizes (audio merge) on the UI thread, so it is done by now.
    prog = _REC.get("progress") or {}
    if written is None:
        written = int(getattr(vp, "frames_written", 0) or prog.get("frames_written", 0) or 0)
    out = _rec_output(_REC["started_at"] - 1)
    exists = bool(out and Path(out).is_file())
    total = _REC.get("total_frames") or 0
    if not exists:
        state = "failed"
    elif _REC.get("stop_requested") or (total and written < total - 1):
        state = "stopped"
    else:
        state = "finished"
    _REC["active"] = False
    _REC["result"] = {
        "state": state,
        "output_path": out,
        "exists": exists,
        "size_bytes": Path(out).stat().st_size if exists else None,
        "incomplete": bool(out and "_incomplete" in Path(out).stem),
        "frames_written": written,
        "duration_s": round(written / _REC["fps"], 2) if _REC.get("fps") and written else None,
        "elapsed_s": round(time.time() - _REC["started_at"], 1),
    }
    if not exists:
        _REC["result"]["hint"] = "no output file found: see the VisoMaster console for the error"
    # VisoMaster's own measure, the one it prints as "Processing completed in ..." (from the
    # start of processing to the end of finalization, audio merge included).
    t0, t1 = getattr(vp, "start_time", None), getattr(vp, "end_time", None)
    if isinstance(t0, (int, float)) and isinstance(t1, (int, float)) and t1 > t0:
        proc = t1 - t0
        _REC["result"]["processing_time_s"] = round(proc, 2)
        if written:
            _REC["result"]["average_fps"] = round(written / proc, 2)
        fin = _REC.get("finalize_started")
        if isinstance(fin, (int, float)) and t0 <= fin <= t1:
            # processing_time_s = frames_time_s + finalize_s
            _REC["result"]["frames_time_s"] = round(fin - t0, 2)
            _REC["result"]["finalize_s"] = round(t1 - fin, 2)
            _REC["result"]["finalize_note"] = (
                "after the last frame: closing the encoder, audio merge, saving the file "
                "(VisoMaster's window is frozen meanwhile)"
            )
        det = _detector_stats(t0, t1, _REC["result"].get("frames_time_s") or proc)
        if det:
            _REC["result"]["detector_thread"] = det
        pre = _detector_stats(t0, t1, _REC["result"].get("frames_time_s") or proc, _PREFETCH)
        if pre:
            _REC["result"]["detector_prefetch"] = pre
        wk = _workers_stats(t0, t1, det, _REC["result"].get("frames_time_s"))
        if wk:
            _REC["result"]["swap_workers"] = wk
        if _GIL_PROBE["on"]:
            fin = _REC.get("finalize_started") or t1
            _REC["result"]["gil_probe"] = {
                "while_processing": _gil_probe_stats(t0, fin),
                "before_recording": _gil_probe_stats(t0 - 6, t0 - 1),
            }
        if _DETECT_LOG["on"]:
            try:
                _REC["result"]["detection_log"] = _detect_log_dump(_REC.get("id"))
            except Exception as e:
                _REC["result"]["detection_log"] = f"failed: {e}"
        if _DETECT_PROF["on"]:
            try:
                _REC["result"]["detector_profile"] = _detector_profile_dump()
            except Exception as e:
                _REC["result"]["detector_profile"] = f"failed: {e}"
    loads = _loads_between(_REC["started_at"], time.time())
    if loads:
        _REC["result"]["models_loaded_during_recording"] = loads
        _REC["result"]["note"] = (
            "models were loaded during this recording, so its time includes loading "
            "(wait_ready before start_recording avoids it)"
        )
    compiled = sorted(set(_REC.get("lazy_at_start") or []) - set(_lazy_pending(mw)))
    engines = _engines_since(_REC["started_at"])
    builds = _builds_between(_REC["started_at"], time.time())
    if compiled or engines or builds:
        # builds: each TensorRT step with its duration (e.g. the lazy build's compile time)
        _REC["result"]["tensorrt_compiled_during_recording"] = {
            "models": compiled, "engine_files": engines, "builds": builds,
            "build_time_s": round(sum(b["seconds"] for b in builds), 1),
        }
        _REC["result"]["note"] = (
            "a TensorRT engine was compiled during this recording (first use of a model whose engine "
            "cannot be built in advance): its time includes the compilation. The engine exists now: "
            "record again once for a clean measure"
        )
    warnings = _console_since(_REC["started_at"] - 1)
    if warnings:
        # e.g. "Tail-drain stalled for too long (8.0s)": VisoMaster waited for a frame at the end
        _REC["result"]["console_warnings"] = warnings
        if any("stalled" in w.lower() for w in warnings):
            _REC["result"]["stall_note"] = (
                "VisoMaster reported a stall during this recording (see console_warnings): its time "
                "includes that wait, record again for a clean measure"
            )
    v = _vram(mw)
    end = v["used_mb"] if v else None
    samples = [x for x in (_REC.get("vram_start_mb"), _REC.get("vram_peak_mb"), end) if x is not None]
    _REC["result"]["vram_mb"] = {
        "start": _REC.get("vram_start_mb"),
        "peak": max(samples) if samples else None,
        "end": end,
        "note": "whole GPU; peak sampled by VisoMaster every 5 s while processing",
    }


_REC_STARTER: dict = {}


def _rec_view(rec: dict | None) -> dict | None:
    if not rec:
        return None
    keys = ("id", "active", "started_by", "media", "start_frame", "total_frames", "fps", "output_folder",
            "swap_enabled", "progress", "result")
    return {k: rec[k] for k in keys if k in rec}


def _scan_active(mw) -> bool:
    try:
        return bool(_vca().is_issue_scan_active(mw))
    except Exception:
        return getattr(mw, "scan_issue_worker", None) is not None


def _trt_building(mw) -> str | None:
    job = _TRT_JOB
    if job is not None and job.get("running"):
        return f"TensorRT cache build in progress ({job.get('current')})"
    dlg = getattr(mw, "build_progress_dialog", None)
    try:
        if dlg is not None and dlg.isVisible():
            return "VisoMaster is building a TensorRT engine"
    except Exception:
        pass
    return None


def busy_reason(mw) -> str | None:
    trt = _trt_building(mw)
    if trt:
        return trt
    if _recording(mw):
        return "recording in progress"
    if _scan_active(mw):
        return "issue scan in progress"
    if _job_running(mw):
        return "job queue running"
    if getattr(mw, "is_batch_processing", False) or _BATCH.get("active"):
        return "batch processing in progress"
    return None


def _plain(d) -> dict:
    if d is None:
        return {}
    data = getattr(d, "data", d)
    return dict(data)


def _defaults(mw) -> dict:
    return _plain(getattr(mw, "default_parameters", None))


def _face_params(mw, face_id) -> dict:
    """Effective parameters of a face (ParametersDict falls back on defaults)."""
    eff = _defaults(mw)
    if face_id is None:
        eff.update(_plain(getattr(mw, "current_widget_parameters", None)))
    else:
        if face_id not in (getattr(mw, "parameters", {}) or {}):
            raise CommandError(f"face '{face_id}' has no parameters")
        eff.update(_plain(mw.parameters[face_id]))
    return eff


def _selected_face(mw):
    fid = getattr(mw, "selected_target_face_id", None)
    return fid if fid and fid in (getattr(mw, "target_faces", {}) or {}) else None


def _resolve_faces(mw, faces) -> tuple[list, bool]:
    """Return (face ids, also_update_current_widget_parameters)."""
    targets = list((getattr(mw, "target_faces", {}) or {}).keys())
    faces = faces or "selected"
    if faces == "all":
        return targets, True
    if faces == "selected":
        sel = _selected_face(mw)
        return ([sel], True) if sel else ([], True)
    if faces in targets:
        return [faces], faces == _selected_face(mw)
    raise CommandError(f"unknown face '{faces}' (available: {targets})")


def _git_info() -> dict:
    info = {"branch": None, "commit": None}
    git = common = APP_DIR / ".git"
    try:
        if git.is_file():  # a worktree: ".git" holds "gitdir: <repo>/.git/worktrees/<name>"
            text = git.read_text(encoding="utf-8").strip()
            git = common = Path(text[len("gitdir:"):].strip()) if text.startswith("gitdir:") else git
            if (git / "commondir").is_file():  # refs and packed-refs live in the main .git
                common = (git / (git / "commondir").read_text(encoding="utf-8").strip()).resolve()
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head[5:]
            info["branch"] = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
            git = common
            ref_file = git / ref
            if ref_file.is_file():
                info["commit"] = ref_file.read_text(encoding="utf-8").strip()
            else:
                packed = git / "packed-refs"
                if packed.is_file():
                    for line in packed.read_text(encoding="utf-8").splitlines():
                        if line.endswith(" " + ref):
                            info["commit"] = line.split(" ", 1)[0]
        else:
            info["commit"] = head
    except OSError:
        pass
    return info


def _version() -> str | None:
    try:
        return json.loads((APP_DIR / "version.json").read_text(encoding="utf-8")).get("version")
    except Exception:
        return None


def _abs(path: str | None) -> Path | None:
    if not path:
        return None
    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()


def _out_path(path: str | None, suffix: str, allowed: tuple = ()) -> Path | None:
    """A file the API writes where the caller says: only with the expected extension (a mistaken or malicious
    path cannot replace a document, a script, a DLL...), and never inside VisoMaster's or vmf_api's folders
    (presets, embeddings, personal_keys.json...), except the `allowed` paths."""
    p = _abs(path)
    if p is None:
        return None
    if p.suffix.lower() != suffix:
        raise CommandError(f"the output file must end with {suffix}: {p}")
    if p not in [Path(a).resolve() for a in allowed]:
        for root in (APP_DIR, VMF_DIR):
            if p.is_relative_to(root.resolve()):
                raise CommandError(f"refused: {p} is inside {root} (VisoMaster's or vmf_api's own files): "
                                   f"choose another folder, or leave the path out for the default")
    return p


OVERWRITTEN_DIR = BACKUP_DIR / "overwritten"


def _keep_previous(p: Path) -> str | None:
    """Copy of a file about to be overwritten, in backups/overwritten (the 20 most recent are kept)."""
    if not p.is_file():
        return None
    OVERWRITTEN_DIR.mkdir(parents=True, exist_ok=True)
    keep, n = OVERWRITTEN_DIR / f"{_stamp()}-{p.name}", 1
    while keep.exists():
        n += 1
        keep = OVERWRITTEN_DIR / f"{_stamp()}.{n}-{p.name}"
    shutil.copy2(p, keep)
    _prune(OVERWRITTEN_DIR, "*", 20)
    return str(keep)


def _prune(folder: Path, pattern: str, keep: int) -> None:
    """Delete all but the `keep` most recent files of folder/pattern (backups the API itself wrote)."""
    try:
        files = sorted(folder.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
        for f in files[keep:]:
            if f.is_file():
                f.unlink()
    except OSError:
        pass


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _refresh(mw):
    try:
        _common().refresh_frame(mw)
    except Exception:
        pass


def _mark_feeder_dirty(mw, face_ids=(), control_keys=()):
    """Same thing VisoMaster does when parameters change during playback."""
    vp = getattr(mw, "video_processor", None)
    if vp is None or not getattr(vp, "processing", False):
        return
    try:
        vp.ui_state_is_dirty = True
        with vp.state_lock:
            fp = getattr(vp, "feeder_parameters", None)
            if fp:
                for fid in face_ids:
                    if fid in fp:
                        fp[fid] = copy.deepcopy(mw.parameters[fid])
            fc = getattr(vp, "feeder_control", None)
            if fc:
                for k in control_keys:
                    if k in fc:
                        fc[k] = mw.control[k]
    except Exception:
        pass


def _widget_defs() -> dict:
    try:
        return dict(_common()._ALL_CONTROL_WIDGET_OPTIONS)
    except Exception:
        return {}


def _set_widget(mw, key, value):
    widget = (getattr(mw, "parameter_widgets", {}) or {}).get(key)
    if widget is None:
        return
    prev = getattr(widget, "enable_refresh_frame", True)
    widget.enable_refresh_frame = False
    try:
        helper = getattr(_common(), "_set_single_widget_value", None)
        if helper is not None:
            helper(widget, value)
        else:
            widget.set_value(value)
    finally:
        widget.enable_refresh_frame = prev


def _apply_control_updates(mw, updates: dict) -> list:
    """Write control values like VisoMaster does (widget + exec_function), only for changed keys."""
    errors = []
    defs = _widget_defs()
    for k, v in updates.items():
        mw.control[k] = v
    for k, v in updates.items():
        try:
            _set_widget(mw, k, v)
            if k == "OutputMediaFolder":  # no widget of its own: VisoMaster's batch and Save Image check this field
                line = getattr(mw, "outputFolderLineEdit", None)
                if line is not None:
                    line.setText(str(v or ""))
            d = defs.get(k) or {}
            fn = d.get("exec_function")
            if fn:
                fn(mw, v, *list(d.get("exec_function_args", [])))
        except Exception as e:
            errors.append(f"{k}: {type(e).__name__}: {e}")
    _mark_feeder_dirty(mw, control_keys=list(updates))
    return errors


_PARAM_SPECS: dict | None = None


def _param_specs() -> dict:
    """key -> live layout spec of the face parameters (exec_function, buttons...)."""
    global _PARAM_SPECS
    if _PARAM_SPECS is None:
        specs: dict = {}
        try:
            from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
            from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA
            from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA

            for data in (COMMON_LAYOUT_DATA, SWAPPER_LAYOUT_DATA, FACE_EDITOR_LAYOUT_DATA):
                for widgets in (data or {}).values():
                    if isinstance(widgets, dict):
                        specs.update({k: v for k, v in widgets.items() if isinstance(v, dict)})
        except Exception:
            pass
        _PARAM_SPECS = specs
    return _PARAM_SPECS


def _shown_params(mw) -> dict:
    """Parameters shown in the panel: the selected face's, or the no-face panel's."""
    sel = _selected_face(mw)
    if sel:
        return _plain((getattr(mw, "parameters", {}) or {}).get(sel))
    return _plain(getattr(mw, "current_widget_parameters", None))


def _run_param_exec(mw, old: dict, new: dict) -> list:
    """Run the exec_function of each changed parameter, like VisoMaster's update_parameter does when
    a widget changes (loads/unloads models: masks, expression restorer, re-aging...)."""
    errors = []
    specs = _param_specs()
    for k, v in new.items():
        spec = specs.get(k) or {}
        fn = spec.get("exec_function")
        if not fn or schema_mod.values_equal(old.get(k), v):
            continue
        try:
            fn(mw, v, *list(spec.get("exec_function_args", []) or []))
        except Exception as e:
            errors.append(f"{k}: {type(e).__name__}: {e}")
    return errors


def _apply_param_updates(mw, face_ids: list, updates_by_face: dict, cwp_updates: dict | None) -> list:
    """Write face parameters like VisoMaster does (data, widgets, feeder, exec_function of the changed
    keys of the shown panel). Returns the exec_function errors."""
    from app.helpers.miscellaneous import ParametersDict

    shown_before = _shown_params(mw)
    defaults = _defaults(mw)
    for fid in face_ids:
        upd = updates_by_face.get(fid) or {}
        if not upd:
            continue
        new = _plain(mw.parameters.get(fid))
        new.update(upd)
        mw.parameters[fid] = ParametersDict(new, defaults)
    if cwp_updates:
        new = _plain(getattr(mw, "current_widget_parameters", None))
        new.update(cwp_updates)
        mw.current_widget_parameters = ParametersDict(new, defaults)
    sel = _selected_face(mw)
    if sel and sel in face_ids:
        # VisoMaster keeps a copy of the selected face's params as the panel state.
        mw.current_widget_parameters = mw.parameters[sel].copy()
    _mark_feeder_dirty(mw, face_ids=face_ids)
    _common().set_widgets_values_using_face_id_parameters(mw, sel)
    return _run_param_exec(mw, shown_before, _shown_params(mw))


def _snapshot_state(mw) -> dict:
    return {
        "_vmf_backup": {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "version": _version(), **_git_info()},
        "control": copy.deepcopy(dict(mw.control)),
        "faces": {fid: _plain(p) for fid, p in (getattr(mw, "parameters", {}) or {}).items()},
        "current_widget_parameters": _plain(getattr(mw, "current_widget_parameters", None)),
        "selected_target_face_id": _selected_face(mw),
    }


def _backup_path(label: str) -> Path:
    """BACKUP_DIR/<time>-<label>.json, never an existing file: two backups in the same second
    (apply_config twice...) must not overwrite the first one, which may hold the original state."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    p, n = BACKUP_DIR / f"{stamp}-{label}.json", 1
    while p.exists():
        n += 1
        p = BACKUP_DIR / f"{stamp}.{n}-{label}.json"  # sorts after the first one: list order stays by time
    return p


def _write_backup(mw, label: str) -> str:
    p = _backup_path(label)
    write_json_atomic(p, _snapshot_state(mw))
    return p.name


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------


@command("status")
def cmd_status(ctx):
    """VisoMaster version, branch/commit, recording state, selected face."""
    mw = ctx.mw
    vp = getattr(mw, "video_processor", None)
    return {
        "api_version": API_VERSION,
        "visomaster_version": _version(),
        **_git_info(),
        "pid": os.getpid(),
        "port": ctx.api_port,
        "recording": _recording(mw),
        "processing": bool(getattr(vp, "processing", False)),
        "scan_active": _scan_active(mw),
        "job_queue_running": _job_running(mw),
        "tensorrt_building": _trt_building(mw),
        "provider": getattr(getattr(mw, "models_processor", None), "provider_name", None),
        "busy": busy_reason(mw),
        "ready": _readiness(mw),
        "media": {
            "type": getattr(vp, "file_type", None),
            "path": str(getattr(vp, "media_path", "") or "") or None,
            "frame": getattr(vp, "current_frame_number", None),
        },
        "selected_target_face_id": _selected_face(mw),
        "target_faces": list((getattr(mw, "target_faces", {}) or {}).keys()),
        "input_faces": len(getattr(mw, "input_faces", {}) or {}),
        "target_media": len(getattr(mw, "target_videos", {}) or {}),
        "output_folder": mw.control.get("OutputMediaFolder"),
        "swap": _swap_state(mw),
        "vram": _vram(mw),
        # current or last recording: progress, then result (output file, size, duration)
        "recording_info": _rec_view(_rec_update(mw)),
        "batch_info": _batch_view(),  # current or last batch (run_action batch...Button)
    }


@command("ready_status", main_thread=False)
def cmd_ready_status(ctx):
    """Is VisoMaster idle and ready (no model loading, no preview frame or playback in progress)?
    Runs outside VisoMaster's UI thread."""
    _install_load_hooks(ctx.mw)
    return _readiness(ctx.mw)


@command("recording_status", main_thread=False)
def cmd_recording_status(ctx):
    """Current or last recording: progress (frames, %, fps, ETA) then final result (output file).
    Runs outside VisoMaster's UI thread so that polling never disturbs a recording."""
    rec = _rec_view(_rec_update(ctx.mw, install_hooks=False))
    batch = _batch_view()
    if batch is not None:
        rec_t0 = _REC.get("started_at") or 0
        # A batch with videos records each one: those recordings belong to the batch, which is reported
        # until it ends (else wait_recording would return after its first video).
        if batch["active"] or rec_t0 <= batch["started_at"] or rec_t0 <= (_BATCH.get("finished_at") or 0):
            if batch["active"] and rec is not None and rec.get("active"):
                batch["current_video"] = rec.get("progress")
            return batch
    if rec is None:
        return {"active": False, "note": "no recording seen since VisoMaster started"}
    return rec


@command("get_config")
def cmd_get_config(ctx, face_id: str | None = None, keys: list | None = None):
    """control + parameters of the selected face (or face_id). keys = optional filter."""
    mw = ctx.mw
    fid = face_id or _selected_face(mw)
    params = _face_params(mw, fid)
    control = dict(mw.control)
    if keys:
        control = {k: v for k, v in control.items() if k in keys}
        params = {k: v for k, v in params.items() if k in keys}
    return {"face_id": fid, "control": control, "parameters": params}


@command("export_config")
def cmd_export_config(ctx, path: str | None = None, face_id: str | None = None, include_personal: bool = False):
    """Export the config in preset format ({parameters, control}), without personal keys."""
    mw = ctx.mw
    fid = face_id or _selected_face(mw)
    params = _face_params(mw, fid)
    control = dict(mw.control)
    removed_c: list = []
    removed_p: list = []
    if not include_personal:
        control, removed_c = merge.strip_personal(control, ctx.schema, ctx.rules)
        params, removed_p = merge.strip_personal(params, ctx.schema, ctx.rules)
    data = {
        "parameters": params,
        "control": control,
        "_vmf_export": {
            "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "visomaster_version": _version(),
            **_git_info(),
            "face_id": fid,
            "excluded_personal_keys": sorted(set(removed_c) | set(removed_p)),
        },
    }
    out = _out_path(path, ".json") or (EXPORT_DIR / f"config-{_stamp()}.json")
    kept = _keep_previous(out)
    write_json_atomic(out, data)
    res = {"path": str(out), "parameters": len(params), "control": len(control), "excluded": len(removed_c) + len(removed_p)}
    if kept:
        res["previous_kept"] = kept
    return res


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def _build_plans(ctx, path, faces, no_control, no_params, source_face, missing):
    mw = ctx.mw
    p = _abs(path)
    if p is None or not p.is_file():
        raise CommandError(f"file not found: {path}")
    if missing not in ("keep", "default"):
        raise CommandError("missing must be 'keep' or 'default'")
    try:
        imported = merge.parse_config(p, ctx.schema, source_face=source_face)
    except (ValueError, json.JSONDecodeError) as e:
        raise CommandError(f"unreadable file: {e}") from None
    rules = ctx.rules
    out = {"file": str(p), "format": imported.fmt, "notes": list(imported.notes)}
    control_plan = None
    if not no_control:
        if imported.control is None:
            out["control"] = "not in file"
        else:
            ctl_defaults = {k: e.default for k, e in ctx.schema.entries.items() if e.scope == "control"}
            control_plan = merge.plan_section(
                imported.control, dict(mw.control), "control", ctx.schema, rules, ctl_defaults, missing
            )
            out["control"] = control_plan.to_dict()
    param_plans: dict = {}
    face_ids: list = []
    cwp_plan = None
    if not no_params:
        if imported.parameters is None:
            out["parameters"] = "not in file"
        else:
            face_ids, with_cwp = _resolve_faces(mw, faces)
            defaults = _defaults(mw)
            for fid in face_ids:
                param_plans[fid] = merge.plan_section(
                    imported.parameters, _face_params(mw, fid), "parameter", ctx.schema, rules, defaults, missing
                )
            if with_cwp and not face_ids:
                cwp_plan = merge.plan_section(
                    imported.parameters, _face_params(mw, None), "parameter", ctx.schema, rules, defaults, missing
                )
            elif faces == "all":
                cwp_plan = merge.plan_section(
                    imported.parameters, _face_params(mw, None), "parameter", ctx.schema, rules, defaults, missing
                )
            out["parameters"] = {fid: pl.to_dict() for fid, pl in param_plans.items()}
            if cwp_plan is not None:
                out["parameters"]["<no-face panel>"] = cwp_plan.to_dict()
            if not face_ids:
                out["notes"].append("no target face: only the parameters panel (template for future faces) is affected")
    return out, control_plan, param_plans, face_ids, cwp_plan


@command("diff")
def cmd_diff(ctx, path: str, faces: str = "selected", no_control: bool = False, no_params: bool = False,
             source_face: str | None = None, missing: str = "keep"):
    """What would change: changed / excluded (personal) / unknown (obsolete) / invalid / missing."""
    out, *_ = _build_plans(ctx, path, faces, no_control, no_params, source_face, missing)
    return out


@command("apply_config", write=True, timeout=120)
def cmd_apply_config(ctx, path: str, faces: str = "selected", no_control: bool = False, no_params: bool = False,
                     source_face: str | None = None, missing: str = "keep", dry_run: bool = False):
    """Import a preset/workspace without touching personal keys. Backup first in vmf_api/backups."""
    mw = ctx.mw
    out, control_plan, param_plans, face_ids, cwp_plan = _build_plans(
        ctx, path, faces, no_control, no_params, source_face, missing
    )
    if dry_run:
        out["dry_run"] = True
        return out
    out["backup"] = _write_backup(mw, "before-apply")
    errors: list = []
    if control_plan is not None and control_plan.updates:
        errors += _apply_control_updates(mw, control_plan.updates)
    if param_plans or cwp_plan is not None:
        errors += _apply_param_updates(
            mw,
            face_ids,
            {fid: pl.updates for fid, pl in param_plans.items()},
            cwp_plan.updates if cwp_plan is not None else None,
        )
    _refresh(mw)
    out["applied"] = {
        "control": len(control_plan.updates) if control_plan else 0,
        "parameters": {fid: len(pl.updates) for fid, pl in param_plans.items()},
    }
    if cwp_plan is not None:
        out["applied"]["parameters"]["<no-face panel>"] = len(cwp_plan.updates)
    if errors:
        out["errors"] = errors
    return out


@command("list_backups")
def cmd_list_backups(ctx):
    """List backups (most recent first)."""
    if not BACKUP_DIR.is_dir():
        return []
    files = [(p.stat().st_mtime, p.name) for p in BACKUP_DIR.glob("*.json") if p.is_file()]
    return [n for _, n in sorted(files, reverse=True)]


WORKSPACE_BACKUPS = ("-workspace-before-load.json", "-workspace-before-save.json")  # whole workspaces


def _backup_label(name: str) -> str:
    label = re.sub(r"[^\w.-]+", "_", (name or "").strip()).strip("._")[:60]
    if not label:
        raise CommandError("give the backup a name (letters, digits, - _ .)")
    return label


@command("save_backup")
def cmd_save_backup(ctx, name: str):
    """Save the current config (control + every face's parameters) under a name, e.g. to come back to a
    baseline later with restore_backup(name). Nothing is changed in VisoMaster."""
    mw = ctx.mw
    label = _backup_label(name)
    file = _write_backup(mw, label)
    return {
        "saved": file,
        "name": label,
        "faces": list((getattr(mw, "parameters", {}) or {}).keys()),
        "restore_with": f'restore_backup(name="{label}")',
        "note": "covers settings and face parameters, not media, faces, embeddings or the swap button",
    }


def _find_backup(names: list, name: str) -> str | None:
    """Exact file name, or the most recent backup saved under that name."""
    if (BACKUP_DIR / Path(name).name).is_file():
        return Path(name).name
    try:
        label = _backup_label(name)
    except CommandError:
        return None
    # The label exactly ("apply" must not find "...-before-apply.json"), most recent first.
    pattern = re.compile(r"\d{8}-\d{6}(\.\d+)?-" + re.escape(label) + r"\.json")
    return next((n for n in names if pattern.fullmatch(n)), None)


@command("restore_backup", write=True, timeout=120)
def cmd_restore_backup(ctx, name: str | None = None):
    """Re-apply a backup: a file name from list_backups, or a name given to save_backup (its most
    recent save). Default: the most recent backup. The current state is backed up first."""
    mw = ctx.mw
    names = cmd_list_backups(ctx)
    if not names:
        raise CommandError("no backup")
    # Default: the latest backup written by the API (time-stamped name), never a whole workspace.
    configs = [n for n in names if re.match(r"\d{8}-\d{6}", n) and not n.endswith(WORKSPACE_BACKUPS)]
    found = _find_backup(names, name) if name else (configs or [None])[0]
    if found is None:
        raise CommandError(f"backup not found: {name} (see list_backups)" if name else "no config backup")
    if found.endswith(WORKSPACE_BACKUPS):
        raise CommandError(f"{found} is a whole workspace (kept by load_workspace / save_workspace), not a config backup: "
                           f"load it with load_workspace {BACKUP_DIR / found}")
    name = found
    p = BACKUP_DIR / name
    data = json.loads(p.read_text(encoding="utf-8"))
    current_backup = _write_backup(mw, "before-restore")
    control = {k: v for k, v in (data.get("control") or {}).items() if k in mw.control}
    skipped = _camera_on(control)  # privacy: never turned on by the API
    for k in skipped:
        del control[k]
    changed_c = {k: v for k, v in control.items() if not schema_mod.values_equal(mw.control.get(k), v)}
    errors = _apply_control_updates(mw, changed_c) if changed_c else []
    faces = {fid: v for fid, v in (data.get("faces") or {}).items() if fid in (mw.parameters or {})}
    from app.helpers.miscellaneous import ParametersDict

    shown_before = _shown_params(mw)
    for fid, params in faces.items():
        mw.parameters[fid] = ParametersDict(dict(params), _defaults(mw))
    cwp = data.get("current_widget_parameters")
    if isinstance(cwp, dict) and not _selected_face(mw):
        mw.current_widget_parameters = ParametersDict(dict(cwp), _defaults(mw))
    errors += _apply_param_updates(mw, list(faces), {}, None)
    errors += _run_param_exec(mw, shown_before, _shown_params(mw))
    _refresh(mw)
    res = {"restored": name, "control_changed": len(changed_c), "faces": list(faces), "backup_of_current": current_backup}
    if skipped:
        res["skipped_camera_toggles"] = skipped
    if errors:
        res["errors"] = errors
    return res


def _current_value(mw, key):
    if key in (getattr(mw, "control", {}) or {}):
        return mw.control.get(key)
    return _shown_params(mw).get(key)


def _unmet(mw, deps: dict | None) -> list:
    """Conditions of a schema depends_on that the current values do not meet."""
    if not deps:
        return []
    bad = [c for c in deps.get("all", []) if not schema_mod.values_equal(_current_value(mw, c["key"]), c["value"])]
    anyof = deps.get("any", [])
    if anyof and not any(schema_mod.values_equal(_current_value(mw, c["key"]), c["value"]) for c in anyof):
        bad.append({"any_of": anyof})
    return bad


@command("press_parameter_button", write=True, timeout=600)
def cmd_press_parameter_button(ctx, key: str, label: str | None = None, confirm: str | None = None):
    """Click a button attached to a setting (schema "buttons", e.g. "Apply" under Face Re-Aging's
    Target Age), exactly like in the UI. Its conditions (depends_on) must be met. A reset button
    needs confirm=<code> (the first call returns the code) and writes a config backup first."""
    from PySide6 import QtWidgets

    mw = ctx.mw
    e = ctx.schema.get(key)
    if e is None or not getattr(e, "buttons", None):
        raise CommandError(f"{key} has no button (see schema buttons)")
    labels = [b["label"] for b in e.buttons]
    want = label or labels[0]
    if want not in labels:
        raise CommandError(f"{key} has no button '{want}': {', '.join(labels)}")
    unmet = _unmet(mw, e.depends_on)
    if unmet:
        raise CommandError(f"{key}: conditions not met for its button: {json.dumps(unmet)}")
    widget = (getattr(mw, "parameter_widgets", {}) or {}).get(key)
    button = None
    for holder in (getattr(widget, "below_row_widget", None), getattr(widget, "row_widget", None)):
        if holder is None:
            continue
        for b in holder.findChildren(QtWidgets.QPushButton):
            if b.text().replace("&", "") == want:
                button = b
                break
        if button is not None:
            break
    if button is None:
        raise CommandError(f"button '{want}' of {key} not found in the UI")
    backup = None
    if _RESET_WORDS.search(want):
        ask = _action_confirmation(f"{key}:{want}", want, confirm, "press_parameter_button")
        if ask is not None:
            ask["warning"] = (f"'{want}' resets settings. A config backup is written first: restore_backup "
                              f"brings them back. To confirm: call press_parameter_button again with "
                              f'confirm="{ask["confirm"]}" (valid 5 min).')
            return ask
        backup = _write_backup(mw, "before-reset")
    button.click()
    res = {"pressed": want, "key": key}
    if backup:
        res["backup"] = backup
    help_text = next((b.get("help") for b in e.buttons if b["label"] == want), None)
    if help_text:
        res["what_it_does"] = help_text
    swap = _swap_state(mw)
    if swap.get("note"):
        res["warning"] = "VisoMaster may ignore it: " + swap["note"]
    return res


def _with_personal(ctx, key: str, res: dict) -> dict:
    """set works on personal keys too (only apply_config protects them): say so in the result."""
    reason = ctx.rules.reason(key, ctx.schema)
    res["personal"] = reason is not None
    if reason:
        res["personal_reason"] = reason
    return res


def _same_folder(key: str, old, new) -> bool:
    """C:/out and C:\\out are the same folder (the CLI / MCP send absolute Windows paths)."""
    if not (key.endswith("Folder") and isinstance(old, str) and isinstance(new, str) and old and new):
        return False
    return os.path.normcase(os.path.normpath(old)) == os.path.normcase(os.path.normpath(new))


@command("set", write=True)
def cmd_set(ctx, key: str, value: Any, face_id: str | None = None):
    """Set one key (validated against the schema: type, bounds, options)."""
    mw = ctx.mw
    # CLI/MCP input: "256" vs 256 vs "true" are all accepted, then typed by the schema.
    ok, coerced, msg = ctx.schema.validate(key, value, coerce_strings=True)
    if not ok:
        raise CommandError(msg)
    if coerced is True and _CAMERA_ON_KEYS.search(key) and key.endswith("Toggle"):
        # e.g. SendVirtCamFramesEnableToggle: its function starts the virtual camera (run_action refuses it too)
        raise CommandError(f"{key} turns on a webcam / the virtual camera (privacy): reserved to the user")
    if key in mw.control and ctx.schema.scope_of(key) != "parameter":
        old = mw.control[key]
        if schema_mod.values_equal(old, coerced) or _same_folder(key, old, coerced):
            # Re-applying runs the key's function anyway (a provider switch clears the VRAM and turns the swap off).
            return _with_personal(ctx, key, {"scope": "control", "key": key, "from": old, "to": coerced,
                                             "unchanged": True})
        errors = _apply_control_updates(mw, {key: coerced})
        _refresh(mw)
        res = {"scope": "control", "key": key, "from": old, "to": coerced, "ui_updated": not errors}
        if errors:
            res["errors"] = errors
        return _with_personal(ctx, key, res)
    if key in _defaults(mw):
        fid = face_id or _selected_face(mw)
        if face_id and face_id not in (mw.target_faces or {}):
            raise CommandError(f"unknown face '{face_id}'")
        old = _face_params(mw, fid).get(key)
        if schema_mod.values_equal(old, coerced):
            return _with_personal(ctx, key, {"scope": "parameter", "face_id": fid, "key": key, "from": old,
                                             "to": coerced, "unchanged": True})
        if fid:
            errors = _apply_param_updates(mw, [fid], {fid: {key: coerced}}, None)
        else:
            errors = _apply_param_updates(mw, [], {}, {key: coerced})
        _refresh(mw)
        shown = fid is None or fid == _selected_face(mw)
        res = {"scope": "parameter", "face_id": fid, "key": key, "from": old, "to": coerced, "ui_updated": shown}
        if errors:
            res["errors"] = errors
        if not shown:
            res["note"] = "stored for this face; the panel shows the selected face, so it is unchanged"
        return _with_personal(ctx, key, res)
    raise CommandError(f"key unknown to this version: {key}")


def _group_name(e) -> str:
    """Group of an entry; a layout without groups (Face Editor tab) is named after the layout."""
    if e.group:
        return e.group
    return " ".join(w.capitalize() for w in e.layout.replace("_LAYOUT_DATA", "").split("_")) or "Other"


def _schema_groups(schema) -> dict:
    """Groups with their key count and their switches: the toggles that turn the group's features on
    (other keys depend on them). E.g. the denoiser only runs when one of its "Enable Denoiser ..."
    toggles is on."""
    groups: dict = {}
    for e in schema.entries.values():
        g = groups.setdefault(_group_name(e), {"keys": 0, "scope": set(), "members": set()})
        g["keys"] += 1
        g["scope"].add(e.scope)
        g["members"].add(e.key)
    needed: set = set()
    for e in schema.entries.values():
        for cond in (e.depends_on or {}).get("all", []) + (e.depends_on or {}).get("any", []):
            needed.add(cond["key"])
    out = {}
    for name, v in sorted(groups.items()):
        switches = sorted(
            k for k in v["members"] & needed if schema.get(k).kind == "toggle" and not schema.get(k).depends_on
        )
        d = {"keys": v["keys"], "scope": "/".join(sorted(v["scope"]))}
        if switches:
            d["switches"] = {k: schema.get(k).label or k for k in switches}
        out[name] = d
    return out


def _schema_find(schema, text: str) -> tuple:
    """(exact key or None, matches, close suggestions). Case-insensitive; matches the key, the UI
    label, the group and the tab ("restore eyes", "pipeline position", "thread"...)."""
    import difflib

    if text in schema.entries:
        return text, [], []
    low = text.lower()
    words = [w for w in re.split(r"[\s_/]+", low) if w]

    def hay(k, e):
        return " ".join(x for x in (k, e.label, e.group, getattr(e, "tab", None)) if x).lower()

    matches = [k for k, e in schema.entries.items() if all(w in hay(k, e) for w in words)]
    names = {k.lower(): k for k in schema.entries}
    for k, e in schema.entries.items():
        if e.label:
            names.setdefault(e.label.lower(), k)
    close = []
    for c in difflib.get_close_matches(low, list(names), n=5, cutoff=0.6):
        if names[c] not in close:
            close.append(names[c])
    return None, sorted(matches), close


@command("schema")
def cmd_schema(ctx, key: str | None = None, group: str | None = None):
    """No argument: the list of groups. key: exact name, or part of a name (case-insensitive) ->
    matching keys. group: every key of a group ("*" = all). Each entry: type, default, bounds,
    options, personal + personal_reason."""
    rules = ctx.rules
    schema = ctx.schema
    if not key and not group:
        return {
            "groups": _schema_groups(schema),
            "total_keys": len(schema.entries),
            "hint": 'schema(group="<name>") for a group\'s keys, schema(key="<part of a name>") to search, '
                    'schema(group="*") for everything',
        }

    def describe(k, e):
        d = e.to_dict()
        d["group"] = _group_name(e)
        if e.options_note:
            # Read the live list from the widget, as VisoMaster filled it.
            w = (getattr(ctx.mw, "parameter_widgets", {}) or {}).get(k)
            if w is not None and hasattr(w, "count") and hasattr(w, "itemText"):
                d["options"] = [w.itemText(i) for i in range(w.count())]
                if not d["options"]:
                    d["options_note"] += ": none available right now"
        reason = rules.reason(k, schema)
        d["personal"] = reason is not None
        if reason:
            d["personal_reason"] = reason
        return d

    if group and group != "*":
        names = {g.lower(): g for g in _schema_groups(schema)}
        if group.lower() not in names:
            raise CommandError(f"unknown group: {group}. Groups: {', '.join(sorted(names.values()))}")
        group = names[group.lower()]
    keys = None
    note = None
    if key:
        exact, matches, close = _schema_find(schema, key)
        if exact:
            keys = [exact]
        elif matches:
            keys = matches
            note = f"no key named exactly '{key}': {len(matches)} key(s) contain it"
        else:
            hint = f"; did you mean: {', '.join(close)}" if close else '; schema() lists the groups'
            raise CommandError(f"unknown key: {key}{hint}")
    out = {}
    for k, e in schema.entries.items():
        if (keys is not None and k not in keys) or (group and group != "*" and _group_name(e) != group):
            continue
        out[k] = describe(k, e)
    if note:
        return {"note": note, "matches": out}
    return out


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


@command("snapshot", timeout=30)
def cmd_snapshot(ctx, path: str | None = None, view: bool = False, include_data: bool = True):
    """Preview PNG: processed frame at full resolution (view=true: the view as displayed)."""
    from PySide6 import QtCore, QtWidgets

    mw = ctx.mw
    gv = getattr(mw, "graphicsViewFrame", None)
    if gv is None:
        raise CommandError("graphicsViewFrame not found")
    image = None
    source = "view"
    if not view:
        scene = gv.scene()
        if scene is not None:
            for item in scene.items():
                if isinstance(item, QtWidgets.QGraphicsPixmapItem) and not item.pixmap().isNull():
                    image = item.pixmap().toImage()
                    source = "frame"
                    break
    if image is None:
        image = gv.grab().toImage()
    if image.isNull():
        raise CommandError("nothing to capture")
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    image.save(buf, "PNG")
    png = bytes(buf.data())
    buf.close()
    res = {"source": source, "width": image.width(), "height": image.height(), "bytes": len(png)}
    out = _out_path(path, ".png")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        kept = _keep_previous(out)
        out.write_bytes(png)
        res["path"] = str(out)
        if kept:
            res["previous_kept"] = kept
    if include_data:
        res["png_base64"] = base64.b64encode(png).decode("ascii")
    return res


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

# Actions known to open a modal dialog / file picker, or to close the app.
# Value = the API command to use instead (or None).
DIALOG_ACTIONS = {
    "actionLoad_SavedWorkspace": "load_workspace",
    "actionSave_CurrentWorkspace": "save_workspace",
    "actionOpen_Videos_Folder": "load_target_media",
    "actionOpen_Video_Files": "load_target_media",
    "actionLoad_Source_Images_Folder": "load_input_faces",
    "actionLoad_Source_Image_Files": "load_input_faces",
    "actionLoad_Embeddings": None,
    "actionSave_Embeddings": None,
    "actionSave_Embeddings_As": None,
    "actionSave_LastWorkspace": "save_workspace",
    "actionReset_to_LastWorkspace": "load_workspace",
    "actionQuit": None,
    "actionQuit_WithoutSaving": None,
    "actionView_Help_Shortcuts": None,
    "actionView_Help_Presets": None,
    "actionHelp_About": None,
    "outputFolderButton": "set OutputMediaFolder",
    "buttonTargetVideosPath": "load_target_media",
    "buttonInputFacesPath": "load_input_faces",
    "saveImageButton": None,
    "addJobButton": None,
    "deleteJobButton": None,
    "savePresetButton": None,
    "saveEmbeddingButton": None,
    "saveEmbeddingAsButton": None,
    "openEmbeddingButton": None,
    "buttonMediaRecord": "start_recording / stop_recording",
    "buttonProcessAll": "process_jobs",
    "buttonProcessSelected": "process_jobs",
}
# Embeddings: the API may only list them and select one for a face's swap.
# Creating, editing, deleting, loading or saving them is reserved to the user.
EMBEDDING_RESERVED = {
    "actionLoad_Embeddings",
    "actionSave_Embeddings",
    "actionSave_Embeddings_As",
    "openEmbeddingButton",
    "saveEmbeddingButton",
    "saveEmbeddingAsButton",
    "openEditorButton",  # embedding editor
    "purgeKVMapsButton",  # deletes K/V map files
}
# Same rule for controls added by future updates.
_EMBEDDING_WORDS = re.compile(r"embed|kv_?map|kvmap|merge", re.IGNORECASE)


def _is_view_toggle(name: str) -> bool:
    """View menu panel toggles (e.g. "Faces / Embeddings"): only show/hide a panel."""
    return name.startswith("actionViewToggle_") or name.endswith("PanelCheckBox")


def _embedding_reserved(name: str, label: str) -> bool:
    if _is_view_toggle(name):
        return False
    return name in EMBEDDING_RESERVED or bool(_EMBEDDING_WORDS.search(name) or _EMBEDDING_WORDS.search(label))


_KV_WORDS = re.compile(r"kv_?map|kvmap|k/v", re.IGNORECASE)


def _reserved_reason(name: str, label: str) -> str:
    if _KV_WORDS.search(name) or _KV_WORDS.search(label):
        return "K/V maps (reference data of the embeddings): reserved to the user"
    return "embeddings: reserved to the user (only list_embeddings / select_embedding are exposed)"


# Webcams / virtual camera: turning them on is a privacy matter, reserved to the user.
_PRIVACY_WORDS = re.compile(r"webcam|virt_?cam|virtual_?cam|camera", re.IGNORECASE)
# Settings toggles whose function turns a camera on (set refuses them = true). Not "camera" alone:
# FaceExpressionCameraGazeToggle only moves the swapped eyes.
_CAMERA_ON_KEYS = re.compile(r"webcam|virt_?cam|virtual_?cam", re.IGNORECASE)


def _privacy_reserved(name: str, label: str) -> bool:
    if _is_view_toggle(name):
        return False
    return bool(_PRIVACY_WORDS.search(name) or _PRIVACY_WORDS.search(label))


# Their slot shows a message box only when no target face is selected.
NEEDS_SELECTED_FACE = {
    "actionEdit_CopyParameters",
    "actionEdit_PasteParameters",
    "actionEdit_ResetParameters",
}
# find_target_faces shows a message box only while a webcam stream is starting:
# allowed when a video or image is the current media.
NEEDS_MEDIA = {"findTargetFacesButton"}
# VisoMaster's batch processing of the target list (name -> process_all_faces): run_action runs the
# same function without its dialogs (see _start_batch), then wait_recording follows it.
BATCH_ACTIONS = {"batchallImageButton": True, "batchImageButton": False}
# Blocked for another reason than a dialog.
MANUAL_REASONS = {
    "actionSave_LastWorkspace": "overwrites the user's last_workspace.json without a backup: use save_workspace",
    "actionReset_to_LastWorkspace": ("replaces the whole session with last_workspace.json without a backup: "
                                     "use load_workspace (saves the current one first)"),
}
# Native buttons with a safer API command.
USE_INSTEAD = {
    "applyPresetButton": "apply_config (imports a preset without touching personal settings)",
}
# Fallback when the slot cannot be found in the source.
_DIALOG_WORDS = re.compile(
    r"(load|save|open|select|browse|choose|delete|remove|clear|quit|exit|export|import|rename|folder|file|about|purge|new\b|add\b)",
    re.IGNORECASE,
)
# What a slot (or what it calls) opens, by priority: something outside VisoMaster,
# a popup menu, a modal dialog.
_MODAL_MARKERS = (
    ("external", re.compile(r"QDesktopServices\.openUrl|webbrowser\.open|os\.startfile")),
    ("menu", re.compile(r"\bQMenu\(")),
    ("dialog", re.compile(
        r"QFileDialog|QMessageBox|QInputDialog|create_and_show_messagebox|\.exec_?\(|[A-Za-z]Dialog\("
    )),
)
MODAL_REASONS = {
    "external": "opens an external document or web link outside VisoMaster",
    "menu": "opens a popup menu",
    "dialog": "opens a dialog",
}
_SLOT_RE = re.compile(
    r"(?:self|main_window)\.(\w+)\.(?:clicked|triggered|toggled)\.connect\(\s*"
    r"(?:partial\(\s*)?(?:lambda[^:]*:\s*)?([A-Za-z_][\w.]*)",
    re.S,
)
# Buttons built from spec tuples: ("attrName", "Label", "tooltip" "...", lambda: slot(main_window), ...)
_SPEC_SLOT_RE = re.compile(
    r"\(\s*\"(\w+)\",\s*\"[^\"\n]*\",\s*(?:\"(?:[^\"\\\n]|\\.)*\"\s*)+,\s*"
    r"(?:partial\(\s*)?(?:lambda[^:]*:\s*)?([A-Za-z_][\w.]*)"
)
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)(?:\.([A-Za-z_]\w*))?\s*\(")
_slot_index_cache: dict | None = None
_dialog_cache: dict = {}


def _resolve_slot(expr: str, module, mw):
    parts = expr.split(".")
    if parts[0] in ("self", "main_window"):
        obj, parts = type(mw), parts[1:]
    elif module is not None:
        obj = module
    else:
        return None
    for p in parts:
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj if callable(obj) else None


def _modal_kind(src: str) -> str | None:
    for kind, rx in _MODAL_MARKERS:
        if rx.search(src):
            return kind
    return None


def _has_dialog(fn, depth: int = 2) -> str | bool | None:
    """Kind of modal thing ("external", "menu", "dialog") fn's source (or the app functions it
    calls) opens; False if none; None: no source."""
    import functools
    import inspect

    while isinstance(fn, functools.partial):
        fn = fn.func
    fn = getattr(fn, "__func__", fn)
    key = (id(fn), depth)
    if key in _dialog_cache:
        return _dialog_cache[key]
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        _dialog_cache[key] = None
        return None
    found = _modal_kind(src) or False
    if not found and depth > 0:
        g = getattr(fn, "__globals__", {}) or {}
        seen = set()
        for a, b in _CALL_RE.findall(src):
            target = g.get(a)
            if b and target is not None:
                target = getattr(target, b, None)
            mod = getattr(target, "__module__", "") or ""
            if not callable(target) or not mod.startswith("app.") or id(target) in seen:
                continue
            seen.add(id(target))
            sub = _has_dialog(target, depth - 1)
            if sub:
                found = sub
                break
    _dialog_cache[key] = found
    return found


def _slot_index(mw) -> dict:
    """attr name -> [slot expressions], read from the app's own source (connect calls)."""
    global _slot_index_cache
    if _slot_index_cache is not None:
        return _slot_index_cache
    import sys

    index: dict = {}
    root = APP_DIR / "app" / "ui"
    for f in root.rglob("*.py"):
        try:
            if f.stat().st_size > 1_000_000:  # generated resources
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        modname = ".".join(f.relative_to(APP_DIR).with_suffix("").parts)
        module = sys.modules.get(modname)
        for attr, expr in _SLOT_RE.findall(text) + _SPEC_SLOT_RE.findall(text):
            index.setdefault(attr, []).append((expr, _resolve_slot(expr, module, mw)))
    _slot_index_cache = index
    return index


def _dialog_info(mw, name, label) -> tuple:
    """(opens_dialog: True|False|None, detected_by, slots, kind: external|menu|dialog|None)."""
    slots = _slot_index(mw).get(name, [])
    results = [_has_dialog(fn) if fn is not None else None for _, fn in slots]
    exprs = [e for e, _ in slots]
    kinds = [r for r in results if isinstance(r, str)]
    kind = next((k for k, _ in _MODAL_MARKERS if k in kinds), None)
    if name in DIALOG_ACTIONS:
        return True, "manual list", [], kind or "dialog"
    if kind:
        return True, "slot source code", exprs, kind
    if slots and all(r is False for r in results):
        return False, "slot source code", exprs, None
    if _DIALOG_WORDS.search(name) or _DIALOG_WORDS.search(label):
        return None, "name/label (slot not found)", exprs, None
    return False, "no dialog in connected slots", exprs, None


# Labels are free text: card buttons show file names, sizes and full paths.
_PATHLIKE = re.compile(r"[A-Za-z]:[\\/]|\\\\|(?:^|\s)/\w+/|\.\w{2,4}\b.*\d+(?:\.\d+)?\s*[KMG]i?B", re.S)


def _safe_label(text: str) -> str:
    text = (text or "").replace("&", "").strip()
    if _PATHLIKE.search(text) or merge.looks_like_path(text):
        return "<hidden: label contains a file name or path>"
    return text.splitlines()[0] if text else ""


def _is_item_button(mw, obj) -> bool:
    """Per-item cards (a video, a face, an embedding) or attributes pointing to the selected one
    (selected_video_button, cur_selected_target_face_button): not stable actions."""
    if "Card" in type(obj).__name__:
        return True
    for attr in ("target_videos", "target_faces", "input_faces", "merged_embeddings"):
        items = getattr(mw, attr, None)
        try:
            if isinstance(items, dict) and any(v is obj for v in items.values()):
                return True
        except Exception:
            pass
    return False


def _is_orphan_action(obj) -> bool:
    """A QAction attached to no menu/toolbar/widget: a leftover of the Qt Designer file
    (e.g. actionLoad_Saved_Workspace, actionTest) that the UI never shows."""
    try:
        return len(obj.associatedObjects()) == 0
    except Exception:
        return False


def _collect_actions(mw) -> dict:
    """QAction / buttons that are attributes of MainWindow, keyed by attribute name (stable)."""
    from PySide6 import QtGui, QtWidgets

    found: dict = {}
    for attr in dir(mw):
        if attr.startswith("_"):
            continue
        try:
            obj = getattr(mw, attr)
        except Exception:
            continue
        if isinstance(obj, QtGui.QAction):
            if obj.isSeparator() or obj.menu() is not None or _is_orphan_action(obj):
                continue
            found[attr] = obj
        elif isinstance(obj, QtWidgets.QAbstractButton):
            if _is_item_button(mw, obj):
                continue
            found[attr] = obj
    return found


# Actions that erase data without a dialog: run_action needs a confirmation code.
DESTRUCTIVE_ACTIONS = {
    "clearTargetFacesButton",
    "dropAllIssueFramesButton",
    "clearDroppedFramesButton",
    "clearScanResultsButton",
    "actionEdit_ResetParameters",
    "removeMarkerButton",
    "applyPresetButton",  # overwrites the selected face's parameters with the preset
}
NOT_DESTRUCTIVE = {
    "clearMemoryButton",  # "Clear VRAM": only unloads models
    "targetVideosSearchResetButton",  # clears the search box text
}
# Whole words only: "Presets" must not match "reset", nor "dropdown" match "drop".
_DESTRUCTIVE_WORD = re.compile(r"(clear|drop|delete|remove|reset|purge|discard|erase|wipe)(s|d|ed|ped|all)?", re.IGNORECASE)
_ACTION_CONFIRM: dict = {}  # code -> {"name", "expires"}


def _words(text: str) -> list:
    """camelCase / snake_case / free text -> lowercase words."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return [w.lower() for w in re.split(r"[^A-Za-z0-9]+", text) if w]


def _is_destructive(name: str, label: str) -> bool:
    if name in NOT_DESTRUCTIVE or _is_view_toggle(name):
        return False
    if name in DESTRUCTIVE_ACTIONS:
        return True
    return any(_DESTRUCTIVE_WORD.fullmatch(w) for w in _words(name) + _words(label))


_RESET_WORDS = re.compile(r"\b(reset|clear|delete|remove|erase)\b", re.IGNORECASE)


def _action_confirmation(name: str, label: str, confirm: str | None, command_name: str = "run_action"):
    """None when confirmed; otherwise the warning to send back (with a new single-use code)."""
    import secrets

    now = time.time()
    for code in [c for c, v in _ACTION_CONFIRM.items() if v["expires"] < now]:
        _ACTION_CONFIRM.pop(code, None)
    pending = _ACTION_CONFIRM.get(confirm or "")
    if pending is not None and pending["name"] == name:
        _ACTION_CONFIRM.pop(confirm, None)
        return None
    code = secrets.token_hex(3)
    _ACTION_CONFIRM[code] = {"name": name, "expires": now + 300}
    return {
        "triggered": False,
        "confirmation_required": True,
        "warning": (
            f"{name} ({label or 'no label'}) erases data in the current VisoMaster session and cannot be "
            "undone from VisoMaster. A config backup (control + face parameters) is written first, but "
            "faces, markers or scan results removed by the action are not restored by restore_backup. "
            f'To confirm: call {command_name} again with confirm="{code}" (valid 5 min).'
        ),
        "confirm": code,
    }


_NAME_NOISE = {"button", "action", "check", "box", "checkbox", "toggle"}


def _label_from_name(name: str) -> str:
    """frameAdvanceButton -> "Frame Advance" (for buttons that only show an icon)."""
    return " ".join(w.capitalize() for w in _words(name) if w not in _NAME_NOISE)


def _split_label(text: str) -> tuple:
    """Qt menu text "Fullscreen\\tF11" -> ("Fullscreen", "F11")."""
    label, _, shortcut = (text or "").partition("\t")
    return label, shortcut.strip()


def _media_ok(mw) -> bool:
    vp = getattr(mw, "video_processor", None)
    return bool(vp is not None and getattr(vp, "media_path", None) and getattr(vp, "file_type", None) in ("video", "image"))


def _describe(mw, name, obj) -> dict:
    from PySide6 import QtGui

    is_action = isinstance(obj, QtGui.QAction)
    raw, shortcut = _split_label(obj.text() or obj.toolTip() or "")
    label = _safe_label(raw) or _label_from_name(name)
    if is_action and not shortcut:
        try:
            shortcut = obj.shortcut().toString()
        except Exception:
            shortcut = ""
    opens, how, slots, kind = _dialog_info(mw, name, label)
    needs_face = name in NEEDS_SELECTED_FACE
    needs_media = name in NEEDS_MEDIA
    batch = name in BATCH_ACTIONS
    conditional = needs_face or needs_media or batch
    reserved = _embedding_reserved(name, label)
    privacy = _privacy_reserved(name, label)
    destructive = _is_destructive(name, label)
    policy_blocked = reserved or privacy or (opens is not False and not conditional)
    enabled = bool(obj.isEnabled())
    requires = []
    if not policy_blocked:
        if needs_face:
            requires.append("selected_face")
        if needs_media:
            requires.append("video_or_image_loaded")
        if destructive:
            requires.append("confirmation")
    d = {
        "name": name,
        "object_name": obj.objectName() or None,
        "kind": "action" if is_action else "button",
        "label": label,
        "enabled": enabled,
        "checkable": bool(obj.isCheckable()),
        "checked": bool(obj.isChecked()) if obj.isCheckable() else None,
        # Can run_action trigger it right now? (plus the conditions listed in "requires")
        "allowed": not policy_blocked and enabled,
        "requires": requires,
        "destructive": destructive,
        # opens_dialog: True / False / None (unknown); opens: what exactly (dialog / menu / external)
        "opens_dialog": False if conditional and not policy_blocked else ((kind == "dialog") if kind else opens),
        "opens": kind,
        "detected_by": how,
        "needs_wrapper": policy_blocked,
    }
    if shortcut:
        d["shortcut"] = shortcut
    if reserved:
        d["reason"] = _reserved_reason(name, label)
    elif privacy:
        d["reason"] = "turns on webcams / virtual camera (privacy): reserved to the user"
    elif name in MANUAL_REASONS:
        d["reason"] = MANUAL_REASONS[name]
    elif policy_blocked:
        d["reason"] = MODAL_REASONS.get(kind, "opens a dialog") if opens else "probable dialog: blocked to be safe"
    elif not enabled:
        d["reason"] = "disabled in VisoMaster right now (depends on the current media, scan or playback state)"
    if slots:
        d["slots"] = slots
    use_instead = DIALOG_ACTIONS.get(name) or USE_INSTEAD.get(name)
    if use_instead:
        d["use_instead"] = use_instead
    if needs_face and not policy_blocked:
        d["opens_dialog_when"] = "no target face is selected (run_action refuses that case)"
        d["note"] = "shows a message box only when no face is selected, so it requires a selected face"
    if needs_media and not policy_blocked:
        d["opens_dialog_when"] = "a webcam stream is starting (run_action refuses that case)"
        d["note"] = ("shows a message box only while a webcam stream is starting, so it requires a "
                     "video or image as the current media")
    if batch and not policy_blocked:
        what = ("every image of the target list; finds ALL faces in each image and swaps them"
                if BATCH_ACTIONS[name] else
                "every image and video of the target list with the current target face settings")
        d["note"] = (f"VisoMaster's own batch processing ({what}), output to OutputMediaFolder. run_action runs "
                     "it without its dialogs (its confirmation is answered Yes, no progress window, the final "
                     "summary is returned instead) and returns at once: then call wait_recording (progress, "
                     "then processing_time_s, processed / failed). stop_recording cancels it.")
    return d


@command("list_actions")
def cmd_list_actions(ctx, include_disabled: bool = True):
    """Menu QActions + main window buttons: stable name, label, allowed, requires, destructive, dialog."""
    actions = []
    for name, obj in sorted(_collect_actions(ctx.mw).items()):
        d = _describe(ctx.mw, name, obj)
        if include_disabled or d["enabled"]:
            actions.append(d)
    allowed = [a for a in actions if a["allowed"]]
    return {
        "counts": {
            "total": len(actions),
            "allowed": len(allowed),
            "allowed_with_confirmation": sum(1 for a in allowed if "confirmation" in a["requires"]),
            "allowed_with_selected_face": sum(1 for a in allowed if "selected_face" in a["requires"]),
            "blocked": len(actions) - len(allowed),
            "blocked_by_policy": sum(1 for a in actions if a["needs_wrapper"]),
            "disabled": sum(1 for a in actions if not a["enabled"]),
        },
        "actions": actions,
    }


@command("run_action", write=True, timeout=300)  # e.g. turning the swap on loads models synchronously
def cmd_run_action(ctx, name: str, confirm: str | None = None):
    """Trigger an action/button that opens NO dialog. Destructive ones need a confirmation code
    (1st call = warning + code, 2nd call with confirm=<code>)."""
    from PySide6 import QtGui

    mw = ctx.mw
    if name == "batch_input_faces":
        # Same command for MCP clients started before the batch_input_faces tool existed.
        return cmd_batch_input_faces(ctx)
    actions = _collect_actions(mw)
    obj = actions.get(name)
    if obj is None:
        for n, o in actions.items():
            if o.objectName() == name:
                name, obj = n, o
                break
    if obj is None:
        raise CommandError(f"unknown action: {name} (see list_actions)")
    d = _describe(mw, name, obj)
    if not d["allowed"]:
        hint = f"; use instead: {d['use_instead']}" if d.get("use_instead") else ""
        raise CommandError(f"{name} refused: {d.get('reason')}{hint}")
    if "selected_face" in d["requires"] and not _selected_face(mw):
        raise CommandError(f"{name} needs a selected target face")
    if "video_or_image_loaded" in d["requires"] and not _media_ok(mw):
        raise CommandError(f"{name} needs a video or image as the current media (not a webcam)")
    if not obj.isEnabled():
        raise CommandError(f"{name} is disabled")
    if name in BATCH_ACTIONS:
        return _start_batch(mw, name, BATCH_ACTIONS[name])
    backup = None
    if d["destructive"]:
        warning = _action_confirmation(name, d["label"], confirm)
        if warning is not None:
            return warning
        backup = _write_backup(mw, f"before-{name}")
    if isinstance(obj, QtGui.QAction):
        obj.trigger()
    else:
        obj.click()
    res = {"triggered": name, "checked": bool(obj.isChecked()) if obj.isCheckable() else None}
    if backup:
        res["config_backup"] = backup
    return res


# ---------------------------------------------------------------------------
# Workspace / media
# ---------------------------------------------------------------------------


# What VisoMaster's load_saved_workspace reads with data[...] (a missing key or a wrong type stops it halfway,
# after it cleared the session, with a modal error box).
_WS_ENTRY_KEYS = {
    "input_faces_data": {"media_path": str},
    "embeddings_data": {"embedding_store": dict},
    "target_faces_data": {"cropped_face": list, "embedding_store": dict, "assigned_merged_embeddings": list,
                          "assigned_input_faces": list, "assigned_input_embedding": dict},
}


def _camera_on(control: dict) -> list:
    """Toggles of a control dict that would turn on a webcam / the virtual camera."""
    return sorted(k for k, v in control.items()
                  if k.endswith("Toggle") and _CAMERA_ON_KEYS.search(k) and v in (True, "True", "true", 1))


def _workspace_shape_errors(data: dict) -> list:
    errors = []
    medias = data.get("target_medias_data", [])
    if not isinstance(medias, list):
        errors.append("target_medias_data is not a list")
    else:
        for i, m in enumerate(medias):
            if not (isinstance(m, dict) and isinstance(m.get("media_path"), str) and "media_id" in m):
                errors.append(f"target_medias_data[{i}] needs media_path and media_id")
    for section, keys in _WS_ENTRY_KEYS.items():
        entries = data.get(section, {})
        if not isinstance(entries, dict):
            errors.append(f"{section} is not an object")
            continue
        for eid, entry in entries.items():
            if not isinstance(entry, dict):
                errors.append(f"{section}[{eid}] is not an object")
                continue
            for k, kind in keys.items():
                if not isinstance(entry.get(k), kind):
                    errors.append(f"{section}[{eid}].{k} missing or not a {kind.__name__}")
            if "parameters" in entry and not isinstance(entry["parameters"], dict):
                errors.append(f"{section}[{eid}].parameters is not an object")
    for k, kind in (("job_marker_pairs", list), ("markers", dict), ("embedding_tabs_state", dict),
                    ("window_state_data", dict)):
        if k in data and data[k] is not None and not isinstance(data[k], kind):
            errors.append(f"{k} is not a {kind.__name__}")
    return errors


def _foreign_kv_maps(path: Path) -> list:
    """K/V map files a workspace would make VisoMaster unpickle, outside VisoMaster's model_assets folder
    (where VisoMaster itself writes them, as kv_<sha256>.pt). A workspace from elsewhere could point to a
    crafted pickle; one written by this VisoMaster only points inside model_assets."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CommandError(f"not a readable workspace JSON: {e}") from None
    # VisoMaster clears the session before it notices a wrong file (then shows a blocking error box).
    if not (isinstance(data, dict) and isinstance(data.get("control"), dict)
            and any(k in data for k in ("target_medias_data", "input_faces_data", "target_faces_data"))):
        raise CommandError("not a VisoMaster workspace (no control / target_medias_data...): "
                           "for a config file use diff then apply_config")
    cams = _camera_on(data["control"])
    if cams:
        raise CommandError(f"refused: this workspace turns on a webcam / the virtual camera ({', '.join(cams)}): "
                           "privacy, reserved to the user (load it yourself in VisoMaster, or turn it off in the file)")
    bad = _workspace_shape_errors(data)
    if bad:
        raise CommandError("damaged workspace, VisoMaster would clear the session then fail: " + "; ".join(bad[:5]))
    found, stack = [], [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "kv_map" and isinstance(v, str) and v:
                    found.append(v)
                else:
                    stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
    root = (APP_DIR / "model_assets").resolve()
    out = []
    for f in found:
        try:
            r = Path(f).resolve()
        except (OSError, ValueError):
            out.append(f)
            continue
        if r.exists() and root not in r.parents:
            out.append(f)
    return out


def _writable(p: Path) -> str | None:
    """Why `p` cannot be written (None: it can). Checked first: VisoMaster's save only reports an error in a
    modal message box (except for a *last_workspace.json name), which would block the API."""
    if p.is_dir():
        return f"{p} is a folder"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        probe = p.parent / f".vmf-write-test-{os.getpid()}-{threading.get_ident()}"
        probe.write_bytes(b"")
        probe.unlink()
        if p.exists():
            with open(p, "ab"):
                pass
    except OSError as e:
        return f"cannot write {p}: {e}"
    return None


def _save_workspace_file(mw, p: Path, previous: str | None = None) -> None:
    """VisoMaster writes the workspace to `p` itself (it registers the K/V map files the workspace uses under
    this path, so the file is not written elsewhere then moved). Checked after: a complete JSON workspace;
    otherwise `previous` (a copy of the old file) is put back."""
    from app.ui.widgets.actions import save_load_actions

    why = _writable(p)
    if why:
        raise CommandError(why + ": nothing saved")
    t0 = time.time()
    save_load_actions.save_current_workspace(mw, str(p))
    try:
        ok = p.stat().st_mtime >= t0 - 2 and isinstance(json.loads(p.read_text(encoding="utf-8")).get("control"), dict)
    except (OSError, ValueError, AttributeError):
        ok = False
    if not ok:
        restored = ""
        if previous:
            try:
                shutil.copy2(previous, p)
                restored = " (its previous content was put back)"
            except OSError:
                restored = f" (its previous content is in {previous})"
        raise CommandError(f"VisoMaster could not write a complete workspace to {p}{restored}: "
                           "see its console")


@command("load_workspace", write=True, timeout=600)
def cmd_load_workspace(ctx, path: str):
    """Load a full workspace (media, faces...). The current one is saved to backups first."""
    from app.ui.widgets.actions import save_load_actions

    mw = ctx.mw
    p = _abs(path)
    if p is None or not p.is_file():
        raise CommandError(f"file not found: {path}")
    outside = _foreign_kv_maps(p)
    if outside:
        raise CommandError(
            f"refused: this workspace makes VisoMaster load K/V map files from outside its own folder "
            f"({outside[:3]}). VisoMaster loads them with torch.load(weights_only=False), which can run code "
            f"hidden in the file. If you trust it, open it yourself in VisoMaster (File > Load Workspace)")
    backup = _backup_path(WORKSPACE_BACKUPS[0][1:-5])
    try:
        _save_workspace_file(mw, backup)
    except CommandError as e:
        raise CommandError(f"the current workspace could not be backed up: {e}. Nothing loaded, "
                           "the session is unchanged") from None
    _prune(BACKUP_DIR, "*" + WORKSPACE_BACKUPS[0], 10)  # whole workspaces (~MBs, face data): the 10 latest
    save_load_actions.load_saved_workspace(mw, str(p))
    return {"loaded": str(p), "workspace_backup": str(backup), "target_faces": list(mw.target_faces)}


@command("save_workspace", write=True, timeout=300)
def cmd_save_workspace(ctx, path: str | None = None):
    """Save the workspace to a .json file (default: last_workspace.json, the one VisoMaster reopens;
    its previous content is kept in backups first). Refused while busy: a batch swaps media and faces
    temporarily, they would be saved."""
    mw = ctx.mw
    last = Path(mw.last_workspace_path)
    p = _out_path(path, ".json", allowed=(last,)) or last
    why = _writable(p)
    if why:
        raise CommandError(why + ": nothing saved")
    out = {}
    if p.resolve() == last.resolve():
        kept = _keep_last_workspace(last)
    else:
        kept = _keep_previous(p)
    if kept:
        out["previous_kept"] = kept
    _save_workspace_file(mw, p, kept)
    return {"saved": str(p), **out}


def _keep_last_workspace(last: Path) -> str | None:
    """Copy of last_workspace.json in backups (the 10 most recent) before it is overwritten."""
    if not last.is_file():
        return None
    keep = _backup_path(WORKSPACE_BACKUPS[1][1:-5])
    shutil.copy2(last, keep)
    _prune(BACKUP_DIR, "*" + WORKSPACE_BACKUPS[1], 10)
    return str(keep)


def _split_paths(path) -> tuple[str | None, list]:
    paths = path if isinstance(path, list) else [path]
    resolved = [_abs(p) for p in paths if p]
    if not resolved:
        raise CommandError("missing path")
    for p in resolved:
        if not p.exists():
            raise CommandError(f"not found: {p}")
    if len(resolved) == 1 and resolved[0].is_dir():
        return str(resolved[0]), []
    if any(p.is_dir() for p in resolved):
        raise CommandError("give either one folder or a list of files")
    return None, [str(p) for p in resolved]


@command("load_target_media", write=True, timeout=120)
def cmd_load_target_media(ctx, path):
    """Load a folder or files as target media (no QFileDialog)."""
    from app.ui.widgets.actions import list_view_actions

    mw = ctx.mw
    folder, files = _split_paths(path)
    shown = folder or os.path.dirname(files[0])
    list_view_actions.select_target_medias(mw, source_type="api", folder_name=folder or False, files_list=files)
    mw.targetVideosPathLineEdit.setText(shown)
    mw.targetVideosPathLineEdit.setToolTip(shown)
    mw.last_target_media_folder_path = shown
    return {"folder": folder, "files": files, "note": "asynchronous loading (thumbnails in progress): " + WAIT_THEN + "use them"}


@command("load_input_faces", write=True, timeout=120)
def cmd_load_input_faces(ctx, path):
    """Load a folder or images as input faces (no QFileDialog)."""
    from app.ui.widgets.actions import list_view_actions

    mw = ctx.mw
    folder, files = _split_paths(path)
    shown = folder or os.path.dirname(files[0])
    list_view_actions.select_input_face_images(mw, source_type="api", folder_name=folder or False, files_list=files)
    mw.inputFacesPathLineEdit.setText(shown)
    mw.inputFacesPathLineEdit.setToolTip(shown)
    mw.last_input_media_folder_path = shown
    return {"folder": folder, "files": files, "note": "asynchronous loading (thumbnails in progress): " + WAIT_THEN + "use them"}


def _card_name(button) -> str:
    path = getattr(button, "media_path", None) or getattr(button, "file_path", None) or ""
    return os.path.basename(str(path)) if path else ""


@command("select_target_media", write=True, timeout=120)
def cmd_select_target_media(ctx, name: str | None = None):
    """Select a loaded target media (file name, media_id or index); no name: list them."""
    mw = ctx.mw
    cards = list((getattr(mw, "target_videos", {}) or {}).items())
    listing = [{"index": i, "media_id": mid, "name": _card_name(b)} for i, (mid, b) in enumerate(cards)]
    if name is None:
        return listing
    match = None
    for i, (mid, b) in enumerate(cards):
        if name in (mid, _card_name(b), str(i)):
            match = b
            break
    if match is None:
        raise CommandError(f"media '{name}' not found; loaded: {listing}")
    if getattr(match, "file_type", None) == "webcam":
        raise CommandError(f"{_card_name(match) or name} is a webcam: turning it on is reserved to the user (privacy)")
    match.click()
    vp = mw.video_processor
    return {"selected": _card_name(match), "type": getattr(vp, "file_type", None)}


def _input_face_listing(mw) -> list:
    targets = getattr(mw, "target_faces", {}) or {}
    out = []
    for iid, b in (getattr(mw, "input_faces", {}) or {}).items():
        out.append({"id": iid, "name": _card_name(b),
                    "assigned_to": [t for t, tb in targets.items() if iid in (getattr(tb, "assigned_input_faces", {}) or {})]})
    return out


@command("select_input_face", write=True)
def cmd_select_input_face(ctx, name: str | None = None, face_id: str | None = None, selected: bool = True):
    """Assign a loaded input face (file name or id) to a target face, like a click on its card in
    VisoMaster: it becomes the face's only input face (selected embeddings stay). selected=false:
    remove it. No name: list the input faces and the target faces they are assigned to."""
    mw = ctx.mw
    inputs = getattr(mw, "input_faces", {}) or {}
    if name is None:
        return _input_face_listing(mw)
    iid, btn = name, inputs.get(name)
    if btn is None:
        matches = [(i, b) for i, b in inputs.items() if _card_name(b) == name]
        if len(matches) > 1:
            raise CommandError(f"several input faces are named '{name}': give the id {[m[0] for m in matches]}")
        if not matches:
            raise CommandError(
                f"input face '{name}' not found ({len(inputs)} loaded; load_input_faces is asynchronous: "
                "retry in a moment if you just loaded it)"
            )
        iid, btn = matches[0]
    faces = getattr(mw, "target_faces", {}) or {}
    fid = face_id or _selected_face(mw)
    if not fid:
        raise CommandError("no target face selected: select_target_face <id> first")
    if fid not in faces:
        raise CommandError(f"unknown face '{fid}' (available: {list(faces)})")
    reselected = False
    if _selected_face(mw) != fid:
        faces[fid].click()  # same as clicking the target face card
        reselected = True
    if getattr(mw, "cur_selected_target_face_button", None) is not faces[fid]:
        raise CommandError(f"could not select face '{fid}'")
    current = list((faces[fid].assigned_input_faces or {}).keys())
    if (selected and current == [iid] and btn.isChecked()) or (not selected and iid not in current):
        return {"face_id": fid, "input_face": iid, "assigned": current, "changed": False}
    # Native path (InputFaceCardButton.load_input_face): click toggles the check state, then assigns
    # exclusively (checked) or removes (unchecked).
    btn.setChecked(not selected)
    btn.click()
    after = list((faces[fid].assigned_input_faces or {}).keys())
    return {"face_id": fid, "input_face": iid, "name": _card_name(btn), "assigned": after,
            "changed": after != current, "face_reselected": reselected}


@command("select_target_face", write=True)
def cmd_select_target_face(ctx, face_id: str | None = None):
    """Select a target face (after detection in VisoMaster); no id: list them."""
    mw = ctx.mw
    faces = getattr(mw, "target_faces", {}) or {}
    if face_id is None:
        return {"target_faces": list(faces), "selected": _selected_face(mw)}
    if face_id not in faces:
        raise CommandError(f"unknown face '{face_id}' (available: {list(faces)})")
    faces[face_id].click()
    return {"selected": _selected_face(mw)}


# ---------------------------------------------------------------------------
# Embeddings: list + select for swap ONLY (no create / edit / delete / load / save)
# ---------------------------------------------------------------------------


def _all_embeddings(mw) -> dict:
    try:
        from app.ui.widgets.actions import list_view_actions

        return dict(list_view_actions.get_all_merged_embeddings(mw))
    except Exception:
        return dict(getattr(mw, "merged_embeddings", {}) or {})


def _embedding_tabs(mw) -> dict:
    """embedding_id -> tab label."""
    tabs = getattr(mw, "embeddingTabs", None)
    out = {}
    for i, state in enumerate(getattr(mw, "embedding_tab_states", None) or []):
        try:
            label = tabs.tabText(i) if tabs is not None and i < tabs.count() else str(i)
        except Exception:
            label = str(i)
        for eid in (state.get("embeddings") or {}):
            out[eid] = label
    if tabs is not None:
        try:
            current = tabs.tabText(tabs.currentIndex())
            for eid in (getattr(mw, "merged_embeddings", {}) or {}):
                out.setdefault(eid, current)
        except Exception:
            pass
    return out


def _assignments(mw) -> dict:
    """face_id -> [embedding ids assigned for the swap]."""
    return {
        fid: list((getattr(btn, "assigned_merged_embeddings", {}) or {}).keys())
        for fid, btn in (getattr(mw, "target_faces", {}) or {}).items()
    }


@command("list_embeddings")
def cmd_list_embeddings(ctx):
    """Merged embeddings (all tabs) and which target faces they are assigned to. Read-only."""
    mw = ctx.mw
    tabs = _embedding_tabs(mw)
    assigned = _assignments(mw)
    items = []
    for eid, btn in _all_embeddings(mw).items():
        items.append({
            "id": eid,
            "name": getattr(btn, "embedding_name", None) or btn.text(),
            "tab": tabs.get(eid),
            "has_kv_map": bool(getattr(btn, "kv_map", None) or getattr(btn, "kv_map_list", None)),
            "assigned_to": [fid for fid, ids in assigned.items() if eid in ids],
        })
    return {"embeddings": items, "selected_target_face_id": _selected_face(mw), "assignments": assigned}


@command("select_embedding", write=True)
def cmd_select_embedding(ctx, embedding: str, face_id: str | None = None, selected: bool = True):
    """Select an embedding (id or exact name) for a target face's swap, like a click in
    VisoMaster: it becomes the face's only merged embedding. selected=false: remove it."""
    mw = ctx.mw
    all_embeds = _all_embeddings(mw)
    btn = all_embeds.get(embedding)
    if btn is None:
        matches = [(eid, b) for eid, b in all_embeds.items()
                   if (getattr(b, "embedding_name", None) or b.text()) == embedding]
        if len(matches) > 1:
            raise CommandError(f"several embeddings are named '{embedding}': give the id {[m[0] for m in matches]}")
        if not matches:
            raise CommandError(f"embedding '{embedding}' not found (see list_embeddings)")
        embedding, btn = matches[0]
    faces = getattr(mw, "target_faces", {}) or {}
    fid = face_id or _selected_face(mw)
    if not fid:
        raise CommandError("no target face selected: select_target_face <id> first")
    if fid not in faces:
        raise CommandError(f"unknown face '{fid}' (available: {list(faces)})")
    reselected = False
    if _selected_face(mw) != fid:
        faces[fid].click()  # same as clicking the target face card
        reselected = True
    if getattr(mw, "cur_selected_target_face_button", None) is not faces[fid]:
        raise CommandError(f"could not select face '{fid}'")
    if not btn.isEnabled():
        raise CommandError("this embedding is disabled in the UI")
    current = list((faces[fid].assigned_merged_embeddings or {}).keys())
    if selected and current == [embedding] and btn.isChecked():
        return {"face_id": fid, "embedding": embedding, "assigned": current, "changed": False}
    if not selected and embedding not in current:
        return {"face_id": fid, "embedding": embedding, "assigned": current, "changed": False}
    # Native path (EmbeddingCardButton.load_embedding): click toggles the
    # check state, then assigns exclusively (selected) or removes (unselected).
    btn.setChecked(not selected)
    btn.click()
    after = list((faces[fid].assigned_merged_embeddings or {}).keys())
    return {"face_id": fid, "embedding": embedding, "assigned": after, "changed": after != current,
            "face_reselected": reselected}


# ---------------------------------------------------------------------------
# TensorRT engine cache
# ---------------------------------------------------------------------------
# Building = ModelsProcessor.load_model(name) with TensorRT active: exactly what
# VisoMaster does the first time a model is used (isolated probe process that
# compiles the engine into VisoMaster-Fusion\tensorrt-engines, build dialog).


def _mp(mw):
    mp = getattr(mw, "models_processor", None)
    if mp is None:
        raise CommandError("models_processor not found")
    return mp


def _trt_active(mp) -> bool:
    for p in getattr(mp, "providers", []) or []:
        if (p[0] if isinstance(p, tuple) else p) == "TensorrtExecutionProvider":
            return True
    return False


def _model_tables():
    try:
        from app.processors import models_data
    except ImportError:  # outside VisoMaster (tests)
        return {}, {}, set()
    return (
        getattr(models_data, "restorer_model_mapping", {}),
        getattr(models_data, "compound_models_mapping", {}),
        set(getattr(models_data, "fp16_safe_models_list", []) or []),
    )


def _atomic_models(name: str) -> list:
    """UI / compound name -> atomic ONNX model names (as load_model resolves them)."""
    restorers, compounds, _ = _model_tables()
    canonical = restorers.get(name, name)
    return list(compounds.get(canonical, (canonical,)))


def _cache_state(mp, name: str) -> str | None:
    path = (mp.models_path or {}).get(name)
    if not path:
        return None
    try:
        # load_model checks the cache with the shape-inferred sidecar when one exists
        # (_ensure_trt_ready_onnx); mirror that without creating it.
        sidecar = path[: -len(".onnx")] + ".trtshape.onnx" if path.lower().endswith(".onnx") else path
        probe_path = sidecar if os.path.exists(sidecar) else path
        return mp._check_tensorrt_cache_state(name, probe_path)
    except Exception:
        return None


LAZY_LABEL = "built on first use"


def _engine_label(mp, name: str, path: str, state: str | None) -> str:
    if state == "EXPLICIT":
        return "built"
    if state == "LEGACY":
        return "built (legacy format)"
    # Probe done (context file written) but no engine: dynamic-shape model that
    # TensorRT compiles at its first real inference.
    base = os.path.splitext(os.path.basename(str(path)))[0]
    cache = APP_DIR / "tensorrt-engines"
    if name in (getattr(mp, "models_pending_build", set()) or set()) or any(
        (cache / f"{p}_ctx.onnx").is_file() for p in (name, base)
    ):
        return LAZY_LABEL
    return "absent"


def _toggled_models() -> set:
    """Models switched on/off by a UI option (others are 'core': swappers, detectors, ArcFace...)."""
    restorers, compounds, _ = _model_tables()
    names = set(restorers.values()) | set(compounds) | {s for subs in compounds.values() for s in subs}
    try:
        from app.ui.widgets.models_toggle_data import MODELS_TOGGLE_MAP

        names |= set(MODELS_TOGGLE_MAP)
    except ImportError:
        pass
    return names


def _model_row(mp, name: str, toggled: set | None = None) -> dict:
    _, _, fp16 = _model_tables()
    path = (mp.models_path or {}).get(name, "")
    onnx = str(path).lower().endswith(".onnx")
    state = _cache_state(mp, name) if onnx else None
    core = name not in (toggled if toggled is not None else _toggled_models())
    # For core models VisoMaster's is_model_active_in_ui() always answers True
    # ("assumed needed"), which says nothing about the current settings.
    active = None
    if not core:
        try:
            active = bool(mp.is_model_active_in_ui(name))
        except Exception:
            active = None
    return {
        "model": name,
        "file": os.path.basename(str(path)) if path else None,
        "downloaded": bool(path) and os.path.isfile(path),
        "tensorrt_capable": onnx,
        "engine": _engine_label(mp, name, path, state) if onnx else None,
        "lazy_build_pending": name in (getattr(mp, "models_pending_build", set()) or set()),
        "loaded": (mp.models or {}).get(name) is not None,
        "core": core,
        "active_in_settings": active,
        "fp16": name in fp16,
    }


def _engines_dir_size() -> int:
    d = APP_DIR / "tensorrt-engines"
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.is_dir() else 0


@command("list_tensorrt_models")
def cmd_list_tensorrt_models(ctx, only: str | None = None):
    """ONNX models and the state of their TensorRT engine. only: missing | built | active | loaded."""
    mp = _mp(ctx.mw)
    toggled = _toggled_models()
    rows = [_model_row(mp, n, toggled) for n in (mp.models_path or {})]
    rows = [r for r in rows if r["tensorrt_capable"]]
    if only == "missing":
        rows = [r for r in rows if r["engine"] == "absent" and r["downloaded"]]
    elif only == "built":
        rows = [r for r in rows if str(r["engine"]).startswith("built") and r["engine"] != LAZY_LABEL]
    elif only == "active":
        rows = [r for r in rows if r["active_in_settings"] or r["loaded"]]
    elif only == "loaded":
        rows = [r for r in rows if r["loaded"]]
    elif only:
        raise CommandError("only: missing | built | active | loaded")
    return {
        "provider": getattr(mp, "provider_name", None),
        "tensorrt_active": _trt_active(mp),
        "engines_dir": str(APP_DIR / "tensorrt-engines"),
        "engines_size_mb": round(_engines_dir_size() / 1048576, 1),
        "models": rows,
    }


_TRT_CONFIRM: dict = {}  # confirmation code -> {"queue", "expires"}

_REBUILD_WARNING = (
    "FORCED rebuild of {n} TensorRT engine(s): {models}. "
    "Existing engines ({existing}) will be DELETED and recompiled, as on first activation: "
    "allow 1 to 15 min per model, GPU busy, and VisoMaster refuses other write commands meanwhile "
    "(no playback/recording). Global TensorRT timing caches are cleared too "
    "(later compilations will be a bit slower). Affected models are unloaded and reloaded "
    "if they were in memory. If a model fails, it is left without an engine and will be recompiled on its "
    "next use. Useful after an NVIDIA driver or TensorRT update. "
    'To confirm: call again with confirm="{code}" (valid 5 min).'
)


def _effective_onnx_path(mp, name: str) -> str:
    path = (mp.models_path or {}).get(name, "")
    if path.lower().endswith(".onnx"):
        sidecar = path[: -len(".onnx")] + ".trtshape.onnx"
        if os.path.exists(sidecar):
            return sidecar
    return path


def _force_unload(mp, name: str):
    """Unload even with KeepModelsAliveToggle, as VisoMaster does when switching provider."""
    previous = getattr(mp, "force_unload_in_progress", False)
    mp.force_unload_in_progress = True
    try:
        mp.unload_model(name, force_immediate=True)
    finally:
        mp.force_unload_in_progress = previous


def _delete_engine(mp, name: str):
    """VisoMaster's own cache cleanup for one model (ctx + engine + timing files)."""
    opts = dict(getattr(mp, "trt_ep_options", {}) or {})
    opts["trt_engine_cache_prefix"] = name
    mp._clean_tensorrt_cache(_effective_onnx_path(mp, name), opts)


PROBE_TIMEOUT_S = 900  # VisoMaster: probe_process.join(timeout=900)
PROBE_ATTEMPTS = 3  # VisoMaster: max_retries = 3
_ATTEMPT_RE = re.compile(r"Probe attempt (\d+) of (\d+)")
_EXIT_RE = re.compile(r"exit code (-?\d+)")


class _ThreadTee:
    """Mirror what ONE thread prints (VisoMaster's build messages) into a callback,
    while everything still reaches the real console."""

    def __init__(self, stream, ident, on_line):
        self._stream, self._ident, self._on_line, self._buf = stream, ident, on_line, ""

    def write(self, s):
        n = self._stream.write(s)
        if threading.get_ident() == self._ident:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    try:
                        self._on_line(line.rstrip())
                    except Exception:
                        pass
        return n

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _load_with_capture(mp, name, item):
    """mp.load_model(name) while recording attempts / timeouts / errors in the job item."""
    item.update(attempt=None, timeouts=0, exit_codes=[], log_tail=[])

    def on_line(line):
        tail = item["log_tail"]
        tail.append(line[:300])
        del tail[:-20]
        m = _ATTEMPT_RE.search(line)
        if m:
            item["attempt"] = f"{m.group(1)}/{m.group(2)}"
        # VisoMaster prints two lines per timeout ("Probe process ... timed out!" then "TensorRT Engine
        # build timed out."): count the probe one only.
        if "probe process" in line.lower() and "timed out" in line.lower():
            item["timeouts"] += 1
        m = _EXIT_RE.search(line)
        if m and "failed" in line.lower():
            item["exit_codes"].append(int(m.group(1)))

    ident = threading.get_ident()
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _ThreadTee(out, ident, on_line), _ThreadTee(err, ident, on_line)
    try:
        return mp.load_model(name)
    finally:
        # Restore only if nobody replaced them meanwhile.
        if isinstance(sys.stdout, _ThreadTee):
            sys.stdout = out
        if isinstance(sys.stderr, _ThreadTee):
            sys.stderr = err


def _failure_reason(item) -> str:
    if item.get("timeouts"):
        return (f"TensorRT compilation timeout ({item['timeouts']} x {PROBE_TIMEOUT_S // 60} min, "
                f"limit set by VisoMaster)")
    if item.get("exit_codes"):
        return f"compilation failed (exit codes {item['exit_codes']})"
    for line in reversed(item.get("log_tail") or []):
        if "[ERROR]" in line or "Error" in line:
            return line.strip()
    return "build failed (see the VisoMaster console)"


def _trt_worker(mw, job):
    mp = mw.models_processor
    try:
        for item in job["items"]:
            name = item["model"]
            if job["cancel"]:
                item.update(status="skipped", reason="cancelled")
                continue
            job["current"] = name
            item.update(status="running", started_ts=time.time(), started=time.strftime("%Y-%m-%dT%H:%M:%S"))
            atoms = _atomic_models(name)
            if not item["rebuild"] and all(_cache_state(mp, a) is not None for a in atoms):
                item.update(status="skipped", reason="engine already built")
                continue
            was_loaded = {a: (mp.models or {}).get(a) is not None for a in atoms}
            try:
                if item["rebuild"]:
                    for a in atoms:
                        _force_unload(mp, a)
                        _delete_engine(mp, a)
                session = _load_with_capture(mp, name, item)  # = first activation with TensorRT
            except Exception as e:
                session = None
                item["error"] = f"{type(e).__name__}: {e}"
            item["seconds"] = round(time.time() - item["started_ts"], 1)
            if session is None:
                item.update(
                    status="failed",
                    error=item.get("error") or _failure_reason(item),
                    after_failure="no engine: VisoMaster cleaned the cache, the model will be recompiled on its "
                    "next use; retry with models=\"failed\"",
                )
                for a in atoms:  # nothing half-loaded left behind
                    if not was_loaded[a]:
                        try:
                            _force_unload(mp, a)
                        except Exception:
                            pass
                continue
            pending = getattr(mp, "models_pending_build", set()) or set()
            lazy = [a for a in atoms if a in pending]
            if lazy:
                # Dynamic input shape: the probe succeeded but TensorRT only compiles the
                # engine at the first real inference (VisoMaster's "lazy build").
                item.update(status="lazy", lazy=lazy,
                            note="dynamic input shape: VisoMaster compiles the engine on first real use")
            else:
                item["status"] = "built"
            # Leave memory as it was before the build.
            for a in atoms:
                if not was_loaded[a]:
                    try:
                        _force_unload(mp, a)
                    except Exception:
                        pass
    finally:
        job["current"] = None
        job["running"] = False
        job["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")


@command("build_tensorrt_cache", write=True)
def cmd_build_tensorrt_cache(ctx, models: Any = "active", rebuild: bool = False, confirm: str | None = None):
    """Build TensorRT engines (same action as a model's first activation).
    models: list of names | "active" (loaded + optional models enabled in the settings; core models
    - swappers, detectors, ArcFace - must be named) | "all" (every downloaded model) | "failed".
    Without rebuild, an existing engine is never recompiled. rebuild=true (e.g. after an NVIDIA driver
    update): delete and recompile; 1st call = warning + code, 2nd call with confirm=<code>.
    Runs in the background: follow with tensorrt_build_status."""
    global _TRT_JOB
    import secrets
    import threading

    mw = ctx.mw
    mp = _mp(mw)
    if not _trt_active(mp):
        raise CommandError(
            f"TensorRT is not the active provider ({getattr(mp, 'provider_name', '?')}). "
            "Set ProvidersPrioritySelection to TensorRT or TensorRT-Engine first (set_value; it is a personal "
            "setting: apply_config never changes it, set_value does)."
        )
    if getattr(getattr(mw, "video_processor", None), "processing", False):
        raise CommandError("playback in progress: stop it first")
    listing = cmd_list_tensorrt_models(ctx)["models"]
    by_name = {r["model"]: r for r in listing}
    if models in (None, "active"):
        selection = [r["model"] for r in listing if r["active_in_settings"] or r["loaded"]]
    elif models in ("all", "missing"):
        selection = [r["model"] for r in listing if r["downloaded"]]
    elif models == "failed":
        selection = [i["model"] for i in (_TRT_JOB or {}).get("items", []) if i["status"] == "failed"]
        if not selection:
            return {"started": False, "message": "no failure in the last build"}
    elif isinstance(models, list) and models:
        restorers, compounds, _ = _model_tables()
        known = set(by_name) | set(restorers) | set(compounds)
        unknown = [m for m in models if m not in known]
        if unknown:
            raise CommandError(f"unknown models: {unknown} (see list_tensorrt_models)")
        selection = list(dict.fromkeys(models))
    else:
        raise CommandError('models: list of names, "active", "all" or "failed"')
    # Models VisoMaster knows (e.g. added by an update) but whose file is not downloaded: cannot be built.
    not_downloaded = [r["model"] for r in listing if not r["downloaded"]] if models in ("all", "missing") else []
    nd_note = {"not_downloaded": not_downloaded,
               "not_downloaded_note": "known to VisoMaster but not downloaded yet: not built "
                                      "(VisoMaster downloads models with its launcher / on first use)"} if not_downloaded else {}
    missing_files = sorted(
        {a for m in selection for a in _atomic_models(m) if a in by_name and not by_name[a]["downloaded"]}
    )
    if missing_files:
        raise CommandError(f"model files missing: {missing_files} (Check / Update Models in the launcher)")

    def built(m):
        return all(_cache_state(mp, a) is not None for a in _atomic_models(m))

    def known_lazy(m):
        # Probed before, no engine: dynamic-shape model, TensorRT compiles it on first real
        # inference. Probing it again would only cost time and give the same answer.
        missing = [a for a in _atomic_models(m) if _cache_state(mp, a) is None]
        return bool(missing) and all(
            _engine_label(mp, a, (mp.models_path or {}).get(a, ""), None) == LAZY_LABEL for a in missing
        )

    already = [m for m in selection if built(m)]
    lazy_known: list = []
    if not rebuild:
        lazy_known = [m for m in selection if m not in already and known_lazy(m)]
        queue = [m for m in selection if m not in already and m not in lazy_known]
        if not queue:
            return {
                "started": False,
                "already_built": already,
                "lazy_known": lazy_known,
                "message": "nothing to build: engines already present"
                + (" or built on first use (dynamic input shape)" if lazy_known else "")
                + " (rebuild=true to force)",
                **nd_note,
            }
    else:
        queue = selection
        if not queue:
            return {"started": False, "message": "no model selected"}
        now = time.time()
        for code in [c for c, v in _TRT_CONFIRM.items() if v["expires"] < now]:
            _TRT_CONFIRM.pop(code, None)
        pending = _TRT_CONFIRM.get(confirm or "")
        if pending is None or pending["queue"] != queue:
            code = secrets.token_hex(3)
            _TRT_CONFIRM[code] = {"queue": queue, "expires": now + 300}
            return {
                "started": False,
                "confirmation_required": True,
                "warning": _REBUILD_WARNING.format(
                    n=len(queue), models=", ".join(queue), existing=", ".join(already) or "none", code=code
                ),
                "models": queue,
                "existing_engines": already,
                "confirm": code,
            }
        _TRT_CONFIRM.pop(confirm, None)
    _TRT_JOB = {
        "running": True,
        "cancel": False,
        "current": None,
        "items": [{"model": m, "status": "pending", "rebuild": bool(rebuild)} for m in queue],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished": None,
    }
    threading.Thread(target=_trt_worker, args=(mw, _TRT_JOB), name="vmf-trt-build", daemon=True).start()
    return {
        "started": True,
        "rebuild": bool(rebuild),
        "queue": queue,
        "skipped_already_built": [] if rebuild else already,
        "lazy_known": lazy_known,
        "note": "several minutes per model; VisoMaster shows its progress dialog; "
        "follow with tensorrt_build_status or wait_tensorrt_build"
        + ("; lazy_known: dynamic input shape, not pre-buildable, VisoMaster compiles them on first real use"
           if lazy_known else ""),
        **nd_note,
    }


@command("tensorrt_build_status")
def cmd_tensorrt_build_status(ctx, detail: bool = False):
    """Progress of build_tensorrt_cache. Compact by default: counts, current model, running and failed
    items (with their log tail) and the final summary; detail=true lists every model."""
    if _TRT_JOB is None:
        return {"running": False, "message": "no build started since VisoMaster started"}
    now = time.time()
    items = []
    for it in list(_TRT_JOB["items"]):  # mutated by the build thread: copy
        it = dict(it)
        ts = it.pop("started_ts", None)
        it["log_tail"] = list(it.get("log_tail") or [])[-8:]
        if it["status"] == "running" and ts:
            it["elapsed_s"] = round(now - ts, 1)
            # VisoMaster gives up after PROBE_ATTEMPTS x PROBE_TIMEOUT_S; beyond that something hangs.
            if it["elapsed_s"] > PROBE_ATTEMPTS * PROBE_TIMEOUT_S + 300:
                it["warning"] = ("abnormally long: beyond VisoMaster's own limit; check the console, "
                                 "restart VisoMaster if nothing moves")
        items.append(it)
    counts = {
        s: sum(1 for i in items if i["status"] == s)
        for s in ("pending", "running", "built", "lazy", "failed", "skipped")
    }
    done = counts["built"] + counts["lazy"] + counts["failed"] + counts["skipped"]
    res = {
        "running": _TRT_JOB["running"],
        "current": _TRT_JOB["current"],
        "progress": f"{done}/{len(items)}",
        "counts": counts,
        "cancel_requested": _TRT_JOB["cancel"],
        "started": _TRT_JOB["started"],
        "finished": _TRT_JOB["finished"],
        "items": items if detail else [i for i in items if i["status"] in ("running", "failed")],
    }
    if not detail:
        res["items_note"] = "running and failed models only: detail=true lists every model"
    if not _TRT_JOB["running"]:
        res["summary"] = _trt_summary(items)
    return res


def _trt_summary(items: list) -> dict:
    """Final result per model, ready to be reported: success / failure, build time, reason."""
    out: dict = {"built": [], "lazy": [], "failed": [], "skipped": []}
    for i in items:
        st = i.get("status")
        if st not in out:
            continue
        row = {"model": i["model"]}
        if i.get("seconds") is not None:
            row["build_time_s"] = i["seconds"]
        if st == "failed":
            row["error"] = i.get("error")
            if i.get("timeouts"):
                row["timeouts"] = i["timeouts"]
        elif st == "skipped":
            row["reason"] = i.get("reason")
        elif st == "lazy":
            row["note"] = "engine compiled by VisoMaster on first real use"
        out[st].append(row)
    out["total_build_time_s"] = round(sum(i.get("seconds") or 0 for i in items), 1)
    out["success"] = not out["failed"]
    return {k: v for k, v in out.items() if v != []}


@command("cancel_tensorrt_build")
def cmd_cancel_tensorrt_build(ctx):
    """Stop after the current model (a compilation in progress is not interrupted)."""
    if _TRT_JOB is None or not _TRT_JOB.get("running"):
        raise CommandError("no build in progress")
    _TRT_JOB["cancel"] = True
    return {"cancel_requested": True, "current": _TRT_JOB.get("current")}


# ---------------------------------------------------------------------------
# Recording / jobs
# ---------------------------------------------------------------------------


@command("start_recording", write=True, timeout=60)
def cmd_start_recording(ctx, from_start: bool = False):
    """Start recording from the playhead (from_start=true: from the first frame). Preconditions are
    checked to avoid message boxes; an error is returned if VisoMaster does not start it."""
    mw = ctx.mw
    vp = mw.video_processor
    if getattr(vp, "file_type", None) not in ("video", "image"):
        raise CommandError("no target video/image selected (webcam not supported)")
    if vp.file_type == "image":
        # VisoMaster would open a "Cannot determine video length" message box (modal: the API would hang).
        raise CommandError("the target is an image: recording needs a video (snapshot saves the swapped image; "
                           "batch_input_faces / process_jobs save images too)")
    if getattr(vp, "processing", False) or getattr(vp, "is_processing_segments", False):
        raise CommandError("playback in progress: stop it first (run_action buttonMediaPlay)")
    if int(getattr(vp, "max_frame_number", 0) or 0) <= 0:
        raise CommandError("cannot determine the video length (single-frame or unreadable video)")
    for i, pair in enumerate(getattr(mw, "job_marker_pairs", None) or []):
        # Each invalid pair would open a message box in VisoMaster.
        if pair[1] is None:
            raise CommandError(f"marker pair {i + 1} ({pair[0]}, None) has no end marker: set it, "
                               "or remove the markers in VisoMaster")
        if pair[0] >= pair[1]:
            raise CommandError(f"marker pair {i + 1} ({pair[0]}, {pair[1]}): the start must be before the end")
    if not str(mw.control.get("OutputMediaFolder", "")).strip():
        raise CommandError("no output folder: set OutputMediaFolder <folder>")
    try:
        from app.helpers import miscellaneous as misc

        if not misc.is_ffmpeg_in_path():
            raise CommandError("ffmpeg not found in PATH")
    except ImportError:
        pass
    if vp.file_type == "video":
        last = int(getattr(vp, "max_frame_number", 0) or 0)
        if from_start and int(mw.videoSeekSlider.value()) != 0:
            # Seeking renders the first frame (and may load models): let it finish first.
            mw.videoSeekSlider.setValue(0)
            raise CommandError(
                NOT_READY + "playhead moved to the first frame, VisoMaster is rendering it: "
                + WAIT_THEN + "start_recording again"
            )
        if last and int(mw.videoSeekSlider.value()) >= last - 1:
            # VisoMaster silently refuses to record from the last frame.
            raise CommandError(
                f"the playhead is at the end of the video (frame {mw.videoSeekSlider.value()}/{last}): "
                "call start_recording with from_start=true, or seek earlier"
            )
    _install_rec_hooks(mw)
    ready = _readiness(mw)
    if not ready["ready"]:
        raise CommandError(
            NOT_READY + "; ".join(ready["waiting_for"]) + ". " + WAIT_THEN + "start_recording "
            "(starting now would measure loading, not processing)"
        )
    previous = _REC.get("id")
    _REC_STARTER["by"] = "api"
    t_request = time.time()
    mw.buttonMediaRecord.setChecked(True)  # VisoMaster starts its clock inside this call
    rec = _rec_update(mw)
    if rec and rec.get("id") != previous and rec.get("started_at", 0) > t_request:
        rec["started_at"] = t_request  # elapsed_s then covers VisoMaster's own processing time
    _REC_STARTER.pop("by", None)
    if not _recording(mw) and (rec or {}).get("id") == previous:
        raise CommandError("VisoMaster did not start the recording (see its console for the reason)")
    swap = _swap_state(mw)
    res = {
        "recording": _recording(mw),
        "recording_info": _rec_view(rec),
        "next": "call " + WAIT_REC + ": it returns when the video file is written (progress meanwhile); "
                "stop_recording stops early",
    }
    if swap.get("note"):
        res["warning"] = swap["note"] + " (recording anyway, without face swap)"
    return res


@command("stop_recording", timeout=120)
def cmd_stop_recording(ctx):
    """Stop recording without the confirmation box (or cancel a batch, like its Cancel button)."""
    mw = ctx.mw
    if _BATCH.get("active"):
        _BATCH["cancel"] = True  # read by the batch loop before each item
        return {"batch_cancel_requested": True, "batch": _batch_view()}
    if not _recording(mw):
        raise CommandError("no recording in progress")
    if _rec_update(mw):
        _REC["stop_requested"] = True
    key = "ConfirmBeforeStoppingRecordingToggle"
    had = key in mw.control
    old = mw.control.get(key)
    mw.control[key] = False
    try:
        mw.buttonMediaRecord.setChecked(False)
    finally:
        if had:
            mw.control[key] = old
        else:
            mw.control.pop(key, None)
    rec = _rec_view(_rec_update(mw))
    return {"recording": _recording(mw), "result": (rec or {}).get("result"), "recording_info": rec}


# ---------------------------------------------------------------------------
# Batch processing ("Batch Process All Faces" / "Batch Process Selected Face")
# ---------------------------------------------------------------------------
# VisoMaster's own process_batch_images runs unchanged; only its dialogs are replaced while it runs:
# the "Confirm Batch Swap" question answers Yes, the modal progress window becomes _SilentProgress
# (nothing shown, also with hidden=true) and message boxes are recorded instead of shown.

_BATCH: dict = {}  # current or last batch: see _start_batch / _batch_view


class _SilentProgress:
    """Same calls as the batch's progress window; progress goes to _BATCH, Cancel = stop_recording."""

    def __init__(self, *a, **k):
        _BATCH["loop_started_at"] = time.time()

    def setValue(self, value):
        _BATCH["done"] = int(value)

    def setLabelText(self, text):
        _BATCH["current"] = str(text).split(": ", 1)[-1]

    def confirmedCanceled(self):
        return bool(_BATCH.get("cancel"))

    def __getattr__(self, name):  # setWindowModality, setWindowTitle, show, close_without_confirmation...
        return lambda *a, **k: None


def _batch_media(mw, all_faces: bool) -> tuple[int, int]:
    """(images, videos) the batch will process: same walk over the target list as VisoMaster's."""
    images = videos = 0
    lst = mw.targetVideosList
    for i in range(lst.count()):
        widget = lst.itemWidget(lst.item(i))
        kind = getattr(widget, "file_type", None)
        if kind == "image":
            images += 1
        elif kind == "video" and not all_faces:
            videos += 1
    return images, videos


def _batch_run(mw, all_faces: bool):
    try:
        from PySide6 import QtWidgets
        from app.ui.widgets import widget_components
        from app.ui.widgets.actions import common_actions
    except Exception as e:  # nothing patched yet: just end the batch
        _BATCH.update({"error": f"{type(e).__name__}: {e}", "finished_at": time.time(), "active": False})
        return

    box_class = QtWidgets.QMessageBox
    native_question = box_class.question
    orig_question = box_class.__dict__.get("question", native_question)  # the class attribute, put back as is
    orig_box = common_actions.create_and_show_messagebox
    orig_progress = widget_components.ProgressDialog

    def question(parent, title, *a, **k):
        if str(title).startswith("Confirm Batch Swap"):
            return box_class.StandardButton.Yes
        return native_question(parent, title, *a, **k)

    def message_box(parent, title, text, *a, **k):
        _BATCH.setdefault("messages", []).append({"title": str(title), "text": str(text)})
        if str(title) == "Batch Complete":
            _BATCH["loop_finished_at"] = time.time()

    try:
        box_class.question = staticmethod(question)
        common_actions.create_and_show_messagebox = message_box
        widget_components.ProgressDialog = _SilentProgress
        _vca().process_batch_images(mw, all_faces)
    except Exception as e:  # reported by wait_recording, never raised into Qt
        _BATCH["error"] = f"{type(e).__name__}: {e}"
    finally:
        _BATCH["finished_at"] = time.time()
        _BATCH["active"] = False  # first: never left stuck, whatever the restore below does
        box_class.question = orig_question
        common_actions.create_and_show_messagebox = orig_box
        widget_components.ProgressDialog = orig_progress


def _start_batch(mw, name: str, all_faces: bool) -> dict:
    """Checks what process_batch_images would report in a message box, then runs it on the next
    turn of the Qt event loop (it blocks the UI thread until the batch ends, like in VisoMaster)."""
    if _BATCH.get("active"):
        raise CommandError("a batch is already running: " + WAIT_REC + ", or stop_recording to cancel it")
    if not str(mw.control.get("OutputMediaFolder", "")).strip():
        raise CommandError("no output folder: set OutputMediaFolder <folder>")
    ready = _readiness(mw)  # first: a list still loading would be counted (and processed) partly
    if not ready["ready"]:
        raise CommandError(NOT_READY + "; ".join(ready["waiting_for"]) + ". " + WAIT_THEN + "run_action again")
    images, videos = _batch_media(mw, all_faces)
    if not images + videos:
        raise CommandError("no image" + ("" if all_faces else " or video") + " in the target list: load_target_media")
    sources = [f for f in (getattr(mw, "input_faces", {}) or {}).values() if f.isChecked()]
    sources += [e for e in (getattr(mw, "merged_embeddings", {}) or {}).values() if e.isChecked()]
    if not sources:
        raise CommandError("no input face or embedding selected: select_embedding or select_input_face first")
    _BATCH.clear()
    _BATCH.update({
        "id": f"batch-{_stamp()}", "action": name, "active": True,
        "mode": "all faces (images only)" if all_faces else "selected face (images and videos)",
        "started_at": time.time(), "total": images + videos, "images": images, "videos": videos, "done": 0,
        "output_folder": str(mw.control.get("OutputMediaFolder")),
    })
    from PySide6 import QtCore

    QtCore.QTimer.singleShot(0, lambda: _batch_run(mw, all_faces))
    return {
        "triggered": name,
        "batch": _batch_view(),
        "next": "call " + WAIT_REC + ": progress while it runs, then processing_time_s, processed / failed",
    }


def _batch_view() -> dict | None:
    b = dict(_BATCH)
    if not b:
        return None
    total, done = b.get("total") or 0, b.get("done") or 0
    total = max(total, done)  # VisoMaster's batch reads its list live: it can grow while it runs
    loop_t0 = b.get("loop_started_at")
    view = {
        "id": b["id"], "kind": "batch", "active": b["active"], "started_by": "api", "action": b["action"],
        "mode": b["mode"], "started_at": b["started_at"], "output_folder": b["output_folder"],
    }
    for k in ("images", "videos", "media", "target_face", "input_faces"):
        if k in b:
            view[k] = b[k]
    if b["active"]:
        unit = "input faces" if b["action"] == "batch_input_faces" else "media items"
        p = {"frames_written": done, "total_frames": total, "unit": unit,
             "percent": round(100.0 * done / total, 1) if total else None, "current": b.get("current")}
        if loop_t0 and done:
            elapsed = time.time() - loop_t0
            p["processing_fps"] = round(done / elapsed, 2)
            p["eta_s"] = round((total - done) * elapsed / done)
        if not loop_t0:
            p["phase"] = "starting"
        elif b.get("cancel"):
            p["phase"] = "cancelling"
        view["progress"] = p
        return view
    if "processed" in b:  # counted by the API itself (batch_input_faces)
        processed, failed, done_ok = b["processed"], b["failed"], True
    else:  # VisoMaster's batch: its summary box, "...processed: N ... Failed...: M"
        summary = next((m["text"] for m in b.get("messages", []) if m["title"] == "Batch Complete"), "")
        counts = [int(n) for n in re.findall(r":\s*(\d+)", summary)]
        processed, failed = (counts + [None, None])[:2]
        done_ok = bool(summary)
    loop_t1 = b.get("loop_finished_at")
    frames_time = round(loop_t1 - loop_t0, 2) if loop_t0 and loop_t1 else None
    state = "failed" if b.get("error") or not done_ok else ("stopped" if b.get("cancel") else "finished")
    result = {
        "state": state,
        "exists": os.path.isdir(b["output_folder"]),
        "output_path": b["output_folder"],
        "processed": processed,
        "failed": failed,
        # From the first item to the summary box, like VisoMaster's progress window; total_s adds the
        # start and the restoring of the media shown before the batch.
        "processing_time_s": frames_time,
        "average_fps": round(processed / frames_time, 2) if processed and frames_time else None,
        "total_s": round(b["finished_at"] - b["started_at"], 2),
        "messages": [m for m in b.get("messages", []) if m["title"] != "Batch Complete"],
    }
    if b.get("error"):
        result["error"] = b["error"]
    view["result"] = result
    return view


# One frame x every input face: VisoMaster has no button for it. Each step is what VisoMaster's own
# batch does for an image (assign the source to the target face, calculate_assigned_input_embedding,
# process_current_frame(synchronous=True)), then the frame is saved like "Save Image".


def _sweep_run(mw, fid: str, ids: list):
    from PIL import Image
    import numpy
    from PySide6 import QtWidgets

    vp = mw.video_processor
    face = saved_inputs = None
    processed = failed = 0
    try:
        vca = _vca()
        face = mw.target_faces[fid]  # KeyError if the user cleared the faces meanwhile: reported, batch ended
        saved_inputs = dict(face.assigned_input_faces or {})
        out_folder = vca.resolve_output_folder(mw, str(vp.media_path))
        os.makedirs(out_folder, exist_ok=True)
        stem = Path(str(vp.media_path)).stem
        jpeg = bool(mw.control.get("ImageFormatToggle"))
        _BATCH["loop_started_at"] = time.time()
        for i, iid in enumerate(ids):
            _BATCH["done"] = i
            QtWidgets.QApplication.processEvents()  # keeps the UI alive, as in VisoMaster's batch
            if _BATCH.get("cancel"):
                break
            btn = mw.input_faces.get(iid)
            name = (_card_name(btn) if btn is not None else "") or str(iid)
            _BATCH["current"] = name
            try:
                if btn is None or not getattr(btn, "embedding_store", None):
                    raise RuntimeError("input face removed or without embedding")
                face.assigned_input_faces = {iid: btn.embedding_store}
                face.calculate_assigned_input_embedding()
                vp.process_current_frame(synchronous=True)
                frame = vp.current_frame
                if not isinstance(frame, numpy.ndarray) or frame.size == 0:
                    raise RuntimeError("processing returned an empty frame")
                path = _free_path(out_folder, f"{stem}__{Path(name).stem}", ".jpg" if jpeg else ".png")
                image = Image.fromarray(frame[..., ::-1])  # same conversion as Save Image
                if jpeg:
                    image.save(path, "JPEG", quality=95)
                else:
                    image.save(path, "PNG")
                processed += 1
            except Exception as e:
                failed += 1
                _BATCH.setdefault("messages", []).append({"title": "failed", "text": f"{name}: {type(e).__name__}: {e}"})
        _BATCH["done"] = processed + failed
        _BATCH["loop_finished_at"] = time.time()
    except Exception as e:
        _BATCH["error"] = f"{type(e).__name__}: {e}"
    finally:
        # Marked ended first: whatever fails below, write commands are not refused until a restart.
        _BATCH.update({"processed": processed, "failed": failed, "finished_at": time.time(), "active": False})
        try:
            if face is not None:
                face.assigned_input_faces = saved_inputs
                face.calculate_assigned_input_embedding()
            _common().refresh_frame(mw)  # preview back to the face's own sources
        except Exception as e:
            _BATCH.setdefault("messages", []).append(
                {"title": "restore", "text": f"input faces of the target face not restored: {type(e).__name__}: {e}"})


def _free_path(folder: str, base: str, ext: str) -> str:
    """folder/base.ext, or base_2.ext, base_3.ext... : never overwrites an earlier result."""
    path, n = os.path.join(folder, base + ext), 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(folder, f"{base}_{n}{ext}")
    return path


@command("batch_input_faces", write=True)
def cmd_batch_input_faces(ctx, input_faces: list | None = None, face_id: str | None = None):
    """Swap the current frame (image, or video frame at the playhead) once per input face, each one
    alone on the target face, and save each result to OutputMediaFolder as <media>__<input face>.png
    (.jpg with ImageFormatToggle; _2, _3... added when the file exists, never overwritten). Asynchronous: follow it with wait_recording (CLI: recording-wait)."""
    mw = ctx.mw
    vp = mw.video_processor
    if _BATCH.get("active"):
        raise CommandError("a batch is already running: " + WAIT_REC + ", or stop_recording to cancel it")
    if getattr(vp, "file_type", None) not in ("video", "image") or not getattr(vp, "media_path", None):
        raise CommandError("no target image or video selected (webcam not supported)")
    if getattr(vp, "processing", False):
        raise CommandError("playback in progress: stop it first (run_action buttonMediaPlay)")
    if not str(mw.control.get("OutputMediaFolder", "")).strip():
        raise CommandError("no output folder: set OutputMediaFolder <folder>")
    faces = getattr(mw, "target_faces", {}) or {}
    fid = face_id or _selected_face(mw)
    if not fid or fid not in faces:
        raise CommandError("no target face: find faces (run_action findTargetFacesButton), then select_target_face")
    if _swap_state(mw)["enabled"] is False:
        raise CommandError("the face swap is off: every image would be the unswapped frame. "
                           "run_action swapfacesButton to turn it on")
    if faces[fid].assigned_merged_embeddings:
        raise CommandError("the target face has an embedding assigned, which VisoMaster would merge with each input "
                           "face: remove it first (select_embedding <name> selected=false)")
    ready = _readiness(mw)  # first: input faces still loading would be swept partly
    if not ready["ready"]:
        raise CommandError(NOT_READY + "; ".join(ready["waiting_for"]) + ". " + WAIT_THEN + "try again")
    inputs = getattr(mw, "input_faces", {}) or {}
    if not inputs:
        raise CommandError("no input face loaded: load_input_faces <folder>")
    if input_faces:
        by_name = {_card_name(b): i for i, b in inputs.items()}
        ids = [n if n in inputs else by_name.get(n) for n in input_faces]
        unknown = [n for n, i in zip(input_faces, ids) if i is None]
        if unknown:
            raise CommandError(f"unknown input faces: {unknown[:10]}")
    else:
        ids = list(inputs)
    _BATCH.clear()
    _BATCH.update({
        "id": f"batch-{_stamp()}", "action": "batch_input_faces", "active": True,
        "mode": "current frame x each input face", "started_at": time.time(), "total": len(ids), "done": 0,
        "media": Path(str(vp.media_path)).name, "target_face": fid, "input_faces": len(ids),
        "output_folder": str(mw.control.get("OutputMediaFolder")),
    })
    from PySide6 import QtCore

    QtCore.QTimer.singleShot(0, lambda: _sweep_run(mw, fid, ids))
    return {"started": True, "batch": _batch_view(),
            "next": "call " + WAIT_REC + ": progress while it runs, then processing_time_s, processed / failed"}


@command("list_jobs")
def cmd_list_jobs(ctx):
    """Saved jobs."""
    from app.ui.widgets.actions import job_manager_actions

    return sorted(job_manager_actions.list_jobs(ctx.mw))


@command("process_jobs", write=True, timeout=60)
def cmd_process_jobs(ctx, jobs: list | None = None):
    """Start the job queue (all jobs, or the given list)."""
    if jobs is not None and not jobs:
        raise CommandError("empty job list: give job names (see list_jobs), or leave jobs out to run them all")
    from app.ui.widgets.actions import job_manager_actions

    mw = ctx.mw
    available = job_manager_actions.list_jobs(mw)
    if not available:
        raise CommandError("no saved job")
    if jobs:
        unknown = [j for j in jobs if j not in available]
        if unknown:
            raise CommandError(f"unknown jobs: {unknown}")
        job_manager_actions.start_job_processor(mw, list(jobs))
    else:
        job_manager_actions.start_processing_all_jobs(mw)
    return {"started": jobs or sorted(available), "running": _job_running(mw)}


@command("quit", write=True)
def cmd_quit(ctx, save: bool = True):
    """Close VisoMaster like File > Quit (save=false: Quit without Saving)."""
    from PySide6 import QtCore

    mw = ctx.mw
    res = {"closing": True, "workspace_saved_on_close": bool(save)}
    if save:
        try:
            kept = _keep_last_workspace(Path(mw.last_workspace_path))  # VisoMaster overwrites it on close
            if kept:
                res["previous_kept"] = kept
        except OSError as e:
            res["warning"] = f"previous last_workspace.json not backed up: {e}"
    mw.quit_without_saving = not save
    # After this reply has been sent.
    QtCore.QTimer.singleShot(200, mw.close)
    return res


@command("help")
def cmd_help(ctx):
    """List of commands."""
    return {n: {"help": c.help, "write": c.write} for n, c in sorted(COMMANDS.items())}
