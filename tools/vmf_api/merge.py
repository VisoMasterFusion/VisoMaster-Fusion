"""Config import logic: personal-key exclusion, file parsing, diff/merge.

Pure Python (stdlib only), no Qt: used by the in-app commands, by the offline
CLI (apply-offline) and by the tests.
"""

from __future__ import annotations

import copy
import fnmatch
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from schema import Schema, values_equal

# ---------------------------------------------------------------------------
# Personal keys
# ---------------------------------------------------------------------------

_WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]")
_EXT = re.compile(r"\.[A-Za-z0-9]{1,5}$")


def looks_like_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v:
        return False
    if _WIN_ABS.match(v) or v.startswith("\\\\") or v.startswith("/"):
        return True
    if v.startswith("./") or v.startswith(".\\") or v.startswith("~"):
        return True
    if "\\" in v:
        return True
    if "/" in v and _EXT.search(v):
        return True
    return False


@dataclass
class PersonalRules:
    groups: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    exclude_path_like_values: bool = True
    force_include: list[str] = field(default_factory=list)
    key_notes: dict = field(default_factory=dict)  # key -> why it is personal

    @classmethod
    def load(cls, path: str | os.PathLike) -> "PersonalRules":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        keys = data.get("keys", [])
        # "keys" is a list of names or a {name: why} object.
        notes = {k: str(v) for k, v in keys.items() if v} if isinstance(keys, dict) else {}
        return cls(
            groups=list(data.get("groups", [])),
            keys=list(keys),
            patterns=list(data.get("patterns", [])),
            exclude_path_like_values=bool(data.get("exclude_path_like_values", True)),
            force_include=list(data.get("force_include", [])),
            key_notes=notes,
        )

    def reason(
        self,
        key: str,
        schema: Schema | None = None,
        incoming: Any = None,
        current: Any = None,
    ) -> str | None:
        """Why `key` is personal (never imported nor overwritten), or None. Says which rule of
        personal_keys.json matched, so the user knows what to edit to change it."""
        if key in self.force_include:
            return None
        if key in self.keys:
            why = self.key_notes.get(key) or "user or machine-specific setting"
            return f"{why} (listed in personal_keys.json 'keys')"
        if schema is not None:
            e = schema.get(key)
            # Groups name sections of the Settings tab only.
            if e is not None and e.layout == "SETTINGS_LAYOUT_DATA" and e.group in self.groups:
                return (f"part of the Settings section '{e.group}', which holds this user's own "
                        "preferences (personal_keys.json 'groups')")
        for pat in self.patterns:
            if fnmatch.fnmatchcase(key, pat):
                return (f"name matches '{pat}': paths, GPU, webcam or UI settings depend on this "
                        "machine (personal_keys.json 'patterns')")
        if self.exclude_path_like_values and (looks_like_path(incoming) or looks_like_path(current)):
            return ("value is a file or folder path, which only exists on one machine "
                    "(personal_keys.json 'exclude_path_like_values')")
        return None


# ---------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------


@dataclass
class ImportedConfig:
    fmt: str  # preset | workspace | preset_folder | flat_params | flat_control
    control: dict | None = None
    parameters: dict | None = None
    source_face: str | None = None  # workspace: face whose parameters are used
    faces: dict = field(default_factory=dict)  # workspace: face_id -> params
    notes: list[str] = field(default_factory=list)


def is_workspace(data: dict) -> bool:
    return any(k in data for k in ("target_faces_data", "target_medias_data", "input_faces_data"))


def parse_config(path: str | os.PathLike, schema: Schema | None = None, source_face: str | None = None) -> ImportedConfig:
    """Read a preset / workspace / VisoMaster presets-folder file."""
    path = os.fspath(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("the JSON file must contain an object")

    if is_workspace(data):
        faces = {
            fid: dict(fd.get("parameters") or {})
            for fid, fd in (data.get("target_faces_data") or {}).items()
            if isinstance(fd, dict)
        }
        cfg = ImportedConfig(fmt="workspace", control=dict(data.get("control") or {}), faces=faces)
        chosen = source_face or data.get("selected_target_face_id")
        if source_face and source_face not in faces:
            raise ValueError(f"source face '{source_face}' not in the workspace (available: {list(faces)})")
        if chosen in faces:
            cfg.source_face = chosen
        elif faces:
            cfg.source_face = next(iter(faces))
        if cfg.source_face is not None:
            cfg.parameters = faces[cfg.source_face]
            if len(faces) > 1:
                cfg.notes.append(
                    f"workspace with {len(faces)} faces: parameters taken from '{cfg.source_face}' (--source-face to change)"
                )
        elif isinstance(data.get("current_widget_parameters"), dict):
            cfg.parameters = dict(data["current_widget_parameters"])
            cfg.notes.append("no target face: parameters taken from current_widget_parameters")
        cfg.notes.append("ignored: media, faces, embeddings, K/V maps, markers, window state")
        return cfg

    if "parameters" in data or "control" in data:
        return ImportedConfig(
            fmt="preset",
            control=dict(data["control"]) if isinstance(data.get("control"), dict) else None,
            parameters=dict(data["parameters"]) if isinstance(data.get("parameters"), dict) else None,
        )

    # Flat dict: VisoMaster "presets/<name>.json" (+ "<name>_ctl.json") or a bare dict.
    if schema is not None:
        n_param = sum(1 for k in data if schema.scope_of(k) == "parameter")
        n_ctl = sum(1 for k in data if schema.scope_of(k) == "control")
    else:
        n_param, n_ctl = len(data), 0
    if n_ctl > n_param:
        return ImportedConfig(fmt="flat_control", control=dict(data))
    cfg = ImportedConfig(fmt="flat_params", parameters=dict(data))
    root, ext = os.path.splitext(path)
    ctl_path = root + "_ctl" + ext
    if os.path.isfile(ctl_path):
        with open(ctl_path, encoding="utf-8") as f:
            ctl = json.load(f)
        if isinstance(ctl, dict):
            cfg.control = ctl
            cfg.fmt = "preset_folder"
            cfg.notes.append(f"controls read from {os.path.basename(ctl_path)}")
    return cfg


# ---------------------------------------------------------------------------
# Diff / merge
# ---------------------------------------------------------------------------


@dataclass
class SectionPlan:
    scope: str
    changed: dict = field(default_factory=dict)  # key -> {"from", "to"}
    excluded: dict = field(default_factory=dict)  # key -> {"reason", "incoming", "current"}
    unknown: dict = field(default_factory=dict)  # key -> incoming value (obsolete / not in this version)
    invalid: dict = field(default_factory=dict)  # key -> {"incoming", "error"}
    missing: list = field(default_factory=list)  # keys of mine absent from the file
    unchanged: int = 0
    missing_policy: str = "keep"
    defaults_applied: dict = field(default_factory=dict)  # key -> {"from","to"} (missing_policy=default)

    @property
    def updates(self) -> dict:
        out = {k: v["to"] for k, v in self.changed.items()}
        out.update({k: v["to"] for k, v in self.defaults_applied.items()})
        return out

    def to_dict(self, verbose: bool = True) -> dict:
        d = {
            "scope": self.scope,
            "counts": {
                "changed": len(self.changed),
                "excluded": len(self.excluded),
                "unknown": len(self.unknown),
                "invalid": len(self.invalid),
                "missing": len(self.missing),
                "unchanged": self.unchanged,
                "defaults_applied": len(self.defaults_applied),
            },
            "changed": self.changed,
            "excluded": self.excluded,
            # Names only: an unknown key is useless to import, and its value could be anything the
            # file holds (diff reads any JSON the caller names).
            "unknown": sorted(self.unknown),
            "invalid": self.invalid,
            "missing": self.missing,
        }
        if self.defaults_applied:
            d["defaults_applied"] = self.defaults_applied
        return d


def plan_section(
    incoming: dict,
    current: dict,
    scope: str,
    schema: Schema,
    rules: PersonalRules,
    defaults: dict | None = None,
    missing_policy: str = "keep",
) -> SectionPlan:
    """Compare a section of the imported file with my current values.

    current  : my values (reference keys = what this VisoMaster version knows)
    defaults : default values, used when missing_policy == "default"
    """
    plan = SectionPlan(scope=scope, missing_policy=missing_policy)
    for key, value in incoming.items():
        if key not in current:
            plan.unknown[key] = value
            continue
        cur = current[key]
        reason = rules.reason(key, schema, value, cur)
        if reason:
            if not values_equal(cur, value):
                plan.excluded[key] = {"reason": reason, "incoming": value, "current": cur}
            continue
        ok, coerced, msg = schema.validate(key, value)
        if not ok:
            plan.invalid[key] = {"incoming": value, "error": msg}
            continue
        if values_equal(cur, coerced):
            plan.unchanged += 1
        else:
            plan.changed[key] = {"from": cur, "to": coerced}
    for key in current:
        if key in incoming:
            continue
        plan.missing.append(key)
        if missing_policy == "default" and defaults is not None and key in defaults:
            if rules.reason(key, schema, defaults[key], current[key]):
                continue
            if not values_equal(current[key], defaults[key]):
                plan.defaults_applied[key] = {"from": current[key], "to": copy.deepcopy(defaults[key])}
    plan.missing.sort()
    return plan


def strip_personal(values: dict, schema: Schema, rules: PersonalRules) -> tuple[dict, list]:
    """Copy of `values` without personal keys (for export)."""
    kept, removed = {}, []
    for k, v in values.items():
        if rules.reason(k, schema, v, None):
            removed.append(k)
        else:
            kept[k] = copy.deepcopy(v)
    return kept, sorted(removed)


# ---------------------------------------------------------------------------
# Offline: merge directly into a last_workspace.json dict
# ---------------------------------------------------------------------------


def select_workspace_faces(ws: dict, faces: str) -> list[str]:
    all_faces = list((ws.get("target_faces_data") or {}).keys())
    if faces == "all":
        return all_faces
    if faces == "selected":
        sel = ws.get("selected_target_face_id")
        return [sel] if sel in all_faces else []
    if faces not in all_faces:
        raise ValueError(f"face '{faces}' not in the workspace (available: {all_faces})")
    return [faces]


def merge_into_workspace(
    ws: dict,
    imported: ImportedConfig,
    schema: Schema,
    rules: PersonalRules,
    faces: str = "selected",
    do_control: bool = True,
    do_params: bool = True,
    missing_policy: str = "keep",
) -> tuple[dict, dict]:
    """Return (new_workspace, report). `ws` is not modified."""
    new = copy.deepcopy(ws)
    report: dict = {"format": imported.fmt, "notes": list(imported.notes)}
    param_defaults = {k: e.default for k, e in schema.entries.items() if e.scope == "parameter"}
    control_defaults = {k: e.default for k, e in schema.entries.items() if e.scope == "control"}

    if do_control and imported.control is not None:
        cur = new.setdefault("control", {})
        plan = plan_section(imported.control, cur, "control", schema, rules, control_defaults, missing_policy)
        cur.update(plan.updates)
        report["control"] = plan.to_dict()
    elif do_control:
        report["control"] = "not in file"

    if do_params and imported.parameters is not None:
        targets = select_workspace_faces(new, faces)
        report["parameters"] = {}
        for fid in targets:
            fd = new["target_faces_data"][fid]
            cur = fd.setdefault("parameters", {})
            # A face saved by an older version may lack keys: use schema keys as reference too.
            ref = dict(param_defaults)
            ref.update(cur)
            plan = plan_section(imported.parameters, ref, "parameter", schema, rules, param_defaults, missing_policy)
            cur.update(plan.updates)
            report["parameters"][fid] = plan.to_dict()
        # Template for new faces / panel shown when no face is selected.
        cwp = new.get("current_widget_parameters")
        if isinstance(cwp, dict) and (faces in ("all", "selected") or not targets):
            ref = dict(param_defaults)
            ref.update(cwp)
            plan = plan_section(imported.parameters, ref, "parameter", schema, rules, param_defaults, missing_policy)
            cwp.update(plan.updates)
            report["parameters"]["<current_widget_parameters>"] = plan.to_dict()
        if not targets:
            report.setdefault("notes", []).append(f"no target face for --faces {faces}")
    elif do_params:
        report["parameters"] = "not in file"
    return new, report
