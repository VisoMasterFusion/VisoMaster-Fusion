"""Settings schema for VisoMaster Fusion, built from its *_LAYOUT_DATA dicts.

Two sources, same output:
  * from_layouts(): the live dicts, when running inside VisoMaster (exact
    options, including the ones computed at import time);
  * from_source(): static AST read of the layout files, no import at all
    (used by the offline CLI and the tests: no Qt, no torch, no cv2).

Stdlib only.
"""

from __future__ import annotations

import ast
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

# layout module -> (variable name, scope). Scope comes from main_ui.py:
# add_widgets_to_tab_layout(..., data_type="control"|"parameter").
LAYOUT_SOURCES = (
    ("settings_layout_data", "SETTINGS_LAYOUT_DATA", "control"),
    ("denoiser_layout_data", "DENOISER_LAYOUT_DATA", "control"),
    ("common_layout_data", "COMMON_LAYOUT_DATA", "parameter"),
    ("swapper_layout_data", "SWAPPER_LAYOUT_DATA", "parameter"),
    ("face_editor_layout_data", "FACE_EDITOR_LAYOUT_DATA", "parameter"),
)

_SPEC_FIELDS = (
    "default", "min_value", "max_value", "options", "decimals", "step",
    "label", "help", "parentToggle", "requiredToggleValue", "parentSelection", "requiredSelectionValue",
)
# Tab of the VisoMaster UI that shows each layout.
LAYOUT_TABS = {
    "SETTINGS_LAYOUT_DATA": "Settings",
    "DENOISER_LAYOUT_DATA": "Denoiser",
    "COMMON_LAYOUT_DATA": "Common",
    "SWAPPER_LAYOUT_DATA": "Face Swap",
    "FACE_EDITOR_LAYOUT_DATA": "Face Editor",
}


def widget_kind(key: str) -> str:
    """Same precedence as layout_actions.add_widgets_to_tab_layout."""
    if "Toggle" in key:
        return "toggle"
    if "Selection" in key:
        return "selection"
    if "DecimalSlider" in key:
        return "decimal"
    if "Slider" in key:
        return "int"
    if "Text" in key:
        return "text"
    return "unknown"


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        n = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if math.isfinite(n) else None  # nan / inf pass every min/max comparison


@dataclass
class Entry:
    key: str
    scope: str  # "control" | "parameter"
    layout: str
    group: str
    kind: str
    default: Any = None
    min: float | None = None
    max: float | None = None
    options: list | None = None
    decimals: int | None = None
    label: str | None = None  # text shown in the UI
    help: str | None = None  # tooltip
    tab: str | None = None
    # Shown (and used) only when these conditions hold: {"all": [{key, value}], "any": [{key, value}]}
    depends_on: dict | None = None
    # Buttons next to / below the widget (e.g. "Apply"): [{label, help}], see press_parameter_button
    buttons: list | None = None
    # Options / default computed by VisoMaster at runtime (e.g. the files of a models folder)
    options_note: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Schema:
    entries: dict[str, Entry] = field(default_factory=dict)
    source: str = ""

    def get(self, key: str) -> Entry | None:
        return self.entries.get(key)

    def group_of(self, key: str) -> str | None:
        e = self.entries.get(key)
        return e.group if e else None

    def scope_of(self, key: str) -> str | None:
        e = self.entries.get(key)
        return e.scope if e else None

    def keys(self, scope: str | None = None) -> list[str]:
        return [k for k, e in self.entries.items() if scope is None or e.scope == scope]

    def groups(self, layout: str | None = None) -> list[str]:
        seen: list[str] = []
        for e in self.entries.values():
            if (layout is None or e.layout == layout) and e.group not in seen:
                seen.append(e.group)
        return seen

    # ---------- validation ----------

    def validate(self, key: str, value: Any, coerce_strings: bool = False):
        """Return (ok, value, message). Unknown keys are accepted as-is.

        coerce_strings: accept "true"/"12"/"0.5" style strings (CLI input).
        """
        e = self.entries.get(key)
        kind = e.kind if e else widget_kind(key)
        if kind == "toggle":
            if isinstance(value, bool):
                return True, value, ""
            if coerce_strings and isinstance(value, str):
                low = value.strip().lower()
                if low in ("1", "true", "yes", "on", "oui"):
                    return True, True, ""
                if low in ("0", "false", "no", "off", "non"):
                    return True, False, ""
            return False, value, f"{key}: boolean expected, got {value!r}"
        if kind in ("int", "decimal"):
            n = _num(value) if (coerce_strings or not isinstance(value, str)) else None
            if n is None:
                return False, value, f"{key}: number expected, got {value!r}"
            # Some layouts declare a default outside [min, max] (e.g.
            # FaceExpressionNormalizeLipsThresholdDecimalSlider: 0.03 < 0.10);
            # VisoMaster stores it as-is, so the declared default is always valid.
            is_default = e is not None and _num(e.default) is not None and abs(_num(e.default) - n) <= 1e-9
            if e is not None and not is_default:
                if e.min is not None and n < e.min - 1e-9:
                    return False, value, f"{key}: {n} < min {e.min:g}"
                if e.max is not None and n > e.max + 1e-9:
                    return False, value, f"{key}: {n} > max {e.max:g}"
            if kind == "int":
                if abs(n - round(n)) > 1e-9:
                    return False, value, f"{key}: integer expected, got {value!r}"
                return True, int(round(n)), ""
            return True, float(n), ""
        if kind == "selection":
            if not isinstance(value, str):
                if coerce_strings:
                    value = str(value)
                else:
                    return False, value, f"{key}: text expected, got {value!r}"
            if e is not None and e.options:
                if value not in e.options:
                    return False, value, f"{key}: {value!r} is not one of {e.options}"
            return True, value, ""
        if kind == "text":
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                return True, value if not coerce_strings else str(value), ""
            return False, value, f"{key}: text expected, got {value!r}"
        return True, value, ""


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b and type(a) is type(b)
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None and not (isinstance(a, str) and isinstance(b, str)):
        return abs(na - nb) <= 1e-9
    return a == b


# ---------- builders ----------


def _typed_default(kind: str, default: Any) -> Any:
    """Layout files write most defaults as strings ("4", "2.5"); type them the way
    layout_actions stores them in control / default_parameters."""
    if default is None:
        return None
    try:
        if kind == "toggle":
            return bool(default)
        if kind == "int":
            return int(float(default))
        if kind == "decimal":
            return float(default)
        if kind in ("selection", "text"):
            return str(default)
    except (TypeError, ValueError):
        pass
    return default


def _text(v: Any) -> str | None:
    return " ".join(v.split()) if isinstance(v, str) and v.strip() else None


def depends_on(spec: dict) -> dict | None:
    """VisoMaster's visibility rules (parentToggle "A & B" / "A | B" / "A" + requiredToggleValue,
    parentSelection + requiredSelectionValue) as {"all": [...], "any": [...]} of {key, value}."""
    all_: list = []
    any_: list = []
    sels = spec.get("parentSelection")
    vals = spec.get("requiredSelectionValue")
    if isinstance(sels, str):
        sels = [sels]
    if isinstance(vals, str):
        vals = [vals]
    if isinstance(sels, (list, tuple)) and isinstance(vals, (list, tuple)):
        all_ += [{"key": k, "value": v} for k, v in zip(sels, vals) if isinstance(k, str)]
    toggles = spec.get("parentToggle")
    if isinstance(toggles, str) and toggles.strip():
        if "|" in toggles:
            any_ += [{"key": t.strip(), "value": True} for t in toggles.split("|") if t.strip()]
        elif "&" in toggles:
            all_ += [{"key": t.strip(), "value": True} for t in toggles.split("&") if t.strip()]
        else:
            req = spec.get("requiredToggleValue", True)
            all_.append({"key": toggles.strip(), "value": bool(req) if isinstance(req, bool) else True})
    out = {}
    if all_:
        out["all"] = all_
    if any_:
        out["any"] = any_
    return out or None


BUTTON_FIELDS = ("action_button", "below_row_button")


def _buttons(spec: dict) -> list | None:
    out = []
    for f in BUTTON_FIELDS:
        b = spec.get(f)
        if isinstance(b, dict) and isinstance(b.get("label"), str):
            out.append({"label": b["label"], "help": _text(b.get("help"))})
    return [{k: v for k, v in b.items() if v} for b in out] or None


def _entry(key: str, spec: dict, scope: str, layout: str, group: str) -> Entry:
    kind = widget_kind(key)
    opts = spec.get("options")
    dynamic = callable(opts) or callable(spec.get("default"))
    options = list(opts) if isinstance(opts, (list, tuple)) else None
    if options is not None and not all(isinstance(o, str) for o in options):
        options = None
    if kind == "unknown" and options:
        kind = "selection"
    dec = spec.get("decimals")
    lo, hi = _num(spec.get("min_value")), _num(spec.get("max_value"))
    if kind == "int":
        lo = int(lo) if lo is not None else None
        hi = int(hi) if hi is not None else None
    return Entry(
        key=key,
        scope=scope,
        layout=layout,
        group=group,
        kind=kind,
        default=None if callable(spec.get("default")) else _typed_default(kind, spec.get("default")),
        min=lo,
        max=hi,
        options=options,
        decimals=int(dec) if isinstance(dec, (int, float)) and not isinstance(dec, bool) else None,
        label=_text(spec.get("label")),
        help=_text(spec.get("help")),
        tab=LAYOUT_TABS.get(layout),
        depends_on=depends_on(spec),
        buttons=_buttons(spec),
        options_note="computed by VisoMaster at runtime (e.g. files found in a models folder)" if dynamic else None,
    )


def from_layouts(layouts: Iterable[tuple[str, dict, str]], source: str = "live") -> Schema:
    """layouts: iterable of (layout_name, LAYOUT_DATA dict, scope)."""
    schema = Schema(source=source)
    for layout_name, data, scope in layouts:
        for group, widgets in (data or {}).items():
            if not isinstance(widgets, dict):
                continue
            for key, spec in widgets.items():
                if isinstance(key, str) and isinstance(spec, dict):
                    schema.entries[key] = _entry(key, spec, scope, layout_name, str(group))
    return schema


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _find_layout_dict(tree: ast.Module, var: str) -> ast.Dict | None:
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == var and isinstance(node.value, ast.Dict):
                return node.value
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == var and isinstance(node.value, ast.Dict):
                    return node.value
    return None


def from_source(app_dir: str | os.PathLike) -> Schema:
    """Parse app/ui/widgets/*_layout_data.py without importing them."""
    widgets_dir = os.path.join(os.fspath(app_dir), "app", "ui", "widgets")
    layouts = []
    for module, var, scope in LAYOUT_SOURCES:
        path = os.path.join(widgets_dir, module + ".py")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        root = _find_layout_dict(tree, var)
        if root is None:
            continue
        data: dict = {}
        for gk, gv in zip(root.keys, root.values):
            group = _literal(gk) if gk is not None else None
            if not isinstance(group, str) or not isinstance(gv, ast.Dict):
                continue
            widgets: dict = {}
            for wk, wv in zip(gv.keys, gv.values):
                key = _literal(wk) if wk is not None else None
                if not isinstance(key, str) or not isinstance(wv, ast.Dict):
                    continue
                spec = {}
                for pk, pv in zip(wv.keys, wv.values):
                    pname = _literal(pk) if pk is not None else None
                    if pname in _SPEC_FIELDS:
                        spec[pname] = _literal(pv)
                    elif pname in BUTTON_FIELDS and isinstance(pv, ast.Dict):
                        spec[pname] = {
                            _literal(bk): _literal(bv) for bk, bv in zip(pv.keys, pv.values) if bk is not None
                        }
                widgets[key] = spec
            data[group] = widgets
        layouts.append((var, data, scope))
    return from_layouts(layouts, source=f"ast:{widgets_dir}")
