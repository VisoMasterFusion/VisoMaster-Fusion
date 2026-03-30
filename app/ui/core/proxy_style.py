from PySide6 import QtWidgets
from PySide6.QtCore import Qt


class ProxyStyle(QtWidgets.QProxyStyle):
    # Re-entry guard: QProxyStyle::styleHint() (C++) re-invokes our Python
    # override via virtual dispatch, causing infinite recursion in PySide6.
    _in_style_hint: bool = False

    def styleHint(self, hint, opt=None, widget=None, returnData=None) -> int:
        # Handle our custom hint directly — no super() call needed.
        if hint == self.StyleHint.SH_Slider_AbsoluteSetButtons:
            return Qt.LeftButton.value
        # Guard against re-entrant calls caused by PySide6's virtual dispatch.
        if ProxyStyle._in_style_hint:
            return 0
        ProxyStyle._in_style_hint = True
        try:
            return super().styleHint(hint, opt, widget, returnData)
        except Exception:
            return 0
        finally:
            ProxyStyle._in_style_hint = False
