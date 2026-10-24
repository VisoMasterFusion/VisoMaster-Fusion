"""A second server on the same port must fail (Windows SO_REUSEADDR trap)."""

import sys
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import server  # needs PySide6 (portable Python)
except ImportError:  # pragma: no cover
    server = None


@unittest.skipIf(server is None, "PySide6 missing: run with the portable Python")
class ExclusivePortTests(unittest.TestCase):
    def test_second_bind_refused(self):
        first = server._ExclusiveHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        try:
            port = first.server_address[1]
            with self.assertRaises(OSError):
                server._ExclusiveHTTPServer(("127.0.0.1", port), BaseHTTPRequestHandler)
        finally:
            first.server_close()


if __name__ == "__main__":
    unittest.main()
