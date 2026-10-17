"""docs/REFERENCE.md must match the code (regenerate with gen_docs.py)."""

import sys
import unittest
from pathlib import Path

VMF = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VMF))

try:
    import mcp  # noqa: F401  (the MCP section needs the SDK: vmf_api\.venv)
except ImportError:  # pragma: no cover
    mcp = None


@unittest.skipIf(mcp is None, "mcp SDK missing: run with vmf_api\\.venv")
class ReferenceTests(unittest.TestCase):
    def test_reference_up_to_date(self):
        import gen_docs

        current = (VMF / "docs" / "REFERENCE.md").read_text(encoding="utf-8")
        self.assertEqual(current, gen_docs.render(),
                         "docs/REFERENCE.md is out of date: run .venv\\Scripts\\python.exe gen_docs.py")


if __name__ == "__main__":
    unittest.main()
