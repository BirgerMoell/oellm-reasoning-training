from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_config", ROOT / "scripts" / "validate_config.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConfigTest(unittest.TestCase):
    def test_repository_recipes_are_consistent(self) -> None:
        self.assertEqual(MODULE.validate(), [])


if __name__ == "__main__":
    unittest.main()
