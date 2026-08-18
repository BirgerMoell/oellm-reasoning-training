from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "promote_data.py"


class PromoteDataTest(unittest.TestCase):
    def test_promotes_matching_complete_manifest_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "isolated" / "data" / "reasoning-v1"
            source.mkdir(parents=True)
            train = source / "train.parquet"
            train.write_bytes(b"parquet")
            (source / "dedup.sqlite3").write_bytes(b"sqlite")
            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "repository_git_sha": "a" * 40,
                        "selected_tokens": 101,
                        "target_tokens": 100,
                        "output": {
                            "path": str(train),
                            "bytes": train.stat().st_size,
                            "sha256": "b" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            destination = root / "canonical" / "data" / "reasoning-v1"
            command = [
                sys.executable,
                str(SCRIPT),
                "--source",
                str(source),
                "--destination",
                str(destination),
                "--expected-repository-sha",
                "a" * 40,
            ]
            first = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("PROMOTED", first.stdout)
            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.resolve(), source.resolve())

            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Refusing existing promotion destination", second.stderr)


if __name__ == "__main__":
    unittest.main()
