from __future__ import annotations

import sys
import unittest
from pathlib import Path

from datasets import Dataset, concatenate_datasets


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_mix import normalize_dataset  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        assert not add_generation_prompt
        return list(range(sum(len(message["content"]) for message in messages)))


def source(source_id: str) -> dict:
    return {
        "id": source_id,
        "adapter": "messages",
        "language": "en",
        "role": "instruction_replay",
        "require_reasoning_trace": False,
    }


class BuildMixTest(unittest.TestCase):
    def test_normalization_projects_nested_messages_to_one_canonical_schema(self) -> None:
        rich_messages = Dataset.from_list(
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is two plus two?",
                            "function_calls": None,
                            "functions": None,
                        },
                        {
                            "role": "assistant",
                            "content": "Four.",
                            "function_calls": None,
                            "functions": None,
                        },
                    ]
                }
            ]
        )
        plain_messages = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Name a prime number."},
                        {"role": "assistant", "content": "Two."},
                    ]
                }
            ]
        )

        rich, _ = normalize_dataset(
            rich_messages, source("rich"), FakeTokenizer(), 1, 1_000, 1
        )
        plain, _ = normalize_dataset(
            plain_messages, source("plain"), FakeTokenizer(), 1, 1_000, 1
        )
        combined = concatenate_datasets([rich, plain])

        self.assertEqual(len(combined), 2)
        self.assertEqual(set(combined[0]["messages"][0]), {"role", "content"})
        self.assertEqual(rich.features, plain.features)


if __name__ == "__main__":
    unittest.main()
