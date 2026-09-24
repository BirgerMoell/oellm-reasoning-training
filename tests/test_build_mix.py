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
    def test_medqa_adapter_keeps_only_correct_completed_reasoning(self) -> None:
        rows = Dataset.from_list(
            [
                {
                    "question": "Which treatment is indicated?",
                    "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                    "reasoning": (
                        "The clinical findings support option B because the alternatives conflict "
                        "with the history, examination, and expected mechanism. The remaining choice "
                        "best matches the presentation and therefore should be selected."
                    ),
                    "response": "Answer: (B)",
                    "correct": True,
                    "finish_reason": "stop",
                },
                {
                    "question": "Which diagnosis is most likely?",
                    "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                    "reasoning": "An incorrect trace.",
                    "response": "Answer: (A)",
                    "correct": False,
                    "finish_reason": "stop",
                },
            ]
        )
        spec = source("medqa")
        spec.update(
            {
                "adapter": "medqa_correct_reasoning",
                "role": "medical_reasoning",
                "require_reasoning_trace": True,
                "reasoning_format": "think_and_answer",
                "reject_strict_repetition": True,
            }
        )

        normalized, reasons = normalize_dataset(rows, spec, FakeTokenizer(), 1, 10_000, 1)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(reasons["medqa_incorrect"], 1)
        self.assertIn("(B) Beta", normalized[0]["messages"][0]["content"])
        self.assertTrue(normalized[0]["messages"][1]["content"].startswith("<think>"))
        self.assertTrue(normalized[0]["messages"][1]["content"].endswith("Answer: (B)"))

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
