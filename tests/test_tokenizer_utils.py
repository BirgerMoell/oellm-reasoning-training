from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tokenizer_utils import load_local_tokenizer  # noqa: E402


class TokenizerUtilsTest(unittest.TestCase):
    def test_legacy_extra_special_token_list_uses_json_backend_directly(self) -> None:
        vocabulary = {
            "<unk>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<|im_start|>": 3,
            "<|im_end|>": 4,
            "<pad>": 5,
        }
        backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            backend.save(str(model / "tokenizer.json"))
            (model / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "LlamaTokenizer",
                        "model_max_length": 262144,
                        "unk_token": "<unk>",
                        "bos_token": "<bos>",
                        "eos_token": "<eos>",
                        "pad_token": "<pad>",
                        "extra_special_tokens": [
                            "<unk>",
                            "<bos>",
                            "<eos>",
                            "<|im_start|>",
                            "<|im_end|>",
                            "<pad>",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch("tokenizer_utils.AutoTokenizer.from_pretrained", side_effect=TypeError):
                tokenizer = load_local_tokenizer(model)

        self.assertEqual(tokenizer.convert_tokens_to_ids("<|im_start|>"), 3)
        self.assertEqual(tokenizer.convert_tokens_to_ids("<|im_end|>"), 4)
        self.assertEqual(tokenizer.pad_token_id, 5)
        self.assertEqual(tokenizer.model_max_length, 262144)


if __name__ == "__main__":
    unittest.main()
