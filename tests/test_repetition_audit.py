from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("audit_repetition", ROOT / "scripts" / "audit_repetition.py")
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_strict_loop_definition_detects_repeated_30gram():
    phrase = " ".join(f"token{index}" for index in range(30))
    result = MODULE.analyze_text(" ".join([phrase] * 20))
    assert result.strict_loop_30gram_20x
    assert result.max_30gram_count >= 20


def test_short_warning_does_not_imply_strict_loop():
    result = MODULE.analyze_text("alpha beta gamma delta epsilon zeta eta theta " * 4)
    assert result.short_repetition_8gram_4x
    assert not result.strict_loop_30gram_20x


def test_think_tag_statuses():
    assert MODULE.think_tag_status("plain answer") == "no_think_tags"
    assert MODULE.think_tag_status("<think>unfinished") == "unclosed_think"
    assert MODULE.think_tag_status("<think> </think>answer") == "empty_think"
    assert MODULE.think_tag_status("<think>reason</think>answer") == "complete_think"
