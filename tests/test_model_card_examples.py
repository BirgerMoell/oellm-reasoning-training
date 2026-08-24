from scripts.generate_model_card_examples import split_reasoning


def test_split_reasoning_complete_tags() -> None:
    reasoning, answer, trace_format = split_reasoning(
        "<think>First calculate 3 × 4.</think>\nFinal answer: 12"
    )
    assert reasoning == "First calculate 3 × 4."
    assert answer == "Final answer: 12"
    assert trace_format == "complete_think_tags"


def test_split_reasoning_does_not_invent_tags() -> None:
    reasoning, answer, trace_format = split_reasoning("The answer is 12.")
    assert reasoning is None
    assert answer == "The answer is 12."
    assert trace_format == "no_think_tags"


def test_split_reasoning_reports_truncation() -> None:
    reasoning, answer, trace_format = split_reasoning("<think>Still working")
    assert reasoning == "Still working"
    assert answer == ""
    assert trace_format == "unclosed_think_tag"
