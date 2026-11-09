"""Tests for the pure helper functions in evals/run_eval.py."""
from evals.run_eval import first_match_rank, parse_verdict, summarize


def test_first_match_rank():
    found = ["site/faq.md", "site/pricing.md", "help/buy-hours.md"]
    assert first_match_rank(found, ["site/pricing.md"]) == 2
    assert first_match_rank(found, ["help/buy-hours.md", "site/faq.md"]) == 1
    assert first_match_rank(found, ["site/contact.md"]) is None


def test_parse_verdict_reads_json_inside_extra_text():
    assert parse_verdict('Sure! {"pass": true, "reason": "Correct price."}') == {
        "pass": True, "reason": "Correct price."
    }


def test_unreadable_verdict_counts_as_fail():
    verdict = parse_verdict("I think it is fine")
    assert verdict["pass"] is False
    assert "could not be read" in verdict["reason"]


def test_summarize_accuracy_by_category():
    def row(category, passed):
        return {"category": category, "pass": passed, "cost": 0.001, "seconds": 2.0,
                "input_tokens": 1000, "searched": True, "judge_cost": 0.0005}

    summary = summarize([row("visitor", True), row("visitor", False), row("tutor", True)])

    assert summary["accuracy"] == 2 / 3
    assert summary["by_category"] == {"tutor": 1.0, "visitor": 0.5}


def test_normalize_ignores_bold_case_and_curly_apostrophes():
    from evals.run_eval import normalize
    assert normalize("Press **Reset Password** if you can’t") == "press reset password if you can't"


def test_retrieval_metrics_skip_questions_without_sources():
    from evals.run_eval import evaluate_retrieval
    off_topic = [{"id": "x", "question": "weather?", "sources": [], "evidence": []}]
    assert evaluate_retrieval(off_topic) is None        # nothing to score, and no crash


def test_aws_errors_are_reported_separately_from_accuracy():
    ok = {"category": "visitor", "pass": True, "cost": 0.001, "seconds": 2.0,
          "input_tokens": 1000, "searched": True, "judge_cost": 0.0005}
    throttled = {**ok, "pass": False, "error": "ThrottlingException", "judge_cost": 0.0}

    summary = summarize([ok, throttled])

    assert summary["accuracy"] == 1.0
    assert summary["errors"] == 1


def test_summary_survives_every_question_erroring():
    failed = {"category": "visitor", "pass": False, "error": "AccessDenied", "cost": 0.0,
              "seconds": 0.0, "input_tokens": 0, "searched": False, "judge_cost": 0.0}
    summary = summarize([failed, failed])
    assert summary["errors"] == 2
    assert summary["accuracy"] == 0.0
