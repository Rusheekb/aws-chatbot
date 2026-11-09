"""Tests for the Streamlit app, run headless with Streamlit's AppTest and a fake assistant."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import bot
import rag

APP = str(Path(__file__).parent.parent / "app.py")


class FakeTurn:
    """Stands in for bot.Turn: answers instantly and costs nothing."""

    def __init__(self, messages):
        self.messages = messages
        self.sources = ["site/pricing.md"]
        self.stop_reason = "end_turn"
        self.input_tokens = 100
        self.output_tokens = 20

    def stream(self):
        self.messages.append({"role": "assistant", "content": [{"text": "Packs start at $140."}]})
        yield "Packs start at $140."

    def cost(self):
        return 0.0001


@pytest.fixture(autouse=True)
def offline_app(monkeypatch):
    monkeypatch.setattr(rag, "ensure_index", list)
    monkeypatch.setattr(bot, "Turn", FakeTurn)
    monkeypatch.delenv("APP_PASSWORD", raising=False)


def run_app():
    return AppTest.from_file(APP, default_timeout=10).run()


def test_new_visitors_see_a_welcome_and_starter_questions():
    at = run_app()
    assert not at.exception
    assert "Hi! I can tell you about Learn2Lead" in at.chat_message[0].markdown[0].value
    assert {"What is Learn2Lead?", "How much does tutoring cost?",
            "What subjects do you offer?", "How do I join my class?"} <= {b.label for b in at.button}


def test_a_question_gets_an_answer_with_sources():
    at = run_app()
    at.chat_input[0].set_value("How much does tutoring cost?").run()

    assert not at.exception
    assert at.chat_message[-1].markdown[0].value == "Packs start at $140."
    assert "Sources: site/pricing.md" in [c.value for c in at.chat_message[-1].caption]


def test_password_gate(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "open-sesame")
    at = run_app()
    assert len(at.chat_input) == 0                    # nothing usable before the password

    at.text_input[0].input("wrong").run()
    assert "isn't right" in at.error[0].value

    at.text_input[0].input("open-sesame").run()
    assert len(at.chat_input) == 1


def test_question_limit_per_session(monkeypatch):
    monkeypatch.setenv("MAX_QUESTIONS_PER_SESSION", "1")
    at = run_app()
    at.chat_input[0].set_value("First question").run()
    at.chat_input[0].set_value("Second question").run()

    answers = [m for m in at.chat_message if m.name == "assistant"][1:]    # skip the welcome
    assert len(answers) == 1                          # the second question was never answered
    assert "limit of 1 questions" in at.info[0].value
    assert at.chat_input[0].disabled
