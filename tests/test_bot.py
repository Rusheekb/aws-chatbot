"""Tests for bot.py: cost, history, and the agent loop (with a fake Bedrock)."""
import pytest

import bot
from tests.conftest import text_events, tool_events


def user(text):
    return {"role": "user", "content": [{"text": text}]}


def assistant(text):
    return {"role": "assistant", "content": [{"text": text}]}


def tool_turn(question, answer):
    """A full turn that used one tool: question, tool request, tool result, answer."""
    return [
        user(question),
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t", "name": "x", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t", "content": [{"text": "r"}]}}]},
        assistant(answer),
    ]


# --- Cost --------------------------------------------------------------------

def test_calculate_cost_uses_per_million_prices(monkeypatch):
    monkeypatch.setattr(bot, "INPUT_PRICE_PER_M", 0.30)
    monkeypatch.setattr(bot, "OUTPUT_PRICE_PER_M", 2.50)
    assert bot.calculate_cost(1_000_000, 0) == pytest.approx(0.30)
    assert bot.calculate_cost(1000, 1000) == pytest.approx(0.0028)


# --- History -----------------------------------------------------------------

def test_turn_starts_ignores_tool_results():
    history = tool_turn("q1", "a1") + [user("q2"), assistant("a2")]
    assert bot.turn_starts(history) == [0, 4]
    assert bot.count_turns(history) == 2


def test_trim_keeps_whole_turns(monkeypatch):
    monkeypatch.setattr(bot, "MAX_TURNS", 2)
    history = [user("q1"), assistant("a1")] + tool_turn("q2", "a2") + [user("q3"), assistant("a3")]

    trimmed = bot.trim_history(history)

    assert trimmed[0] == user("q2")              # starts at a real question...
    assert bot.count_turns(trimmed) == 2         # ...and keeps the last 2 turns
    assert trimmed[-1] == assistant("a3")


def test_trim_leaves_short_history_alone(monkeypatch):
    monkeypatch.setattr(bot, "MAX_TURNS", 3)
    history = [user("q1"), assistant("a1")]
    assert bot.trim_history(history) == history


def test_history_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "HISTORY_FILE", tmp_path / "history.json")
    assert bot.load_history() == []              # no file yet

    history = tool_turn("q1", "a1")
    bot.save_history(history)
    assert bot.load_history() == history


def test_corrupted_history_starts_fresh(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    path.write_text("oops", encoding="utf-8")
    monkeypatch.setattr(bot, "HISTORY_FILE", path)
    assert bot.load_history() == []


# --- One model call (Reply) --------------------------------------------------

def test_reply_streams_text_and_records_usage(fake_bedrock):
    fake_bedrock(text_events("Hello there", tokens=(12, 3)))
    reply = bot.Reply([user("hi")])

    chunks = list(reply.stream())

    assert chunks == ["Hello there"]
    assert reply.text == "Hello there"
    assert reply.stop_reason == "end_turn"
    assert reply.usage["inputTokens"] == 12
    assert reply.message() == assistant("Hello there")


def test_reply_joins_tool_input_pieces(fake_bedrock):
    fake_bedrock(tool_events("search_documents", ['{"que', 'ry": "pri', 'ces"}']))
    reply = bot.Reply([user("how much?")])

    list(reply.stream())

    assert reply.stop_reason == "tool_use"
    assert reply.tool_calls == [
        {"toolUseId": "tool-1", "name": "search_documents", "input": {"query": "prices"}}
    ]
    assert reply.message()["content"] == [{"toolUse": reply.tool_calls[0]}]


# --- The agent loop (Turn) ---------------------------------------------------

def test_turn_runs_tool_then_answers(fake_bedrock, monkeypatch):
    monkeypatch.setattr(bot, "run_tool", lambda call: {
        "toolResult": {"toolUseId": call["toolUseId"], "content": [{"text": "ok"}], "status": "success"}
    })
    fake = fake_bedrock(
        tool_events("search_documents", ['{"query": "x"}'], tokens=(20, 8)),
        text_events("Final answer", tokens=(30, 6)),
    )
    messages = [user("question")]
    turn = bot.Turn(messages)

    text = "".join(turn.stream())

    assert text == "Final answer"
    assert turn.tools_used == ["search_documents"]
    assert (turn.input_tokens, turn.output_tokens) == (50, 14)    # both calls are counted
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert len(fake.calls) == 2


def test_turn_stops_after_max_tool_rounds(fake_bedrock, monkeypatch):
    monkeypatch.setattr(bot, "MAX_TOOL_ROUNDS", 2)
    monkeypatch.setattr(bot, "run_tool", lambda call: {
        "toolResult": {"toolUseId": call["toolUseId"], "content": [{"text": "ok"}], "status": "success"}
    })
    fake_bedrock(
        tool_events("search_documents", ['{"query": "a"}']),
        tool_events("search_documents", ['{"query": "b"}']),
    )
    messages = [user("question")]

    text = "".join(bot.Turn(messages).stream())

    assert "too many tool calls" in text
    # The history must still alternate and end on an assistant message, or the next request fails.
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]


def test_turn_records_sources_from_search_results(fake_bedrock, monkeypatch):
    monkeypatch.setattr(bot, "run_tool", lambda call: {"toolResult": {
        "toolUseId": call["toolUseId"], "status": "success",
        "content": [{"json": {"results": [
            {"source": "site/pricing.md", "text": "...", "score": 0.5},
            {"source": "help/buy-hours.md", "text": "...", "score": 0.4},
            {"source": "site/pricing.md", "text": "...", "score": 0.3},
        ]}}],
    }})
    fake_bedrock(tool_events("search_documents", ['{"query": "price"}']), text_events("$35 an hour"))
    turn = bot.Turn([user("How much?")])

    list(turn.stream())

    assert turn.sources == ["site/pricing.md", "help/buy-hours.md"]    # in order, no repeats
