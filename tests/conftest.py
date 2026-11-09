"""Shared test setup. Tests never call AWS: every Bedrock client is replaced by a fake."""
import os

# boto3 needs a region to create a client, even a client that is never used.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-2")

import pytest

import bot
import rag


class NoNetworkClient:
    """Stands in for a real Bedrock client and fails loudly if a test forgets to mock it."""

    def __getattr__(self, name):
        raise AssertionError(f"Test tried to call real Bedrock: {name}()")


class FakeBedrock:
    """Returns pre-written streams of events, one per converse_stream() call."""

    def __init__(self, *streams):
        self.streams = list(streams)
        self.calls = []

    def converse_stream(self, **kwargs):
        self.calls.append(kwargs)
        return {"stream": self.streams.pop(0)}


def text_events(text, stop_reason="end_turn", tokens=(10, 5)):
    """The events Bedrock streams for a plain text reply."""
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": text}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": stop_reason}},
        {"metadata": {"usage": {"inputTokens": tokens[0], "outputTokens": tokens[1],
                                "totalTokens": sum(tokens)}}},
    ]


def tool_events(name, input_pieces, tool_use_id="tool-1", tokens=(20, 8)):
    """The events Bedrock streams when the model asks for a tool (input arrives in pieces)."""
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {"toolUse": {"toolUseId": tool_use_id, "name": name}},
                               "contentBlockIndex": 0}},
    ]
    for piece in input_pieces:
        events.append({"contentBlockDelta": {"delta": {"toolUse": {"input": piece}},
                                             "contentBlockIndex": 0}})
    events += [
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": tokens[0], "outputTokens": tokens[1],
                                "totalTokens": sum(tokens)}}},
    ]
    return events


@pytest.fixture(autouse=True)
def no_real_aws(monkeypatch):
    """Make sure no test can reach AWS by accident."""
    monkeypatch.setattr(bot, "bedrock_client", NoNetworkClient())
    monkeypatch.setattr(rag, "client", NoNetworkClient())
    rag.embed_query.cache_clear()       # cached embeddings must not leak between tests


@pytest.fixture
def fake_bedrock(monkeypatch):
    """Install a FakeBedrock. Call it with the streams the model should 'send'."""
    def install(*streams):
        fake = FakeBedrock(*streams)
        monkeypatch.setattr(bot, "bedrock_client", fake)
        return fake
    return install
