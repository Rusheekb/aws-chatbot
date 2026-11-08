"""Core chatbot logic: settings, Bedrock calls, cost tracking, and history.

Shared by the terminal app (chatbot.py) and the web app (app.py).
"""
import json
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

from tools import TOOL_CONFIG, run_tool

load_dotenv()

# --- Settings: read from .env, with defaults if missing ---
MODEL_ID = os.getenv("MODEL_ID", "global.amazon.nova-2-lite-v1:0")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "3"))
INPUT_PRICE_PER_M = float(os.getenv("INPUT_PRICE_PER_M", "0.30"))
OUTPUT_PRICE_PER_M = float(os.getenv("OUTPUT_PRICE_PER_M", "2.50"))
MAX_TOOL_ROUNDS = 5

BASE_DIR = Path(__file__).parent
SYSTEM_PROMPT = (BASE_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()
HISTORY_FILE = BASE_DIR / "chat_history.json"

bedrock_client = boto3.client("bedrock-runtime")


def calculate_cost(input_tokens, output_tokens):
    """Return the cost in USD for one request."""
    input_cost = input_tokens / 1_000_000 * INPUT_PRICE_PER_M
    output_cost = output_tokens / 1_000_000 * OUTPUT_PRICE_PER_M
    return input_cost + output_cost


def turn_starts(history):
    """Return the positions of real user questions (not tool results)."""
    return [
        i for i, message in enumerate(history)
        if message["role"] == "user" and "text" in message["content"][0]
    ]


def count_turns(history):
    """Return how many question/answer turns are in the history."""
    return len(turn_starts(history))


def trim_history(history):
    """Keep only the most recent MAX_TURNS turns (tool messages stay with their turn)."""
    starts = turn_starts(history)
    if len(starts) <= MAX_TURNS:
        return history
    return history[starts[-MAX_TURNS]:]


def load_history():
    """Load saved messages, or return an empty list if there's nothing valid."""
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("   [Warning: history file is unreadable — starting fresh]")
        return []


def save_history(history):
    """Write messages to the history file."""
    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


class Reply:
    """One streamed reply from Bedrock (one model call).

    Iterate over reply.stream() to get text chunks as they arrive.
    Afterwards, text, tool_calls, stop_reason and usage are filled in.
    """

    def __init__(self, messages):
        self.messages = messages
        self.text = ""
        self.tool_calls = []
        self.stop_reason = None
        self.usage = None

    def stream(self):
        """Call Bedrock and yield text chunks as they arrive."""
        response = bedrock_client.converse_stream(
            modelId=MODEL_ID,
            messages=self.messages,
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig={
                "maxTokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            },
            toolConfig=TOOL_CONFIG,
        )
        for event in response["stream"]:
            if "contentBlockStart" in event:
                start = event["contentBlockStart"]["start"]
                if "toolUse" in start:
                    self.tool_calls.append({
                        "toolUseId": start["toolUse"]["toolUseId"],
                        "name": start["toolUse"]["name"],
                        "input": "",
                    })
            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    self.text += delta["text"]
                    yield delta["text"]
                elif "toolUse" in delta:
                    self.tool_calls[-1]["input"] += delta["toolUse"]["input"]
            elif "messageStop" in event:
                self.stop_reason = event["messageStop"]["stopReason"]
            elif "metadata" in event:
                self.usage = event["metadata"]["usage"]

        for call in self.tool_calls:
            call["input"] = json.loads(call["input"] or "{}")

    def message(self):
        """Return this reply as an assistant message for the history."""
        content = []
        if self.text:
            content.append({"text": self.text})
        for call in self.tool_calls:
            content.append({"toolUse": call})
        return {"role": "assistant", "content": content}


class Turn:
    """Everything that happens between a user question and the final answer.

    Calls the model, runs any tools it asks for, and calls it again,
    until it gives a final answer. Appends every step to `messages`.
    """

    def __init__(self, messages):
        self.messages = messages
        self.input_tokens = 0
        self.output_tokens = 0
        self.stop_reason = None
        self.tools_used = []
        self.sources = []

    def stream(self):
        """Yield text chunks from every model call in this turn."""
        for _ in range(MAX_TOOL_ROUNDS):
            reply = Reply(self.messages)
            yield from reply.stream()

            self.input_tokens += reply.usage["inputTokens"]
            self.output_tokens += reply.usage["outputTokens"]
            self.stop_reason = reply.stop_reason
            self.messages.append(reply.message())

            if reply.stop_reason != "tool_use":
                return

            results = []
            for call in reply.tool_calls:
                self.tools_used.append(call["name"])
                result = run_tool(call)
                self._collect_sources(result)
                results.append(result)
            self.messages.append({"role": "user", "content": results})

        notice = "[Stopped: too many tool calls in one turn]"
        self.messages.append({"role": "assistant", "content": [{"text": notice}]})
        yield notice

    def _collect_sources(self, tool_result):
        """Remember which documents a search returned, in order, without repeats."""
        for block in tool_result["toolResult"]["content"]:
            for item in block.get("json", {}).get("results", []):
                if item["source"] not in self.sources:
                    self.sources.append(item["source"])

    def cost(self):
        """Return the total cost in USD of every model call in this turn."""
        return calculate_cost(self.input_tokens, self.output_tokens)
