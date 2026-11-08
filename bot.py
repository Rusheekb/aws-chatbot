"""Core chatbot logic: settings, Bedrock calls, cost tracking, and history.

Shared by the terminal app (chatbot.py) and the web app (app.py).
"""
import json
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

# --- Settings: read from .env, with defaults if missing ---
MODEL_ID = os.getenv("MODEL_ID", "global.amazon.nova-2-lite-v1:0")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "3"))
INPUT_PRICE_PER_M = float(os.getenv("INPUT_PRICE_PER_M", "0.30"))
OUTPUT_PRICE_PER_M = float(os.getenv("OUTPUT_PRICE_PER_M", "2.50"))

BASE_DIR = Path(__file__).parent
SYSTEM_PROMPT = (BASE_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()
HISTORY_FILE = BASE_DIR / "chat_history.json"

bedrock_client = boto3.client("bedrock-runtime")


def calculate_cost(input_tokens, output_tokens):
    """Return the cost in USD for one request."""
    input_cost = input_tokens / 1_000_000 * INPUT_PRICE_PER_M
    output_cost = output_tokens / 1_000_000 * OUTPUT_PRICE_PER_M
    return input_cost + output_cost


def trim_history(history):
    """Keep only the most recent MAX_TURNS question/answer pairs."""
    max_messages = MAX_TURNS * 2
    return history[-max_messages:]


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
    """One streamed reply from Bedrock.

    Iterate over reply.stream() to get text chunks as they arrive.
    Afterwards, reply.text, reply.stop_reason and reply.usage are filled in.
    """

    def __init__(self, messages):
        self.messages = messages
        self.text = ""
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
        )
        for event in response["stream"]:
            if "contentBlockDelta" in event:
                chunk = event["contentBlockDelta"]["delta"].get("text", "")
                self.text += chunk
                yield chunk
            elif "messageStop" in event:
                self.stop_reason = event["messageStop"]["stopReason"]
            elif "metadata" in event:
                self.usage = event["metadata"]["usage"]

    def cost(self):
        """Return this reply's cost in USD. Call after streaming finishes."""
        return calculate_cost(self.usage["inputTokens"], self.usage["outputTokens"])
