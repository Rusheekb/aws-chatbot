import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

load_dotenv()

# setup: runs once
bedrock_client = boto3.client('bedrock-runtime')
MODEL_ID = 'global.amazon.nova-2-lite-v1:0'
SYSTEM_PROMPT = """You are a friendly assistant helping a beginner learn to code.
Keep answers short (2-4 sentences) unless the user asks for more detail.
If you don't know something, say so instead of guessing."""
MAX_TOKENS = 300
TEMPERATURE = 0.7
MAX_TURNS = 3

# Prices in USD per 1 million tokens — check aws.amazon.com/bedrock/pricing
INPUT_PRICE_PER_M = 0.30
OUTPUT_PRICE_PER_M = 2.50


def calculate_cost(input_tokens, output_tokens):
    """Return the cost in USD for one request."""
    input_cost = input_tokens / 1_000_000 * INPUT_PRICE_PER_M
    output_cost = output_tokens / 1_000_000 * OUTPUT_PRICE_PER_M
    return input_cost + output_cost


def trim_history(history):
    """Keep only the most recent MAX_TURNS question/answer pairs."""
    max_messages = MAX_TURNS * 2
    return history[-max_messages:]


messages = []
total_input_tokens = 0
total_output_tokens = 0
total_cost = 0.0

print("Chatbot ready! Type 'quit' to exit, 'clear' to forget the conversation.\n")

# chat loop: runs once per question
while True:
    try:
        question = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        question = "quit"

    if question.lower() == "quit":
        print("\n--- Session summary ---")
        print(f"Input tokens:  {total_input_tokens:,}")
        print(f"Output tokens: {total_output_tokens:,}")
        print(f"Total cost:    ${total_cost:.6f}")
        print("Goodbye!")
        break

    if question.lower() == "clear":
        messages.clear()
        print("Memory cleared.\n")
        continue
    
    if not question:
        continue

    messages.append({"role": "user", "content": [{"text": question}]})

    try:
        response = bedrock_client.converse_stream(
            modelId=MODEL_ID,
            messages=messages,
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig={
                "maxTokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            },
        )

        print("Bot: ", end="", flush=True)
        answer = ""
        stop_reason = None
        usage = None

        for event in response["stream"]:
            if "contentBlockDelta" in event:
                chunk = event["contentBlockDelta"]["delta"].get("text", "")
                print(chunk, end="", flush=True)
                answer += chunk
            elif "messageStop" in event:
                stop_reason = event["messageStop"]["stopReason"]
            elif "metadata" in event:
                usage = event["metadata"]["usage"]

        print()
    except ClientError as error:
        print()
        code = error.response["Error"]["Code"]
        detail = error.response["Error"]["Message"]
        print(f"   [AWS error: {code} — {detail}]\n")
        messages.pop()
        continue
    except BotoCoreError as error:
        print()
        print(f"   [Connection problem: {error}]\n")
        messages.pop()
        continue

    messages.append({"role": "assistant", "content": [{"text": answer}]})
    messages = trim_history(messages)

    if stop_reason == "max_tokens":
        print("   [reply was cut off — hit the max token limit]")
    
    turn_cost = calculate_cost(usage["inputTokens"], usage["outputTokens"])

    total_input_tokens += usage["inputTokens"]
    total_output_tokens += usage["outputTokens"]
    total_cost += turn_cost

    print(f"   [{usage['inputTokens']} in / {usage['outputTokens']} out"
          f" | this turn: ${turn_cost:.6f} | session: ${total_cost:.6f}"
          f" | memory: {len(messages) // 2}/{MAX_TURNS} turns]\n")
