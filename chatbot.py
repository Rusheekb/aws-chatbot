"""Terminal interface for the Bedrock chatbot."""
from botocore.exceptions import BotoCoreError, ClientError

from bot import (
    MAX_TOKENS,
    MAX_TURNS,
    MODEL_ID,
    Reply,
    load_history,
    save_history,
    trim_history,
)


def main():
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    print(f"Chatbot ready! Model: {MODEL_ID} | max tokens: {MAX_TOKENS} | memory: {MAX_TURNS} turns")
    print("Type 'quit' to exit, 'clear' to forget the conversation.\n")
    messages = trim_history(load_history())

    if messages:
        print(f"Restored {len(messages) // 2} turns from your last session.\n")

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
            save_history(messages)
            print("Memory cleared.\n")
            continue

        if not question:
            continue

        messages.append({"role": "user", "content": [{"text": question}]})
        reply = Reply(messages)

        try:
            print("Bot: ", end="", flush=True)
            for chunk in reply.stream():
                print(chunk, end="", flush=True)
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

        messages.append({"role": "assistant", "content": [{"text": reply.text}]})
        messages = trim_history(messages)
        save_history(messages)

        if reply.stop_reason == "max_tokens":
            print("   [reply was cut off — hit the max token limit]")

        turn_cost = reply.cost()
        total_input_tokens += reply.usage["inputTokens"]
        total_output_tokens += reply.usage["outputTokens"]
        total_cost += turn_cost

        print(f"   [{reply.usage['inputTokens']} in / {reply.usage['outputTokens']} out"
              f" | this turn: ${turn_cost:.6f} | session: ${total_cost:.6f}"
              f" | memory: {len(messages) // 2}/{MAX_TURNS} turns]\n")


if __name__ == "__main__":
    main()
