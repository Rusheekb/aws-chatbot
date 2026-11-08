import boto3
from dotenv import load_dotenv

load_dotenv()

# setup: runs once
bedrock_client = boto3.client('bedrock-runtime')
MODEL_ID = 'global.amazon.nova-2-lite-v1:0'

messages = []   # NEW: the conversation history, created once

print("Chatbot ready! Type 'quit' to exit, 'clear' to forget the conversation.\n")

# chat loop: runs once per question
while True:
    question = input("You: ").strip()

    if question.lower() == "quit":
        print("Goodbye!")
        break

    if question.lower() == "clear":          # NEW: reset memory
        messages.clear()
        print("Memory cleared.\n")
        continue

    if not question:
        continue

    # NEW: add your question to the history
    messages.append({"role": "user", "content": [{"text": question}]})

    response = bedrock_client.converse(
        modelId=MODEL_ID,
        messages=messages,                   # CHANGED: send the WHOLE history
    )

    # NEW: save the bot's reply into the history
    assistant_message = response["output"]["message"]
    messages.append(assistant_message)

    answer = assistant_message["content"][0]["text"]
    print(f"Bot: {answer}")

    # NEW (optional): watch the cost grow
    print(f"   [input tokens this turn: {response['usage']['inputTokens']}]\n")
