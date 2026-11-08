import boto3
from dotenv import load_dotenv

load_dotenv()

# setup: runs once
bedrock_client = boto3.client('bedrock-runtime')
MODEL_ID='global.amazon.nova-2-lite-v1:0'

messages = []

print("Chatbot ready! Type 'quit' to exit, 'clear' to forget the conversation.\n")

# chat loop: runs once per question
while True:
    question = input("You: ").strip()

    if question.lower() == "quit":
        print("Goodbye!")
        break

    if question.lower() == "clear":
        messages.clear()
        print("Memory cleared.\n")
        continue
    
    if not question:
        continue

    messages.append({"role": "user", "content": [{"text": question}]})
    
    response = bedrock_client.converse(
        modelId=MODEL_ID,
        messages=messages,
    )

    assistant_message = response["output"]["message"]
    messages.append(assistant_message)

    answer = assistant_message["content"][0]["text"]
    print(f"Bot: {answer}")
    
    print(f"   [input tokens this turn: {response['usage']['inputTokens']}]\n")