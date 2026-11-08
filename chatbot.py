import boto3
from dotenv import load_dotenv

load_dotenv()

# setup: runs once
bedrock_client = boto3.client('bedrock-runtime')
MODEL_ID='global.amazon.nova-2-lite-v1:0'

print("Chatbot ready! Type 'quit to exit.\n")

# chat loop: runs once per question
while True:
    question = input("You: ").strip()

    if question.lower() == "quit":
        print("Goodbye!")
        break

    if not question:
        continue

    response = bedrock_client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [{"text": question}]
            }
        ]
    )
    answer = response["output"]["message"]["content"][0]["text"]
    print(f"Bot: {answer}\n")