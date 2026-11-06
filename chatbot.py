import boto3
from dotenv import load_dotenv

load_dotenv()

# create bedrock client
bedrock_client = boto3.client('bedrock-runtime')

question = "What is 2 + 2? Reply with just the number."

response = bedrock_client.converse(

    modelId='global.amazon.nova-2-lite-v1:0',
    messages=[
        {
            "role": "user",
            "content": [{"text": question}]
        }
    ]
)

print(response)