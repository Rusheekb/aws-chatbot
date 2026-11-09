"""Check that the demo's AWS key can do what the app needs, and nothing more.

Run it with the demo user's key (not your own) before deploying:

    AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=us-east-2 \
        python3 deploy/check_permissions.py

Costs a fraction of a cent (two tiny model calls and one embedding).
"""
import json
import sys

import boto3
from botocore.exceptions import ClientError

CHAT_MODEL = "global.amazon.nova-2-lite-v1:0"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
NOT_ALLOWED_MODEL = "us.amazon.nova-pro-v1:0"     # any model the policy doesn't list

client = boto3.client("bedrock-runtime")
hello = [{"role": "user", "content": [{"text": "Say OK"}]}]


def check(description, call, should_work):
    try:
        call()
        worked = True
    except ClientError as error:
        if error.response["Error"]["Code"] != "AccessDeniedException":
            raise
        worked = False
    ok = worked == should_work
    print(f"{'OK  ' if ok else 'FAIL'}  {description}: {'allowed' if worked else 'denied'}")
    return ok


results = [
    check("Chat, streaming (the web app)", lambda: list(client.converse_stream(
        modelId=CHAT_MODEL, messages=hello, inferenceConfig={"maxTokens": 5})["stream"]), True),
    check("Chat, not streaming (the eval judge path)", lambda: client.converse(
        modelId=CHAT_MODEL, messages=hello, inferenceConfig={"maxTokens": 5}), True),
    check("Embeddings (document search)", lambda: client.invoke_model(
        modelId=EMBED_MODEL, body=json.dumps({"inputText": "hello", "dimensions": 512})), True),
    check("A model the demo doesn't use", lambda: client.converse(
        modelId=NOT_ALLOWED_MODEL, messages=hello, inferenceConfig={"maxTokens": 5}), False),
]
sys.exit(0 if all(results) else 1)
