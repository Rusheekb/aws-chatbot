"""Send the same questions to several Bedrock models and compare quality, speed, and cost.

Run with: python3 compare_models.py
"""
from datetime import datetime
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

load_dotenv()

# Prices in USD per 1M tokens — check aws.amazon.com/bedrock/pricing
MODELS = [
    {"name": "Nova Micro", "id": "us.amazon.nova-micro-v1:0", "input_price": 0.035, "output_price": 0.14},
    {"name": "Nova 2 Lite", "id": "global.amazon.nova-2-lite-v1:0", "input_price": 0.30, "output_price": 2.50},
    {"name": "Nova Pro", "id": "us.amazon.nova-pro-v1:0", "input_price": 0.80, "output_price": 3.20},
    {"name": "Llama 4 Scout", "id": "us.meta.llama4-scout-17b-instruct-v1:0", "input_price": 0.17, "output_price": 0.66},
]

# "expect" = text that must appear in a correct answer (questions without it are judged by you)
QUESTIONS = [
    {
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
                  "How much does the ball cost? Answer with just the dollar amount, like $1.23.",
        "expect": "0.05",
    },
    {
        "prompt": "What is 17 times 23? Answer with just the number.",
        "expect": "391",
    },
    {
        "prompt": "Write a Python function named is_palindrome(s) that returns True if a string "
                  "reads the same forwards and backwards. Code only, no explanation.",
        "expect": "def is_palindrome",
    },
    {
        "prompt": "List exactly three fruits, lowercase, separated by commas, with nothing else.",
    },
    {
        "prompt": "Explain what an API is to a 10-year-old in two sentences.",
    },
]

MAX_TOKENS = 300
REPORT_FILE = Path(__file__).parent / "comparison_results.md"

client = boto3.client("bedrock-runtime")


def ask(model, prompt):
    """Ask one model one question. Return a dict of results, or {"error": ...}."""
    try:
        response = client.converse(
            modelId=model["id"],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": 0},
        )
    except (ClientError, BotoCoreError) as error:
        return {"error": str(error)}

    usage = response["usage"]
    cost = (usage["inputTokens"] * model["input_price"]
            + usage["outputTokens"] * model["output_price"]) / 1_000_000
    return {
        "answer": response["output"]["message"]["content"][0]["text"].strip(),
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
        "latency_ms": response["metrics"]["latencyMs"],
        "cost": cost,
    }


def passed(question, result):
    """Return True/False if the question has an expected answer, or None if not."""
    if "expect" not in question:
        return None
    if "error" in result:
        return False
    return question["expect"] in result["answer"]


def summarize(results):
    """Return totals for one model's list of results."""
    answered = [r for r in results if "error" not in r]
    checks = [passed(q, r) for q, r in zip(QUESTIONS, results)]
    graded = [c for c in checks if c is not None]
    return {
        "answered": len(answered),
        "passed": sum(graded),
        "graded": len(graded),
        "avg_latency": sum(r["latency_ms"] for r in answered) / len(answered) if answered else 0,
        "output_tokens": sum(r["output_tokens"] for r in answered),
        "cost": sum(r["cost"] for r in answered),
    }


def print_summary(all_results):
    """Print a table with one row per model."""
    print(f"\n{'Model':<15}{'Passed':>8}{'Avg ms':>9}{'Out tok':>9}{'Cost':>12}")
    print("-" * 53)
    for model in MODELS:
        s = summarize(all_results[model["name"]])
        if s["answered"] == 0:
            print(f"{model['name']:<15}{'ERROR':>8}  {all_results[model['name']][0]['error'][:60]}")
            continue
        print(f"{model['name']:<15}{s['passed']:>5}/{s['graded']:<2}{s['avg_latency']:>9.0f}"
              f"{s['output_tokens']:>9}{s['cost']:>12.6f}")


def write_report(all_results):
    """Save a Markdown report with the summary and every answer side by side."""
    lines = [
        "# Model comparison",
        "",
        f"Run on {datetime.now():%Y-%m-%d %H:%M} · temperature 0 · max {MAX_TOKENS} tokens",
        "",
        "| Model | Passed | Avg latency (ms) | Output tokens | Cost (USD) |",
        "|---|---|---|---|---|",
    ]
    for model in MODELS:
        s = summarize(all_results[model["name"]])
        lines.append(f"| {model['name']} | {s['passed']}/{s['graded']} | {s['avg_latency']:.0f} "
                     f"| {s['output_tokens']} | ${s['cost']:.6f} |")

    for i, question in enumerate(QUESTIONS):
        lines += ["", f"## Q{i + 1}: {question['prompt']}", ""]
        for model in MODELS:
            result = all_results[model["name"]][i]
            check = passed(question, result)
            mark = {True: " ✅", False: " ❌", None: ""}[check]
            lines.append(f"**{model['name']}**{mark}")
            lines.append("")
            lines.append(result.get("answer", f"Error: {result.get('error')}"))
            lines.append("")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    all_results = {}
    for model in MODELS:
        print(f"Asking {model['name']}...")
        all_results[model["name"]] = [ask(model, q["prompt"]) for q in QUESTIONS]

    print_summary(all_results)
    write_report(all_results)
    print(f"\nFull answers saved to {REPORT_FILE.name}")


if __name__ == "__main__":
    main()
