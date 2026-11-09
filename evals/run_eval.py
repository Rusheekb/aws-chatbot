"""Evaluate the Learn2Lead assistant on a fixed set of questions.

Retrieval only (cheap, no chat model):  python3 -m evals.run_eval --retrieval-only
Full evaluation:                        python3 -m evals.run_eval --label baseline

Settings come from .env / environment variables, so experiments need no code changes:
    MODEL_ID=us.amazon.nova-pro-v1:0 INPUT_PRICE_PER_M=0.80 OUTPUT_PRICE_PER_M=3.20 \
        python3 -m evals.run_eval --label nova-pro
"""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

import bot
import rag

QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

# A different, stronger model grades the answers ("LLM as judge").
JUDGE_MODEL_ID = "us.amazon.nova-pro-v1:0"
JUDGE_INPUT_PRICE_PER_M = 0.80
JUDGE_OUTPUT_PRICE_PER_M = 3.20

BEHAVIOR_RULES = {
    "answer": "PASS if the answer agrees with the reference answer and includes its key facts "
              "(extra correct detail is fine). FAIL if it contradicts the reference, misses the key "
              "fact, or invents details such as prices, policies or steps.",
    "decline": "PASS if it politely declines because the question isn't about Learn2Lead (mentioning "
               "that Learn2Lead's tutors can help is fine). FAIL if it does the off-topic task itself, "
               "even partly.",
    "unknown": "PASS if it says it doesn't have that information and points the user to Learn2Lead's "
               "contact options. FAIL if it invents a policy or an answer.",
}

JUDGE_PROMPT = """You are grading an answer from the Learn2Lead website assistant.

Question: {question}
Reference answer: {expected}
Assistant's answer: {answer}

Rule: {rule}

Reply with only JSON, like {{"pass": true, "reason": "one short sentence"}}."""


def load_questions(ids=None):
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return questions


def first_match_rank(found, expected):
    """Position (1 = first) of the first expected source in the results, or None."""
    for rank, source in enumerate(found, start=1):
        if source in expected:
            return rank
    return None


def normalize(text):
    """Lowercase, drop Markdown bold, and straighten apostrophes, for phrase matching."""
    return text.lower().replace("**", "").replace("’", "'")


def evaluate_retrieval(questions):
    """Measure document search for questions that have expected sources.

    - hit rate: an expected document is in the top results
    - MRR (mean reciprocal rank): 1 if it's first, 1/2 if second, ... averaged
    - evidence recall: a phrase that answers the question is in the retrieved text
      (stricter: finding the right document but the wrong chunk doesn't count)
    """
    rows = []
    for q in questions:
        if not q["sources"]:
            continue
        results = rag.search(q["question"])
        found = [r["source"] for r in results]
        retrieved_text = normalize(" ".join(r["text"] for r in results))
        rows.append({
            "id": q["id"],
            "rank": first_match_rank(found, q["sources"]),
            "evidence_found": any(normalize(phrase) in retrieved_text for phrase in q["evidence"]),
            "found": found,
        })
    if not rows:
        return None                     # e.g. only off-topic questions were selected
    hits = [row for row in rows if row["rank"]]
    return {
        "hit_rate": len(hits) / len(rows),
        "mrr": sum(1 / row["rank"] for row in hits) / len(rows),
        "evidence_recall": sum(row["evidence_found"] for row in rows) / len(rows),
        "misses": [row for row in rows if not row["rank"] or not row["evidence_found"]],
    }


def ask_assistant(question):
    """Ask the full assistant one question with an empty history."""
    turn = bot.Turn([{"role": "user", "content": [{"text": question}]}])
    started = time.perf_counter()
    answer = "".join(turn.stream())
    return {
        "answer": answer,
        "searched": "search_documents" in turn.tools_used,
        "sources": turn.sources,
        "input_tokens": turn.input_tokens,
        "output_tokens": turn.output_tokens,
        "cost": turn.cost(),
        "seconds": round(time.perf_counter() - started, 2),
    }


def parse_verdict(text):
    """Read the judge's JSON reply. Anything unreadable counts as a fail."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        verdict = json.loads(match.group(0))
        return {"pass": bool(verdict["pass"]), "reason": str(verdict.get("reason", ""))}
    except (AttributeError, json.JSONDecodeError, KeyError):
        return {"pass": False, "reason": f"Judge reply could not be read: {text[:120]}"}


def judge(question, answer):
    """Have the judge model grade one answer. Returns (verdict, judge cost in USD)."""
    prompt = JUDGE_PROMPT.format(
        question=question["question"], expected=question["expected"],
        answer=answer, rule=BEHAVIOR_RULES[question["behavior"]],
    )
    response = bot.bedrock_client.converse(
        modelId=JUDGE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 200, "temperature": 0},
    )
    usage = response["usage"]
    cost = (usage["inputTokens"] * JUDGE_INPUT_PRICE_PER_M
            + usage["outputTokens"] * JUDGE_OUTPUT_PRICE_PER_M) / 1_000_000
    return parse_verdict(response["output"]["message"]["content"][0]["text"]), cost


def current_settings():
    return {
        "model_id": bot.MODEL_ID,
        "max_tokens": bot.MAX_TOKENS,
        "temperature": bot.TEMPERATURE,
        "max_chunk_words": rag.MAX_CHUNK_WORDS,
        "top_k": rag.TOP_K,
        "chunk_headers": rag.CHUNK_HEADERS,
        "section_chunks": rag.SECTION_CHUNKS,
        "search_mode": rag.SEARCH_MODE,
        "system_prompt": bot.SYSTEM_PROMPT,
    }


THINKING = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)


def evaluate_answers(questions):
    """Run every question through the assistant and the judge.

    The judge grades the answer's content with any <thinking> blocks removed, and leaked
    tags are recorded separately. A question passes only if the content is right AND
    nothing leaked, but keeping both lets us see how good a model would be once fixed.
    """
    rows = []
    for q in questions:
        try:
            result = ask_assistant(q["question"])
            content = THINKING.sub("", result["answer"]).strip()
            verdict, judge_cost = judge(q, content)
        except (ClientError, BotoCoreError) as error:
            result = {"answer": "", "error": str(error), "cost": 0.0, "seconds": 0.0,
                      "input_tokens": 0, "output_tokens": 0, "searched": False, "sources": []}
            verdict, judge_cost = {"pass": False, "reason": "Error while answering"}, 0.0
        leaks = "<thinking>" in result["answer"]
        row = {**q, **result, "content_pass": verdict["pass"], "leaks_thinking": leaks,
               "pass": verdict["pass"] and not leaks, "reason": verdict["reason"], "judge_cost": judge_cost}
        rows.append(row)
        note = " (leaks <thinking>)" if leaks else ""
        print(f"  {'PASS' if row['pass'] else 'FAIL'}  {q['id']:<18} {verdict['reason'][:60]}{note}")
    return rows


def summarize(rows):
    """Accuracy overall and per category, plus average cost, speed and tokens.

    Questions that hit an AWS error (e.g. throttling) are counted separately, so an
    infrastructure problem doesn't look like a drop in answer quality.
    """
    answered = [row for row in rows if not row.get("error")]
    if not answered:
        return {"accuracy": 0.0, "content_accuracy": 0.0, "thinking_leaks": 0, "errors": len(rows),
                "by_category": {}, "avg_cost": 0.0, "avg_seconds": 0.0, "avg_input_tokens": 0.0,
                "search_rate": 0.0, "judge_cost": 0.0}
    categories = sorted({row["category"] for row in answered})
    return {
        "accuracy": sum(row["pass"] for row in answered) / len(answered),
        "content_accuracy": sum(row.get("content_pass", row["pass"]) for row in answered) / len(answered),
        "thinking_leaks": sum(row.get("leaks_thinking", False) for row in answered),
        "errors": len(rows) - len(answered),
        "by_category": {
            c: sum(r["pass"] for r in answered if r["category"] == c)
            / sum(r["category"] == c for r in answered)
            for c in categories
        },
        "avg_cost": sum(row["cost"] for row in answered) / len(answered),
        "avg_seconds": sum(row["seconds"] for row in answered) / len(answered),
        "avg_input_tokens": sum(row["input_tokens"] for row in answered) / len(answered),
        "search_rate": sum(row["searched"] for row in answered) / len(answered),
        "judge_cost": sum(row["judge_cost"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-only", action="store_true", help="only measure document search")
    parser.add_argument("--label", default="run", help="name for the results file")
    parser.add_argument("--ids", nargs="*", help="only run these question ids")
    args = parser.parse_args()

    rag.ensure_index()
    questions = load_questions(args.ids)
    retrieval = evaluate_retrieval(questions)
    if retrieval:
        print(f"Retrieval: hit rate {retrieval['hit_rate']:.0%}, MRR {retrieval['mrr']:.2f}, "
              f"evidence recall {retrieval['evidence_recall']:.0%} (chunks of {rag.MAX_CHUNK_WORDS} "
              f"words, top {rag.TOP_K}, headers {rag.CHUNK_HEADERS}, sections {rag.SECTION_CHUNKS}, "
              f"{rag.SEARCH_MODE} search)")
        for miss in retrieval["misses"]:
            problem = "wrong documents" if not miss["rank"] else "right document, answer not in chunk"
            print(f"  MISS {miss['id']} ({problem}): got {miss['found']}")
    if args.retrieval_only:
        return

    print(f"\nAnswers ({bot.MODEL_ID}):")
    rows = evaluate_answers(questions)
    summary = summarize(rows)

    answered = len(rows) - summary["errors"]
    print(f"\nAccuracy: {summary['accuracy']:.0%} ({sum(r['pass'] for r in rows)}/{answered})")
    if summary["thinking_leaks"]:
        print(f"Content accuracy ignoring leaked <thinking> tags: {summary['content_accuracy']:.0%} "
              f"({summary['thinking_leaks']} answers leaked tags)")
    if summary["errors"]:
        print(f"WARNING: {summary['errors']} questions hit AWS errors and are excluded. Re-run later.")
    for category, accuracy in summary["by_category"].items():
        print(f"  {category:<15} {accuracy:.0%}")
    print(f"Avg cost per question: ${summary['avg_cost']:.5f} · avg {summary['avg_seconds']:.1f}s · "
          f"avg {summary['avg_input_tokens']:.0f} input tokens · searched {summary['search_rate']:.0%}")
    print(f"Judge cost: ${summary['judge_cost']:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M}-{args.label}.json"
    record = {"label": args.label, "settings": current_settings(), "retrieval": retrieval,
              "summary": summary, "rows": rows}
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


if __name__ == "__main__":
    main()
