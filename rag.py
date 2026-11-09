"""Search Learn2Lead's documents using embeddings (a simple RAG system).

Build the index:    python3 rag.py
Test a search:      python3 rag.py "how much does tutoring cost?"

The index rebuilds itself whenever the documents or chunk settings change.
"""
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIMENSIONS = 512
MAX_CHUNK_WORDS = int(os.getenv("MAX_CHUNK_WORDS", "100"))
TOP_K = int(os.getenv("TOP_K", "5"))
# Label each chunk with its document title and section, e.g. "[Apply to tutor › Then]".
CHUNK_HEADERS = os.getenv("CHUNK_HEADERS", "false").lower() == "true"
# Start a new chunk at every "## " heading, so a heading is never cut off from its section.
SECTION_CHUNKS = os.getenv("SECTION_CHUNKS", "false").lower() == "true"
# "semantic" = embeddings only; "hybrid" = embeddings + keyword search (BM25), merged.
SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid")

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_FILE = BASE_DIR / "doc_index.json"

# "adaptive" retries back off and slow down automatically when AWS says we're sending too fast.
client = boto3.client("bedrock-runtime", config=Config(retries={"max_attempts": 8, "mode": "adaptive"}))


def embed(text):
    """Turn text into a list of numbers (a vector) that captures its meaning."""
    response = client.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": EMBED_DIMENSIONS, "normalize": True}),
    )
    return json.loads(response["body"].read())["embedding"]


@lru_cache(maxsize=1000)
def embed_query(query):
    """Embed a search query, remembering recent ones so repeated questions cost nothing."""
    return tuple(embed(query))


def split_into_chunks(text):
    """Split text into chunks of whole paragraphs, each about MAX_CHUNK_WORDS words or less.

    With CHUNK_HEADERS on, each chunk starts with its document title and section, so a
    chunk like "1. Sign up..." still says which guide it came from.
    """
    chunks = []
    current = []
    word_count = 0
    title = section = chunk_section = None

    def finish_chunk():
        body = "\n\n".join(current)
        if CHUNK_HEADERS and title and not body.startswith("# "):
            header = f"{title} › {chunk_section}" if chunk_section else title
            body = f"[{header}]\n\n{body}"
        chunks.append(body)

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if paragraph.startswith("# "):
            title = paragraph[2:].strip()
        elif paragraph.startswith("## "):
            section = paragraph[3:].strip()
        words = len(paragraph.split())
        starts_section = SECTION_CHUNKS and paragraph.startswith("## ")
        if current and (starts_section or word_count + words > MAX_CHUNK_WORDS):
            finish_chunk()
            current = []
            word_count = 0
        if not current:
            chunk_section = section
        current.append(paragraph)
        word_count += words
    if current:
        finish_chunk()
    return chunks


def doc_files():
    """Return every .md and .txt file under docs/, including subfolders."""
    return sorted(DOCS_DIR.rglob("*.md")) + sorted(DOCS_DIR.rglob("*.txt"))


def fingerprint():
    """A hash of the settings and every document. It changes when anything the index depends on changes."""
    digest = hashlib.sha256()
    digest.update(json.dumps(
        [EMBED_MODEL_ID, EMBED_DIMENSIONS, MAX_CHUNK_WORDS, CHUNK_HEADERS, SECTION_CHUNKS]
    ).encode())
    for path in doc_files():
        digest.update(path.relative_to(DOCS_DIR).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_index():
    """Chunk and embed every document, save the index, and return its chunks."""
    chunks = []
    for path in doc_files():
        source = path.relative_to(DOCS_DIR).as_posix()
        for chunk in split_into_chunks(path.read_text(encoding="utf-8")):
            chunks.append({"source": source, "text": chunk, "embedding": embed(chunk)})
    index = {"fingerprint": fingerprint(), "chunks": chunks}
    INDEX_FILE.write_text(json.dumps(index), encoding="utf-8")
    return chunks


def ensure_index():
    """Return the index's chunks, rebuilding it first if it's missing or out of date."""
    if not doc_files():
        raise ValueError(f"No documents found in {DOCS_DIR}")
    if INDEX_FILE.exists():
        index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if isinstance(index, dict) and index.get("fingerprint") == fingerprint():
            return index["chunks"]
    return build_index()


def similarity(a, b):
    """How similar two normalized vectors are: about 1 = same meaning, about 0 = unrelated."""
    return sum(x * y for x, y in zip(a, b))


def tokenize(text):
    """Lowercase words and numbers, for keyword search."""
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_scores(query, texts, k1=1.5, b=0.75):
    """Score each text for the query's keywords with BM25, the classic search-engine formula.

    Words that are rare across all texts count more (idf), repeats help but with diminishing
    returns (k1), and long texts don't win just by being long (b).
    """
    documents = [tokenize(text) for text in texts]
    average_length = sum(len(d) for d in documents) / len(documents)
    document_frequency = Counter(word for d in documents for word in set(d))
    scores = []
    for words in documents:
        counts = Counter(words)
        score = 0.0
        for term in set(tokenize(query)):
            if term not in counts:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            tf = counts[term]
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(words) / average_length))
        scores.append(score)
    return scores


def reciprocal_rank_fusion(*rankings, k=60):
    """Merge several rankings (lists of positions, best first) into one.

    Each item earns 1 / (k + rank) from every ranking it appears in, so items that rank well
    in both semantic and keyword search rise to the top. k = 60 is the standard choice.
    """
    fused = Counter()
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] += 1 / (k + rank)
    return [item for item, _ in fused.most_common()]


def search(query):
    """Return the TOP_K chunks that best match the query."""
    chunks = ensure_index()
    query_embedding = embed_query(query)

    semantic = [similarity(query_embedding, item["embedding"]) for item in chunks]
    order = sorted(range(len(chunks)), key=lambda i: semantic[i], reverse=True)
    if SEARCH_MODE == "hybrid":
        keyword = bm25_scores(query, [item["text"] for item in chunks])
        keyword_order = sorted(range(len(chunks)), key=lambda i: keyword[i], reverse=True)
        order = reciprocal_rank_fusion(order, keyword_order)

    return [
        {"source": chunks[i]["source"], "text": chunks[i]["text"], "score": round(semantic[i], 3)}
        for i in order[:TOP_K]
    ]


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        for result in search(query):
            print(f"[{result['score']}] {result['source']}: {result['text'][:80]}...")
        return

    print(f"Indexing documents in {DOCS_DIR.name}/ ...")
    chunks = build_index()
    sources = {item["source"] for item in chunks}
    print(f"Indexed {len(chunks)} chunks from {len(sources)} files into {INDEX_FILE.name}")


if __name__ == "__main__":
    main()
