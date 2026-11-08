"""Search your own documents using embeddings (a simple RAG system).

Build the index:    python3 rag.py
Test a search:      python3 rag.py "how much does a plot cost?"
"""
import json
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIMENSIONS = 512
MAX_CHUNK_WORDS = 60
TOP_K = 3

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_FILE = BASE_DIR / "doc_index.json"

client = boto3.client("bedrock-runtime")


def embed(text):
    """Turn text into a list of numbers (a vector) that captures its meaning."""
    response = client.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": EMBED_DIMENSIONS, "normalize": True}),
    )
    return json.loads(response["body"].read())["embedding"]


def split_into_chunks(text):
    """Split text into chunks of whole paragraphs, each about MAX_CHUNK_WORDS words or less."""
    chunks = []
    current = []
    word_count = 0
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        words = len(paragraph.split())
        if current and word_count + words > MAX_CHUNK_WORDS:
            chunks.append("\n\n".join(current))
            current = []
            word_count = 0
        current.append(paragraph)
        word_count += words
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def doc_files():
    """Return every .md and .txt file under docs/, including subfolders."""
    return sorted(DOCS_DIR.rglob("*.md")) + sorted(DOCS_DIR.rglob("*.txt"))


def build_index():
    """Chunk and embed every document, and save the index."""
    index = []
    for path in doc_files():
        source = path.relative_to(DOCS_DIR).as_posix()
        for chunk in split_into_chunks(path.read_text(encoding="utf-8")):
            index.append({"source": source, "text": chunk, "embedding": embed(chunk)})
    INDEX_FILE.write_text(json.dumps(index), encoding="utf-8")
    return index


def similarity(a, b):
    """How similar two normalized vectors are: about 1 = same meaning, about 0 = unrelated."""
    return sum(x * y for x, y in zip(a, b))


def search(query):
    """Return the TOP_K chunks whose meaning is closest to the query."""
    if not INDEX_FILE.exists():
        raise ValueError("No document index yet. Run: python3 rag.py")
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    query_embedding = embed(query)

    scored = [(similarity(query_embedding, item["embedding"]), item) for item in index]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [
        {"source": item["source"], "text": item["text"], "score": round(score, 3)}
        for score, item in scored[:TOP_K]
    ]


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        for result in search(query):
            print(f"[{result['score']}] {result['source']}: {result['text'][:80]}...")
        return

    print(f"Indexing documents in {DOCS_DIR.name}/ ...")
    index = build_index()
    sources = {item["source"] for item in index}
    print(f"Indexed {len(index)} chunks from {len(sources)} files into {INDEX_FILE.name}")


if __name__ == "__main__":
    main()
