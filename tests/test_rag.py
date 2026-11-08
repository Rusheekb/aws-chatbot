"""Tests for rag.py: chunking, similarity, and search (with a fake embedding model)."""
import math
import re
from pathlib import Path

import pytest

import rag

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "docs"


def fake_embed(text, dimensions=64):
    """A stand-in for Titan: hash each word into a vector, then normalize.

    Texts that share words get similar vectors, which is enough to test search.
    """
    vector = [0.0] * dimensions
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vector[hash(word) % dimensions] += 1.0
    length = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / length for x in vector]


@pytest.fixture
def fixture_index(tmp_path, monkeypatch):
    """Point rag at the fixture docs, a temporary index file, and the fake embedder."""
    monkeypatch.setattr(rag, "DOCS_DIR", FIXTURE_DOCS)
    monkeypatch.setattr(rag, "INDEX_FILE", tmp_path / "index.json")
    monkeypatch.setattr(rag, "embed", fake_embed)
    return rag.build_index()


def test_chunks_keep_whole_paragraphs(monkeypatch):
    monkeypatch.setattr(rag, "MAX_CHUNK_WORDS", 5)
    text = "one two three\n\nfour five\n\nsix seven eight nine"

    chunks = rag.split_into_chunks(text)

    assert chunks == ["one two three\n\nfour five", "six seven eight nine"]


def test_a_long_paragraph_is_never_split(monkeypatch):
    monkeypatch.setattr(rag, "MAX_CHUNK_WORDS", 2)
    assert rag.split_into_chunks("a b c d e") == ["a b c d e"]


def test_blank_paragraphs_are_skipped():
    assert rag.split_into_chunks("\n\n  \n\nhello\n\n\n\n") == ["hello"]


def test_similarity_of_normalized_vectors():
    assert rag.similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert rag.similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_index_covers_every_fixture_file(fixture_index):
    assert {item["source"] for item in fixture_index} == {"garden_faq.md", "garden_handbook.md"}


def test_search_finds_the_relevant_document(fixture_index):
    results = rag.search("can I bring my dog")
    assert results[0]["source"] == "garden_faq.md"
    assert "Dogs are allowed" in results[0]["text"]
    assert len(results) == rag.TOP_K


def test_search_without_index_explains_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "INDEX_FILE", tmp_path / "missing.json")
    with pytest.raises(ValueError, match="python3 rag.py"):
        rag.search("anything")
