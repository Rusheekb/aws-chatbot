"""Tests for rag.py: chunking, similarity, and search (with a fake embedding model)."""
import math
import re
import shutil
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
    assert len(results) == min(rag.TOP_K, len(fixture_index))


def test_search_with_no_documents_explains_the_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "DOCS_DIR", tmp_path / "empty")
    with pytest.raises(ValueError, match="No documents found"):
        rag.search("anything")


def test_index_is_reused_until_documents_change(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    shutil.copytree(FIXTURE_DOCS, docs)
    monkeypatch.setattr(rag, "DOCS_DIR", docs)
    monkeypatch.setattr(rag, "INDEX_FILE", tmp_path / "index.json")
    calls = []
    monkeypatch.setattr(rag, "embed", lambda text: calls.append(text) or fake_embed(text))

    rag.ensure_index()
    first_build = len(calls)
    rag.ensure_index()
    assert len(calls) == first_build                 # nothing changed: no re-embedding

    (docs / "garden_faq.md").write_text("# FAQ\n\nNew answer about parking.", encoding="utf-8")
    chunks = rag.ensure_index()
    assert len(calls) > first_build                  # a document changed: rebuilt
    assert any("parking" in chunk["text"] for chunk in chunks)


def test_index_rebuilds_when_chunk_size_changes(tmp_path, monkeypatch, fixture_index):
    before = rag.fingerprint()
    monkeypatch.setattr(rag, "MAX_CHUNK_WORDS", rag.MAX_CHUNK_WORDS + 10)
    assert rag.fingerprint() != before


def test_chunk_headers_name_the_document_and_section(monkeypatch):
    monkeypatch.setattr(rag, "CHUNK_HEADERS", True)
    monkeypatch.setattr(rag, "MAX_CHUNK_WORDS", 6)
    text = "# Apply to tutor\n\nA short application.\n\n## Then\n\nWe email you to book an interview."

    chunks = rag.split_into_chunks(text)

    assert chunks[0].startswith("# Apply to tutor")          # the title chunk needs no header
    assert chunks[-1].startswith("[Apply to tutor › Then]")


def test_bm25_prefers_rare_matching_words():
    texts = ["the tutor helps the student", "apply to become a tutor", "the student joins class"]
    scores = rag.bm25_scores("become a tutor", texts)
    assert scores.index(max(scores)) == 1
    assert scores[2] == 0.0                                    # no matching words


def test_reciprocal_rank_fusion_rewards_agreement():
    semantic = [0, 1, 2]
    keyword = [1, 2, 0]
    assert rag.reciprocal_rank_fusion(semantic, keyword)[0] == 1   # ranked high by both


def test_hybrid_search_returns_top_k(fixture_index, monkeypatch):
    monkeypatch.setattr(rag, "SEARCH_MODE", "hybrid")
    results = rag.search("dogs on a leash")
    assert results[0]["source"] == "garden_faq.md"
    assert len(results) == min(rag.TOP_K, len(fixture_index))


def test_repeated_queries_are_embedded_once(monkeypatch):
    calls = []
    monkeypatch.setattr(rag, "embed", lambda text: calls.append(text) or fake_embed(text))
    rag.embed_query.cache_clear()

    rag.embed_query("how much does it cost?")
    rag.embed_query("how much does it cost?")

    assert calls == ["how much does it cost?"]
    rag.embed_query.cache_clear()
