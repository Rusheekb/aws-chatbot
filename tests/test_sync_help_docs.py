"""Tests for scripts/sync_help_docs.py: Help Center data -> Markdown."""
from scripts.sync_help_docs import article_to_markdown, clean

AUDIENCES = {"student": "Students", "parent": "Parents", "tutor": "Tutors"}
TOPICS = {"hours": "Hours and payments"}


def test_clean_converts_site_markup():
    text = "Press [[Join class]] or see [pricing](/pricing)."
    assert clean(text) == "Press **Join class** or see pricing."


def test_article_to_markdown():
    article = {
        "id": "buy-hours", "title": "Buy hours", "topic": "hours", "kind": "guide",
        "summary": "Hours never expire.", "for": ["student", "parent"], "payerOnly": True,
        "blocks": [
            {"steps": ["Go to [[Pricing]].", "Pick a pack."]},
            {"heading": "For families", "for": ["parent"], "text": "One shared balance."},
            {"heading": "Good to know", "points": ["No subscription."]},
        ],
    }

    markdown = article_to_markdown(article, AUDIENCES, TOPICS)

    assert markdown.startswith("# Buy hours\n\nFor: Students, Parents · Topic: Hours and payments")
    assert "Only the person who pays" in markdown
    assert "1. Go to **Pricing**.\n2. Pick a pack." in markdown
    assert "## For families (for Parents)\n\nOne shared balance." in markdown
    assert "- No subscription." in markdown
