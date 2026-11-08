"""Copy Learn2Lead's Help Center articles into docs/help/ as Markdown.

The articles live in the Learn2Lead repo as TypeScript data
(src/services/help/articles.ts). This script compiles that file with the
esbuild already installed in Learn2Lead, loads it with Node, and writes one
Markdown file per article, so the chatbot answers from the same text the
website shows.

Usage:  python3 scripts/sync_help_docs.py ../Learn2Lead
Then rebuild the search index:  python3 rag.py
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "docs" / "help"
PAYER_ONLY_NOTE = (
    "Only the person who pays for the account can do this. "
    "If a parent pays for a student, the parent manages it."
)

EXPORT_JS = """
import(process.argv[1]).then((m) => {
  process.stdout.write(JSON.stringify({
    articles: m.ARTICLES, audiences: m.AUDIENCE_LABEL, topics: m.TOPICS,
  }));
});
"""


def export_articles(learn2lead_dir):
    """Compile articles.ts to JavaScript and return its data as a dict."""
    source = learn2lead_dir / "src" / "services" / "help" / "articles.ts"
    esbuild = learn2lead_dir / "node_modules" / ".bin" / "esbuild"
    with tempfile.TemporaryDirectory() as tmp:
        compiled = Path(tmp) / "articles.mjs"
        subprocess.run(
            [str(esbuild), str(source), "--format=esm", "--platform=node",
             f"--outfile={compiled}", "--log-level=warning"],
            check=True,
        )
        output = subprocess.run(
            ["node", "--input-type=module", "-e", EXPORT_JS, str(compiled)],
            check=True, capture_output=True, text=True,
        )
    return json.loads(output.stdout)


def clean(text):
    """Turn the Help Center's markup into plain Markdown."""
    text = re.sub(r"\[\[(.+?)\]\]", r"**\1**", text)        # [[Button]] -> **Button**
    text = re.sub(r"\[([^\]]+)\]\(/[^)]*\)", r"\1", text)    # [words](/path) -> words
    return text


def article_to_markdown(article, audiences, topics):
    """Render one Help Center article as a Markdown document."""
    readers = ", ".join(audiences[a] for a in article["for"])
    lines = [
        f"# {article['title']}",
        "",
        f"For: {readers} · Topic: {topics[article['topic']]}",
        "",
        clean(article["summary"]),
    ]
    if article.get("payerOnly"):
        lines += ["", PAYER_ONLY_NOTE]

    for block in article["blocks"]:
        lines.append("")
        if block.get("heading"):
            heading = block["heading"]
            if block.get("for"):
                heading += f" (for {', '.join(audiences[a] for a in block['for'])})"
            lines += [f"## {heading}", ""]
        if block.get("text"):
            lines.append(clean(block["text"]))
        if block.get("steps"):
            lines += [f"{i}. {clean(step)}" for i, step in enumerate(block["steps"], start=1)]
        if block.get("points"):
            lines += [f"- {clean(point)}" for point in block["points"]]

    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 scripts/sync_help_docs.py <path to the Learn2Lead repo>")
    data = export_articles(Path(sys.argv[1]).resolve())
    topics = {t["key"]: t["label"] for t in data["topics"]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.md"):
        old.unlink()
    for article in data["articles"]:
        markdown = article_to_markdown(article, data["audiences"], topics)
        (OUT_DIR / f"{article['id']}.md").write_text(markdown, encoding="utf-8")

    print(f"Wrote {len(data['articles'])} articles to {OUT_DIR}")


if __name__ == "__main__":
    main()
