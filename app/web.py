import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, render_template, abort  # noqa: E402
from markdown_it import MarkdownIt  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.pipeline import ask  # noqa: E402
from src.ingestion import get_chroma_collection  # noqa: E402

load_dotenv()

POLICIES_DIR = Path(__file__).parent.parent / "data" / "policies"

markdown = MarkdownIt("commonmark", {"html": False}).enable("table")

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static",
)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")


@app.route("/health", methods=["GET"])
def health():
    try:
        collection = get_chroma_collection()
        index_size = collection.count()
        status = "ok"
    except Exception as e:
        index_size = 0
        status = f"degraded: {str(e)}"

    return (
        jsonify(
            {
                "status": status,
                "index_size": index_size,
                "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            }
        ),
        200,
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/policies/<filename>", methods=["GET"])
def policy_document(filename):
    """Serve a source policy document, rendered, so citations can link to it."""
    if filename not in {p.name for p in POLICIES_DIR.iterdir() if p.is_file()}:
        abort(404)

    text = (POLICIES_DIR / filename).read_text(encoding="utf-8")
    title = next(
        (
            line.lstrip("#").strip()
            for line in text.split("\n")
            if line.startswith("# ")
        ),
        filename,
    )

    return render_template(
        "policy.html",
        title=title,
        filename=filename,
        content=markdown.render(text),
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)

    if not data or "question" not in data:
        return jsonify({"error": "Request body must include a 'question' field."}), 400

    question = data["question"]

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Question must be a non-empty string."}), 400

    result = ask(question)

    if result.get("error") and not result.get("answer"):
        return jsonify({"error": result["error"]}), 500

    return (
        jsonify(
            {
                "answer": result["answer"],
                "sources": result["sources"],
                "latency_ms": result["latency_ms"],
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
