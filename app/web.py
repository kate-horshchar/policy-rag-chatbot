import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from src.pipeline import ask
from src.ingestion import get_chroma_collection

load_dotenv()

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

    return jsonify({
        "status": status,
        "index_size": index_size,
        "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    }), 200


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


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

    return jsonify({
        "answer": result["answer"],
        "sources": result["sources"],
        "latency_ms": result["latency_ms"],
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
