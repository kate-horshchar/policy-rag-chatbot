from src.ingestion import chunk_text, extract_section_title, load_markdown


def test_chunk_text_basic():
    text = "Hello world. " * 100
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "text" in chunk
        assert len(chunk["text"]) <= 200  # allow some tolerance


def test_chunk_text_short():
    text = "Short text."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short text."


def test_chunk_text_empty():
    chunks = chunk_text("")
    assert chunks == []


def test_extract_section_title_found():
    text = "## Introduction\nSome text here.\n## Section Two\nMore text."
    offset = text.index("More text")
    title = extract_section_title(text, offset)
    assert title == "Section Two"


def test_extract_section_title_not_found():
    text = "No headings here. Just plain text."
    title = extract_section_title(text, 10)
    assert title == "General"


def test_load_markdown(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Title\n\nSome content.", encoding="utf-8")
    content = load_markdown(md_file)
    assert "Title" in content
    assert "Some content" in content
