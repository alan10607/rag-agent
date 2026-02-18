"""Tests for the ingestion pipeline (app.ingest.ingest)."""

import os
import re

import pytest

from ragent.indexing.indexer import (
    _clean_chinese_text,
    _extract_text_from_pdf,
    _find_page_number,
)


# ---------------------------------------------------------------------------
# Fixture: generate a minimal PDF with text on each page
# ---------------------------------------------------------------------------

def _write_raw_pdf(path, pages: int = 3):
    """Write a minimal raw PDF with English text content on each page."""
    pdf_bytes = bytearray()

    def w(s: str):
        pdf_bytes.extend(s.encode("latin-1"))

    page_texts = [f"Page {i} test content here." for i in range(1, pages + 1)]

    w("%PDF-1.4\n")
    obj_offsets: list[int] = []

    # obj 1: Font
    obj_offsets.append(len(pdf_bytes))
    w("1 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    # obj 2: Resources
    obj_offsets.append(len(pdf_bytes))
    w("2 0 obj\n<< /Font << /F1 1 0 R >> >>\nendobj\n")

    page_obj_start = 3
    content_obj_start = page_obj_start + pages
    pages_obj_num = content_obj_start + pages

    # Page objects
    for i in range(pages):
        obj_offsets.append(len(pdf_bytes))
        w(
            f"{page_obj_start + i} 0 obj\n"
            f"<< /Type /Page /Parent {pages_obj_num} 0 R "
            f"/MediaBox [0 0 612 792] /Resources 2 0 R "
            f"/Contents {content_obj_start + i} 0 R >>\n"
            f"endobj\n"
        )

    # Content stream objects
    for i in range(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({page_texts[i]}) Tj ET"
        obj_offsets.append(len(pdf_bytes))
        w(
            f"{content_obj_start + i} 0 obj\n"
            f"<< /Length {len(stream)} >>\n"
            f"stream\n{stream}\nendstream\n"
            f"endobj\n"
        )

    # Pages object
    kids = " ".join(f"{page_obj_start + i} 0 R" for i in range(pages))
    obj_offsets.append(len(pdf_bytes))
    w(
        f"{pages_obj_num} 0 obj\n"
        f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>\n"
        f"endobj\n"
    )

    # Catalog
    catalog_num = pages_obj_num + 1
    obj_offsets.append(len(pdf_bytes))
    w(
        f"{catalog_num} 0 obj\n"
        f"<< /Type /Catalog /Pages {pages_obj_num} 0 R >>\n"
        f"endobj\n"
    )

    # Xref
    xref_offset = len(pdf_bytes)
    total_objs = catalog_num
    w(f"xref\n0 {total_objs + 1}\n")
    w("0000000000 65535 f \n")
    for offset in obj_offsets:
        w(f"{offset:010d} 00000 n \n")

    w(
        f"trailer\n<< /Size {total_objs + 1} /Root {catalog_num} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )

    with open(path, "wb") as f:
        f.write(bytes(pdf_bytes))


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal 3-page PDF for testing."""
    pdf_path = tmp_path / "test.pdf"
    _write_raw_pdf(str(pdf_path), pages=3)
    return str(pdf_path)


# ---------------------------------------------------------------------------
# Tests: _clean_chinese_text
# ---------------------------------------------------------------------------

class TestCleanChineseText:
    """Tests for _clean_chinese_text function."""

    def test_removes_spaces_between_cjk_characters(self):
        assert _clean_chinese_text("孫 悟 空") == "孫悟空"

    def test_removes_consecutive_cjk_spaces(self):
        assert _clean_chinese_text("今 天 天 氣 很 好") == "今天天氣很好"

    def test_preserves_spaces_between_english_words(self):
        assert _clean_chinese_text("hello world") == "hello world"

    def test_preserves_mixed_cjk_english(self):
        result = _clean_chinese_text("使用 Python 開發")
        assert "Python" in result

    def test_removes_spaces_between_cjk_punctuation(self):
        assert _clean_chinese_text("你好 。 世界") == "你好。世界"

    def test_normalizes_multiple_blank_lines(self):
        text = "第一段\n\n\n\n\n第二段"
        result = _clean_chinese_text(text)
        assert "\n\n\n" not in result
        assert "第一段\n\n第二段" == result

    def test_preserves_single_newline(self):
        assert _clean_chinese_text("第一行\n第二行") == "第一行\n第二行"

    def test_strips_whitespace(self):
        assert _clean_chinese_text("  你好世界  ") == "你好世界"

    def test_empty_text(self):
        assert _clean_chinese_text("") == ""

    def test_tabs_between_cjk_removed(self):
        assert _clean_chinese_text("你\t好") == "你好"


# ---------------------------------------------------------------------------
# Tests: _extract_text_from_pdf
# ---------------------------------------------------------------------------

class TestExtractTextFromPdf:
    """Tests for _extract_text_from_pdf function."""

    def test_returns_tuple(self, sample_pdf):
        result = _extract_text_from_pdf(sample_pdf)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_full_text_is_string(self, sample_pdf):
        full_text, _ = _extract_text_from_pdf(sample_pdf)
        assert isinstance(full_text, str)
        assert len(full_text) > 0

    def test_page_map_has_correct_structure(self, sample_pdf):
        _, page_map = _extract_text_from_pdf(sample_pdf)
        assert len(page_map) > 0
        for entry in page_map:
            assert "page" in entry
            assert "start_char" in entry
            assert isinstance(entry["page"], int)
            assert isinstance(entry["start_char"], int)

    def test_page_numbers_are_sequential(self, sample_pdf):
        _, page_map = _extract_text_from_pdf(sample_pdf)
        pages = [entry["page"] for entry in page_map]
        assert pages == sorted(pages)

    def test_start_chars_are_ascending(self, sample_pdf):
        _, page_map = _extract_text_from_pdf(sample_pdf)
        starts = [entry["start_char"] for entry in page_map]
        assert starts == sorted(starts)

    def test_first_page_starts_at_zero(self, sample_pdf):
        _, page_map = _extract_text_from_pdf(sample_pdf)
        assert page_map[0]["start_char"] == 0

    def test_contains_all_page_content(self, sample_pdf):
        full_text, _ = _extract_text_from_pdf(sample_pdf)
        assert "Page 1" in full_text
        assert "Page 2" in full_text
        assert "Page 3" in full_text


# ---------------------------------------------------------------------------
# Tests: _find_page_number
# ---------------------------------------------------------------------------

class TestFindPageNumber:
    """Tests for _find_page_number function."""

    @pytest.fixture
    def page_map(self):
        return [
            {"page": 1, "start_char": 0},
            {"page": 2, "start_char": 100},
            {"page": 3, "start_char": 250},
            {"page": 4, "start_char": 400},
        ]

    def test_first_page(self, page_map):
        assert _find_page_number(0, page_map) == 1

    def test_middle_of_first_page(self, page_map):
        assert _find_page_number(50, page_map) == 1

    def test_start_of_second_page(self, page_map):
        assert _find_page_number(100, page_map) == 2

    def test_middle_of_last_page(self, page_map):
        assert _find_page_number(450, page_map) == 4

    def test_boundary_returns_new_page(self, page_map):
        assert _find_page_number(250, page_map) == 3

    def test_just_before_boundary(self, page_map):
        assert _find_page_number(99, page_map) == 1

    def test_empty_page_map(self):
        assert _find_page_number(0, []) is None
