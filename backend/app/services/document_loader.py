import pymupdf


def extract_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    """Return (page_number, text) pairs. Same OCR fallback as before."""
    pages = []
    with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text().strip()
            if not page_text:
                try:
                    ocr = page.get_textpage_ocr(language="eng", dpi=300, full=True)
                    page_text = page.get_text(textpage=ocr).strip()
                except Exception:
                    page_text = ""
            if page_text:
                pages.append((page_number, page_text))
    return pages
