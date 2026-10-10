import pymupdf


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF stored in memory."""

    extracted_pages = []
    with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text()
            if not page_text:
                try:
                    ocr_page = page.get_textpage_ocr(language="eng", dpi=300, full=True)
                    page_text = page.get_text(textpage=ocr_page).strip()
                except Exception:
                    page_text = ""

            if page_text:
                extracted_pages.append(f"Page {page_number}\n{page_text}")

    return "\n\n".join(extracted_pages)
