import pymupdf


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF stored in memory."""

    extracted_page = []
    with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text()

            extracted_page.append(f"Page {page_number}\n{page_text}")

    return "\n\n".join(extracted_page)
