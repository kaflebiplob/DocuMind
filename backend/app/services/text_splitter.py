def split_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split text into character based chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError("Chunk sixe must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non negative and smaller than chunk-size.")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap
    return chunks
