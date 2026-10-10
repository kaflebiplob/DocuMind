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


# Search chunks function


def search_chunks(query: str, chunks: list[str], top_k: int = 3) -> list[dict]:
    """Find chunks that share most words with a query."""

    query_words = set(query.lower().split())
    if not query_words:
        return []

    results = []

    for index, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        match_words = query_words.intersection(chunk_words)

        if match_words:
            results.append(
                {
                    "chunk_index": index,
                    "score": len(match_words),
                    "text": chunk,
                }
            )

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_k]


