from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split text into character based chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0.")
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


def search_chunks(query: str, chunks: list[str], top_k: int = 3,**tfidf_options) -> list[dict]:
    """Find the most relevant chunks using TF-IDF similarity."""

    query_words = set(query.lower().split())
    if not query_words:
        return []

    if not query.strip() or not chunks:
        return []

    vectorizer = TfidfVectorizer(**tfidf_options)

    try:
        chunk_vectors = vectorizer.fit_transform(chunks)
    except ValueError as exc:
        print("TF-IDF failed:", exc)  # every word was a stop word, so the vocabulary is empty
        return []
    query_vector = vectorizer.transform([query])

    # calculate the similarity between query and chunks
    scores = cosine_similarity(query_vector, chunk_vectors).flatten()

    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:
        score = float(scores[index])

        if score > 0:
            results.append(
                {
                    "score": round(score, 4),
                    "chunk_index": int(index),
                    "text": chunks[index],
                }
            )

    return results
