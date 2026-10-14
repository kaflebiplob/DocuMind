from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TFIDF_OPTIONS = {"stop_words": "english"}
MIN_SCORE = 0.1


def build_index(texts: list[str]):
    vectorizer = TfidfVectorizer(**TFIDF_OPTIONS)
    matrix = vectorizer.fit_transform(texts)  # ValueError if no usable words
    return vectorizer, matrix


def search_index(query, vectorizer, matrix, top_k=3, min_score=MIN_SCORE):
    if not query.strip():
        return []
    scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    results = []
    for idx in scores.argsort()[::-1][:top_k]:
        score = float(scores[idx])
        if score > 0 and score >= min_score:
            results.append((int(idx), round(score, 4)))
    return results


def chunk_pages(pages, chunk_size=500, overlap=50):
    chunks = []
    for page_number, page_text in pages:
        for piece in split_text(page_text, chunk_size, overlap):
            chunks.append({"page": page_number, "text": piece})
    return chunks


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
        if end < len(text):
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if cut > start + chunk_size // 2:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        next_start = max(start + 1, end - overlap)
        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1  # begin the next chunk at a word boundary
        start = next_start
    return chunks


# Search chunks function


def search_chunks(
    query: str, chunks: list[str], top_k: int = 3, **tfidf_options
) -> list[dict]:
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
        print(
            "TF-IDF failed:", exc
        )  # every word was a stop word, so the vocabulary is empty
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
