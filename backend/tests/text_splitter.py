from app.services.text_splitter import split_text, search_chunks

text = """
Machine learning allows computers to learn patterns from data.
Supervised learning uses labeled examples.
Unsupervised learning discovers patterns without labels.
"""

chunks = split_text(text, chunk_size=60, overlap=10)

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")
    print(chunk)

print("\nTotal chunks:", len(chunks))


# search words test


def test_search_chunks():
    chunks = [
        "Machine Learning learns patterns from data.",
        "Supervised Learning uses labeled examples.",
        "unsupervised learning discover hidden patterns.",
    ]

    results = search_chunks(
        query="supervised learning labeled",
        chunks=chunks,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_index"] == 1
    assert results[0]["score"] == 3
    print("Search test passed")
    print(results)

if __name__ == "__main__":
    test_search_chunks()