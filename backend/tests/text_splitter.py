from app.services.text_splitter import split_text

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
