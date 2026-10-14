import uuid

_documents: dict[str, dict] = {}


def save_document(filename, chunks, vectorizer, matrix) -> str:
    document_id = str(uuid.uuid4())
    _documents[document_id] = {
        "filename": filename,
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }
    return document_id


def get_documents(document_id: str) -> dict | None:
    return _documents.get(document_id)
