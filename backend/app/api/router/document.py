from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.services.document_loader import extract_pages
from app.services.text_splitter import (
    split_text,
    search_chunks,
    search_index,
    chunk_pages,
    build_index,
)
from app.schemas.schemas import AskRequest, SearchRequest
from app.services.document_store import get_documents, save_document
from app.services.llm import generate_answer

router = APIRouter(prefix="/documents", tags=["Documents"])


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TFIDF_OPTIONS = {"stop_words": "english"}
MIN_SCORE = 0.1
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@router.post("/documents")
def upload_document(file: UploadFile):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large (max 20 MB).")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "This file is not a valid PDF.")

    pages = extract_pages(data)
    chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(422, "No readable text found in this PDF.")
    try:
        vectorizer, matrix = build_index([c["text"] for c in chunks])
    except ValueError:
        raise HTTPException(422, "No searchable words found in this PDF.")

    document_id = save_document(file.filename, chunks, vectorizer, matrix)
    return {
        "document_id": document_id,
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
    }


@router.post("/documents/{document_id}/search")
def search_document(document_id: str, body: SearchRequest):
    doc = get_documents(document_id)
    if doc is None:
        raise HTTPException(404, "Document not found.")
    hits = search_index(
        body.question, doc["vectorizer"], doc["matrix"], min(max(body.top_k, 1), 10)
    )
    return {
        "found": bool(hits),
        "results": [
            {
                "page": doc["chunks"][i]["page"],
                "score": s,
                "text": doc["chunks"][i]["text"],
            }
            for i, s in hits
        ],
    }


@router.post("/ask")
async def ask_question(body: AskRequest):
    document = get_documents(body.document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunks = document["chunks"]
    hits = search_index(
        body.question, document["vectorizer"], document["matrix"], top_k=5
    )

    if hits:
        selected, mode = hits, "retrieved"
    elif len(chunks) <= 10:
        selected, mode = [(i, 0.0) for i in range(len(chunks))], "full_document"
    else:
        return {"answer": "I couldn't find that in the document.", "sources": []}

    context_chunks = [
        f"[Page {chunks[i]['page']}] {chunks[i]['text']}" for i, _ in selected
    ]

    sources = (
        []
        if mode == "full_document"
        else [
            {"page": chunks[i]["page"], "score": s, "text": chunks[i]["text"]}
            for i, s in selected
        ]
    )

    try:
        answer = await run_in_threadpool(generate_answer, body.question, context_chunks)
    except Exception:
        raise HTTPException(
            status_code=503, detail="The answer service is unavailable."
        )

    return {"answer": answer, "mode": mode, "sources": sources}
