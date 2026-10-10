from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.services.document_loader import extract_text_from_pdf
from app.services.text_splitter import split_text, search_chunks
from app.schemas.schemas import AskRequest
from app.services.document_store import get_documents, save_document
from app.services.llm import generate_answer

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please Upload a PDF file.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    try:
        text = await run_in_threadpool(extract_text_from_pdf, file_bytes)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the PDF. please upload a Valid PDF.",
        )

    if not text.strip():
        raise HTTPException(
            status_code=400, detail="No text found in PDF. It may be scanned document."
        )

    chunks = split_text(
        text,
        chunk_size=500,
        overlap=50,
    )

    document_id = save_document(file.filename, chunks)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "characters": len(text),
        # "text": text,
        # "chunks": chunks,
        "chunk_count": len(chunks),
    }


@router.post("/ask")
async def ask_question(body: AskRequest):
    document = get_documents(body.document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunks = document["chunks"]
    results = search_chunks(body.question, document["chunks"], top_k=5)
    if results:
        context_chunks = [r["text"] for r in results]
    elif len(chunks) <= 10:
        context_chunks = chunks
    else:
        return {"answer": "No relevant text found.", "sources": []}

    try:
        answer = await run_in_threadpool(generate_answer, body.question, context_chunks)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {e}")

    return {"answer": answer, "sources": results}
