from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document_loader import extract_text_from_pdf
from app.services.text_splitter import split_text, search_chunks
from app.schemas.schemas import AskRequest
from app.services.document_store import get_documents, save_document

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please Upload a PDF file.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    try:
        text = extract_text_from_pdf(file_bytes)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not reach the PDF. please upload a Valid PDF.",
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

    return {
        "filename": file.filename,
        "characters": len(text),
        "text": text,
        "chunks": chunks,
    }


@router.post("/ask")
async def ask_question(body: AskRequest):
    document = get_documents(body.document_id)
    if not document:
        raise HTTPException(status_code=400, detail="Document not found.")

    results = search_chunks(body.question, document["chunks"], top_k=3)
    if not results:
        return {"answer": "No relevant text found.", "sources": []}

    return {"sources": results}
