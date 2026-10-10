from fastapi import FastAPI
from app.api.router.document import router as document_router

app = FastAPI(
    title="DocuMind API", description="Ask questions about your own documents."
)

app.include_router(document_router)

@app.get("/")
def home():
    return {"message": "Welcome to DocuMind!!!"}


@app.get("/health")
def health():
    return {"status": "I am Healthy"}
