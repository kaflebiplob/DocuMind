from pydantic import BaseModel


class AskRequest(BaseModel):
    document_id: str
    question: str
class SearchRequest(BaseModel):
    question: str
    top_k: int = 3