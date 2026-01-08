"""
Pydantic schemas for Document Chatbot RAG System
"""
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, List


class DocumentUploadResponse(BaseModel):
    """
    Schema for document upload response
    """
    id: int
    filename: str
    file_size: int
    num_pages: Optional[int]
    num_chunks: Optional[int]
    status: str
    uploaded_at: datetime
    processed_at: Optional[datetime]

    class Config:
        from_attributes = True


class QuestionRequest(BaseModel):
    """
    Schema for asking questions
    """
    question: str = Field(..., min_length=5, max_length=500, description="Question to ask")

    @validator("question")
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty or only whitespace")
        return v.strip()


class RetrievedChunk(BaseModel):
    """
    Schema for retrieved document chunks
    """
    text: str
    document_name: str
    page_number: Optional[int]
    similarity_score: float


class AnswerResponse(BaseModel):
    """
    Schema for answer response
    """
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    confidence_score: float
    response_time: float
    sources: List[str]


class ChatHistoryResponse(BaseModel):
    """
    Schema for chat history response
    """
    id: int
    question: str
    answer: str
    confidence_score: Optional[float]
    response_time: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentUpdate(BaseModel):
    """
    Schema for updating document metadata
    """
    filename: Optional[str] = Field(None, min_length=1, max_length=255)
    status: Optional[str] = Field(None, pattern="^(processing|ready|failed)$")


class ErrorResponse(BaseModel):
    """
    Schema for error responses
    """
    detail: str
    error_code: Optional[str] = None
