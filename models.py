"""
SQLAlchemy database models for Document Chatbot RAG System
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, LargeBinary
from datetime import datetime
from database import Base


class Document(Base):
    """
    Document model representing uploaded documents
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, index=True)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    num_pages = Column(Integer, nullable=True)
    num_chunks = Column(Integer, nullable=True)
    status = Column(String(50), default="processing")  # processing, ready, failed
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"


class ChatHistory(Base):
    """
    Chat history model for storing Q&A interactions
    """
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    retrieved_chunks = Column(Text, nullable=True)  # JSON string
    confidence_score = Column(Float, nullable=True)
    response_time = Column(Float, nullable=True)  # in seconds
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ChatHistory(id={self.id}, question='{self.question[:50]}...')>"
