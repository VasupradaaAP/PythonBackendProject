"""
FastAPI Application - AI Document Chatbot RAG System
"""
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import os
import shutil
from pathlib import Path

from database import get_db, init_db
from models import Document, ChatHistory
from schemas import (
    DocumentUploadResponse, QuestionRequest, AnswerResponse,
    ChatHistoryResponse, DocumentUpdate, ErrorResponse, RetrievedChunk
)
from ai_service import rag_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Document Chatbot RAG API",
    description="REST API for document-based Q&A using RAG (Retrieval-Augmented Generation)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================
# Exception Handlers
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Global exception handler for HTTP exceptions
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": f"HTTP_{exc.status_code}"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Global exception handler for unexpected errors
    """
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred. Please try again later.",
            "error_code": "INTERNAL_SERVER_ERROR"
        }
    )


# ============================================
# Startup Event
# ============================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize database on startup
    """
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise


# ============================================
# API Endpoints
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API health check
    """
    return {
        "message": "AI Document Chatbot RAG API",
        "status": "operational",
        "version": "1.0.0",
        "features": ["PDF Upload", "RAG-based Q&A", "Vector Search", "Chat History"]
    }


@app.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    responses={
        201: {"description": "Document uploaded and processed successfully"},
        422: {"model": ErrorResponse, "description": "Invalid file format"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    }
)
async def upload_document(
    file: UploadFile = File(..., description="PDF document to upload"),
    db: Session = Depends(get_db)
):
    """
    Upload and process a PDF document
    
    This endpoint:
    1. Validates file type (PDF only)
    2. Saves file to uploads directory
    3. Extracts text and creates chunks
    4. Generates embeddings using sentence-transformers
    5. Stores in FAISS vector database
    6. Saves metadata to PostgreSQL
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Only PDF files are supported"
            )
        
        # Create document entry in database
        new_document = Document(
            filename=file.filename,
            file_path="",  # Will update after saving
            file_size=0,
            status="processing"
        )
        db.add(new_document)
        db.commit()
        db.refresh(new_document)
        
        # Save file
        file_path = UPLOAD_DIR / f"{new_document.id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update file path and size
        new_document.file_path = str(file_path)
        new_document.file_size = os.path.getsize(file_path)
        db.commit()
        
        logger.info(f"File uploaded: {file.filename} (ID: {new_document.id})")
        
        # Process document with RAG service (extract, chunk, embed, index)
        result = await rag_service.process_document(
            file_path=str(file_path),
            document_id=new_document.id,
            filename=file.filename
        )
        
        # Update document status
        new_document.num_pages = result["num_pages"]
        new_document.num_chunks = result["num_chunks"]
        new_document.status = result["status"]
        
        if result["status"] == "ready":
            from datetime import datetime
            new_document.processed_at = datetime.utcnow()
        
        db.commit()
        db.refresh(new_document)
        
        logger.info(f"Document processed: ID={new_document.id}, Status={new_document.status}")
        return new_document
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload and process document. Please try again."
        )


@app.get(
    "/documents",
    response_model=List[DocumentUploadResponse],
    tags=["Documents"],
    responses={
        200: {"description": "List of documents retrieved successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_documents(
    skip: int = 0,
    limit: int = 100,
    status_filter: str = None,
    db: Session = Depends(get_db)
):
    """
    Retrieve all uploaded documents with optional filtering
    """
    try:
        query = db.query(Document)
        
        # Apply status filter if provided
        if status_filter:
            if status_filter not in ["processing", "ready", "failed"]:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid status filter. Must be: processing, ready, or failed"
                )
            query = query.filter(Document.status == status_filter)
        
        documents = query.order_by(Document.uploaded_at.desc()).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents. Please try again."
        )


@app.post(
    "/chat",
    response_model=AnswerResponse,
    tags=["Chat"],
    responses={
        200: {"description": "Answer generated successfully"},
        422: {"model": ErrorResponse, "description": "Invalid question"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def ask_question(
    request: QuestionRequest,
    db: Session = Depends(get_db)
):
    """
    Ask a question:
    
    1. Generates embedding for the question
    2. Searches FAISS vector database for relevant chunks
    3. Retrieves top-k most similar document chunks
    4. Uses FLAN-T5-Base to generate answer from context
    5. Saves Q&A to chat history
    """
    try:
        # Generate answer using RAG
        result = await rag_service.answer_question(
            question=request.question
        )
        
        # Save to chat history
        chat_entry = ChatHistory(
            question=request.question,
            answer=result["answer"],
            retrieved_chunks=str(result["retrieved_chunks"]),
            confidence_score=result["confidence_score"],
            response_time=result["response_time"]
        )
        db.add(chat_entry)
        db.commit()
        
        logger.info(f"Question answered: '{request.question[:50]}...'")
        
        # Format response
        return AnswerResponse(
            question=request.question,
            answer=result["answer"],
            retrieved_chunks=[
                RetrievedChunk(**chunk) for chunk in result["retrieved_chunks"]
            ],
            confidence_score=result["confidence_score"],
            response_time=result["response_time"],
            sources=result["sources"]
        )
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer. Please try again."
        )


@app.get(
    "/chat/history",
    response_model=List[ChatHistoryResponse],
    tags=["Chat"],
    responses={
        200: {"description": "Chat history retrieved successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_chat_history(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Retrieve chat history
    """
    try:
        query = db.query(ChatHistory)
        
        history = query.order_by(ChatHistory.created_at.desc()).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(history)} chat history entries")
        return history
        
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat history. Please try again."
        )


@app.put(
    "/documents/{document_id}",
    response_model=DocumentUploadResponse,
    tags=["Documents"],
    responses={
        200: {"description": "Document updated successfully"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def update_document(
    document_id: int,
    update_data: DocumentUpdate,
    db: Session = Depends(get_db)
):
    """
    Update document metadata
    """
    try:
        # Find existing document
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found"
            )
        
        # Update fields if provided
        update_dict = update_data.model_dump(exclude_unset=True)
        
        for field, value in update_dict.items():
            setattr(document, field, value)
        
        db.commit()
        db.refresh(document)
        
        logger.info(f"Document {document_id} updated successfully")
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document. Please try again."
        )


@app.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Documents"],
    responses={
        204: {"description": "Document deleted successfully"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """
    Delete a document and its file
    """
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found"
            )
        
        # Delete physical file
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
            logger.info(f"Deleted file: {document.file_path}")
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        logger.info(f"Document {document_id} deleted successfully")
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document. Please try again."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
