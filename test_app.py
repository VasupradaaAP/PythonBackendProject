"""
Test suite for AI Document Chatbot RAG System
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, AsyncMock, MagicMock
import io

from app import app
from database import Base, get_db
from models import Document, ChatHistory

# Test database configuration
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ============================================
# Fixtures
# ============================================

@pytest.fixture(scope="function")
def test_db():
    """
    Create a fresh database for each test
    """
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(test_db):
    """
    Create test client with dependency override
    """
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def mock_rag_service():
    """
    Mock RAG service to avoid loading models and processing during testing
    """
    with patch("app.rag_service.process_document", new_callable=AsyncMock) as mock_process, \
         patch("app.rag_service.answer_question", new_callable=AsyncMock) as mock_answer:
        
        mock_process.return_value = {
            "num_pages": 5,
            "num_chunks": 10,
            "status": "ready"
        }
        
        mock_answer.return_value = {
            "answer": "This is a test answer from the RAG system.",
            "retrieved_chunks": [
                {
                    "text": "Test chunk 1",
                    "document_name": "test.pdf",
                    "page_number": 1,
                    "similarity_score": 0.95
                }
            ],
            "confidence_score": 0.85,
            "response_time": 0.5,
            "sources": ["test.pdf (Page 1)"]
        }
        
        yield {"process": mock_process, "answer": mock_answer}


# ============================================
# Unit Tests
# ============================================

def test_root_endpoint(client):
    """
    Test root endpoint returns correct response
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "version" in data
    assert "features" in data


# ============================================
# POST /documents Tests
# ============================================

def test_upload_document_success(client, mock_rag_service):
    """
    Test successful document upload and processing
    """
    # Create a fake PDF file
    pdf_content = b"%PDF-1.4 fake pdf content"
    files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
    
    response = client.post("/documents", files=files)
    
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert data["status"] == "ready"
    assert data["num_pages"] == 5
    assert data["num_chunks"] == 10
    assert "id" in data
    
    # Verify RAG service was called
    mock_rag_service["process"].assert_called_once()


def test_upload_document_invalid_filetype(client, mock_rag_service):
    """
    Test upload with non-PDF file
    """
    txt_content = b"This is a text file"
    files = {"file": ("test.txt", io.BytesIO(txt_content), "text/plain")}
    
    response = client.post("/documents", files=files)
    assert response.status_code == 422
    assert "PDF" in response.json()["detail"]


# ============================================
# GET /documents Tests
# ============================================

def test_get_documents_empty(client):
    """
    Test retrieving documents when database is empty
    """
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == []


def test_get_documents_with_data(client, mock_rag_service):
    """
    Test retrieving documents with existing data
    """
    # Upload test documents
    pdf_content = b"%PDF-1.4 fake pdf"
    files1 = {"file": ("doc1.pdf", io.BytesIO(pdf_content), "application/pdf")}
    files2 = {"file": ("doc2.pdf", io.BytesIO(pdf_content), "application/pdf")}
    
    client.post("/documents", files=files1)
    client.post("/documents", files=files2)
    
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_get_documents_with_status_filter(client, mock_rag_service):
    """
    Test retrieving documents with status filter
    """
    pdf_content = b"%PDF-1.4 fake pdf"
    files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
    client.post("/documents", files=files)
    
    response = client.get("/documents?status_filter=ready")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["status"] == "ready"


def test_get_documents_invalid_status_filter(client):
    """
    Test retrieving documents with invalid status filter
    """
    response = client.get("/documents?status_filter=invalid")
    assert response.status_code == 422


# ============================================
# POST /chat Tests
# ============================================

def test_ask_question_success(client, mock_rag_service):
    """
    Test successful question answering
    """
    question_data = {
        "question": "What is the main topic of the document?"
    }
    
    response = client.post("/chat", json=question_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == question_data["question"]
    assert "answer" in data
    assert "retrieved_chunks" in data
    assert "confidence_score" in data
    assert "sources" in data
    
    # Verify RAG service was called
    mock_rag_service["answer"].assert_called_once()


def test_ask_question_invalid(client, mock_rag_service):
    """
    Test asking invalid question (too short)
    """
    question_data = {
        "question": "Hi"  # Too short
    }
    
    response = client.post("/chat", json=question_data)
    assert response.status_code == 422


# ============================================
# GET /chat/history Tests
# ============================================

def test_get_chat_history_empty(client):
    """
    Test retrieving chat history when empty
    """
    response = client.get("/chat/history")
    assert response.status_code == 200
    assert response.json() == []


def test_get_chat_history_with_data(client, mock_rag_service):
    """
    Test retrieving chat history with existing data
    """
    # Ask some questions
    question1 = {"question": "What is AI?"}
    question2 = {"question": "What is machine learning?"}
    
    client.post("/chat", json=question1)
    client.post("/chat", json=question2)
    
    response = client.get("/chat/history")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


# ============================================
# PUT /documents/{document_id} Tests
# ============================================

def test_update_document_success(client, mock_rag_service):
    """
    Test successful document update
    """
    # Upload a document
    pdf_content = b"%PDF-1.4 fake pdf"
    files = {"file": ("original.pdf", io.BytesIO(pdf_content), "application/pdf")}
    upload_response = client.post("/documents", files=files)
    doc_id = upload_response.json()["id"]
    
    # Update the document
    update_data = {"filename": "updated.pdf"}
    
    response = client.put(f"/documents/{doc_id}", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "updated.pdf"


def test_update_document_not_found(client):
    """
    Test updating a non-existent document
    """
    update_data = {"filename": "updated.pdf"}
    
    response = client.put("/documents/9999", json=update_data)
    assert response.status_code == 404


# ============================================
# DELETE /documents/{document_id} Tests
# ============================================

def test_delete_document_success(client, mock_rag_service):
    """
    Test successful document deletion
    """
    # Upload a document
    pdf_content = b"%PDF-1.4 fake pdf"
    files = {"file": ("delete_me.pdf", io.BytesIO(pdf_content), "application/pdf")}
    upload_response = client.post("/documents", files=files)
    doc_id = upload_response.json()["id"]
    
    # Delete the document
    response = client.delete(f"/documents/{doc_id}")
    assert response.status_code == 204
    
    # Verify document no longer exists
    get_response = client.get("/documents")
    docs = get_response.json()
    assert not any(doc["id"] == doc_id for doc in docs)


def test_delete_document_not_found(client):
    """
    Test deleting a non-existent document
    """
    response = client.delete("/documents/9999")
    assert response.status_code == 404


# ============================================
# Integration Tests
# ============================================

def test_full_document_lifecycle(client, mock_rag_service):
    """
    Test complete document lifecycle: Upload -> Ask Question -> Update -> Delete
    """
    # Upload
    pdf_content = b"%PDF-1.4 fake pdf content"
    files = {"file": ("lifecycle.pdf", io.BytesIO(pdf_content), "application/pdf")}
    upload_response = client.post("/documents", files=files)
    assert upload_response.status_code == 201
    doc_id = upload_response.json()["id"]
    
    # Ask Question
    question = {"question": "What is in the document?"}
    chat_response = client.post("/chat", json=question)
    assert chat_response.status_code == 200
    assert chat_response.json()["answer"]
    
    # Update
    update_data = {"filename": "lifecycle_updated.pdf"}
    update_response = client.put(f"/documents/{doc_id}", json=update_data)
    assert update_response.status_code == 200
    assert update_response.json()["filename"] == "lifecycle_updated.pdf"
    
    # Delete
    delete_response = client.delete(f"/documents/{doc_id}")
    assert delete_response.status_code == 204


def test_multiple_questions(client, mock_rag_service):
    """
    Test asking multiple questions
    """
    # Ask multiple questions
    questions = [
        "What is machine learning?",
        "How does AI work?",
        "What are neural networks?"
    ]
    
    for q in questions:
        response = client.post("/chat", json={"question": q})
        assert response.status_code == 200
    
    # Check chat history
    history_response = client.get("/chat/history")
    history = history_response.json()
    assert len(history) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
