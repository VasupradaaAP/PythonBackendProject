# 🤖 AI Document Chatbot – RAG System API

A FAST API for document-based question answering using **Retrieval-Augmented Generation (RAG)**. Users can upload PDF documents and ask questions to retrieve answers.

---

## Problem Understanding & Assumptions

### 🔍 Interpretation (Core Requirements)

The core problem is to design and implement a backend system that:

* Accepts **PDF documents** from users
* Extracts and processes document text
* Enables **question answering grounded only in uploaded documents**
* Provides **traceability** by returning retrieved context chunks and sources
* Exposes all functionality via a **clean, testable REST API**


### 🎯 Use Case Chosen

**Use Case:** *AI-Powered Document Question Answering (RAG-based)*

**Example Scenario:**

* A user uploads internal PDFs (technical docs, manuals, reports)
* The user asks questions like:

  > “What is machine learning mentioned in this document?”

* The system retrieves relevant passages and generates answers strictly from document context

This use case mirrors real-world needs in:

* Enterprise knowledge bases
* Legal / compliance document review
* Technical documentation assistants


### RAG Pipeline Design

```
┌─────────────┐
│  PDF Upload │
└──────┬──────┘
       ▼
┌─────────────────────┐
│ Text Extraction     │  PyPDF2: Extract text page-by-page
│ (Page Metadata)     │
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│ Text Chunking       │  500 words, 50-word overlap
│ (Sliding Window)    │  Preserves context at boundaries
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│ Embedding Generation│  sentence-transformers: all-MiniLM-L6-v2
│ (384-dim vectors)   │  Fast, lightweight, accurate
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│ FAISS Indexing      │  L2 distance, flat index
│ (Vector Storage)    │  ~1ms search for 10k chunks
└─────────────────────┘

              USER QUESTION
                    ▼
┌─────────────────────────────────────┐
│ Question Embedding → FAISS Search   │  Retrieve top-3 chunks
└──────┬──────────────────────────────┘
       ▼
┌─────────────────────┐
│ Context Assembly    │  Combine chunks with metadata
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│ LLM Generation      │  FLAN-T5-Base: context + question → answer
│ (FLAN-T5-Base)      │
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│ Response + Sources  │  Answer + citations + confidence
└─────────────────────┘
```

### 📌 Assumptions (Mandatory)

#### 1. Data Formats

* Only **PDF files** are supported
* PDFs are assumed to be **text-based** (not scanned images)
* Extracted text is treated as UTF-8 plain text

#### 2. External API / Model Reliability

* Embedding and LLM models run **locally**
* Models are assumed to be available at runtime via HuggingFace

#### 3. Business Logic Constraints

* Answers **must only come from retrieved document context**
* Only the **top-K most relevant chunks** are used to control hallucination

#### 4. Ambiguities & Chosen Approach

* **Ambiguity:** Should vector data be stored in the database?
* **Decision:** FAISS index is kept in memory for performance and simplicity

---

## Design Decisions

### 🗄️ Database Schema

#### Tables

**documents**

* Stores metadata about uploaded PDFs
* Tracks processing lifecycle (`processing`, `ready`, `failed`)

**chat_history**

* Stores every Q&A interaction
* Enables analytics, debugging, and auditability

---

### 🏗️ Project Structure

The project follows a **Layered Architecture** with clear separation of concerns:

```
TECHIE/
├── app.py          # API layer (FastAPI routes, middleware, handlers)
├── ai_service.py   # RAG logic (PDF processing, embeddings, FAISS, LLM)
├── database.py     # DB engine, session, initialization
├── models.py       # SQLAlchemy ORM models
├── schemas.py      # Pydantic validation schemas
├── test_app.py     # Unit & integration tests
├── uploads/        # Uploaded PDF storage
├── requirements.txt
└── README.md
```

**Why this structure?**

* Easy to test each layer independently
* Business logic is isolated from HTTP concerns
* AI logic can be swapped or scaled without touching API code

---

### ✅ Validation Logic

Beyond basic type checking, the system enforces:

* **Question validation**: minimum length, no empty or whitespace-only input
* **File validation**: only `.pdf` files allowed
* **Graceful fallbacks** when no vectors or documents are available

---

### 🌐 External API / Model Design

Model handling follows API best practices:

* **Rate Limits**: Implicitly controlled by FastAPI and server capacity
* **Timeouts**: Long operations are isolated in service layer
* **Lazy Loading**: Models load only when first needed to reduce startup time

---

## Solution Approach (Data Flow)

### Step-by-Step Walkthrough

1. **User uploads PDF** (`POST /documents`)
2. PDF is saved to disk and metadata stored in PostgreSQL
3. Text is extracted page-by-page using PyPDF2
4. Text is chunked using a sliding window strategy
5. Embeddings are generated using SentenceTransformers
6. Vectors are indexed in FAISS
7. Document status is updated to `ready`

**Question Flow:**

1. User submits question (`POST /chat`)
2. Question embedding is generated
3. FAISS retrieves top-K similar chunks
4. Context is assembled with metadata
5. FLAN-T5 generates an answer strictly from context
6. Response, sources, confidence, and timing are returned
7. Interaction is stored in `chat_history`

---

## Error Handling Strategy

### Global Exception Handling

* FastAPI global exception handlers ensure:

  * Consistent error format
  * No stack traces leaked to users
  * Proper HTTP status codes

### Failure Scenarios Covered

| Failure              | Handling                          |
| -------------------- | --------------------------------- |
| Database unavailable | Logged + 500 response             |
| PDF parsing failure  | Document marked `failed`          |
| Model loading error  | Graceful AI error message         |
| Empty vector index   | User informed to upload documents |
| Invalid input        | 422 with validation details       |

Logging is implemented at **INFO, WARNING, and ERROR** levels for observability.

---


## 🧪 Testing

### **Run All Tests**

```bash
pytest test_app.py -v
```

### **Run Specific Test Category**

```bash
# Document upload tests
pytest test_app.py -k "upload" -v

# Chat tests
pytest test_app.py -k "chat" -v

# Integration tests
pytest test_app.py -k "lifecycle" -v
```

### **Test Coverage**

The test suite includes:

**Unit Tests**:
- ✅ Input validation (Pydantic schemas)
- ✅ Database models
- ✅ RAG service

**Integration Tests**:
- ✅ POST /documents - Valid/invalid file uploads
- ✅ GET /documents - Retrieve all, with filters
- ✅ POST /chat - Question answering
- ✅ GET /chat/history - Chat history retrieval
- ✅ PUT /documents/{id} - Update metadata
- ✅ DELETE /documents/{id} - Delete and verify
- ✅ Full document lifecycle test


## How to Run the Project

### 🔧 Setup Instructions

#### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Configure Environment Variables

Create a `.env` file based on `.env.example`:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/doc_chatbot_db
```

#### 4. Run the Application

```bash
python app.py
```

---

### 📡 API Usage Examples

#### Upload Document

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@document.pdf"
```

#### Ask Question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

### 📡 API Documentation

#### **Base URL**: `http://localhost:8000`

#### **1. Upload Document (POST /documents)**

Upload a PDF document for processing.

**Request**:
```bash
curl -X POST "http://localhost:8000/documents" \
  -F "file=@/path/to/document.pdf"
```

**Response** (201 Created):
```json
{
  "id": 1,
  "filename": "document.pdf",
  "file_size": 245678,
  "num_pages": 10,
  "num_chunks": 25,
  "status": "ready",
  "uploaded_at": "2026-01-06T10:30:00",
  "processed_at": "2026-01-06T10:30:15"
}
```

**Error** (422 Unprocessable Entity):
```json
{
  "detail": "Only PDF files are supported",
  "error_code": "HTTP_422"
}
```

#### **2. Get All Documents (GET /documents)**

Retrieve all uploaded documents with optional filtering.

**Request**:
```bash
# Get all documents
curl "http://localhost:8000/documents"

# With filters
curl "http://localhost:8000/documents?status_filter=ready&limit=10"
```

**Response** (200 OK):
```json
[
  {
    "id": 1,
    "filename": "ml_guide.pdf",
    "file_size": 245678,
    "num_pages": 10,
    "num_chunks": 25,
    "status": "ready",
    "uploaded_at": "2026-01-06T10:30:00",
    "processed_at": "2026-01-06T10:30:15"
  }
]
```

#### **3. Ask Question (POST /chat)**

Ask a question about uploaded documents.

**Request**:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "document_id": 1
  }'
```

**Response** (200 OK):
```json
{
  "question": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
  "retrieved_chunks": [
    {
      "text": "Machine learning is a method of data analysis...",
      "document_name": "ml_guide.pdf",
      "page_number": 3,
      "similarity_score": 0.92
    }
  ],
  "confidence_score": 0.89,
  "response_time": 1.25,
  "sources": ["ml_guide.pdf (Page 3)", "ml_guide.pdf (Page 5)"]
}
```

#### **4. Get Chat History (GET /chat/history)**

Retrieve past Q&A interactions.

**Request**:
```bash
# All history
curl "http://localhost:8000/chat/history"

# Filter by document
curl "http://localhost:8000/chat/history?document_id=1&limit=20"
```

**Response** (200 OK):
```json
[
  {
    "id": 1,
    "document_id": 1,
    "question": "What is machine learning?",
    "answer": "Machine learning is...",
    "confidence_score": 0.89,
    "response_time": 1.25,
    "created_at": "2026-01-06T10:35:00"
  }
]
```

#### **5. Update Document (PUT /documents/{id})**

Update document metadata.

**Request**:
```bash
curl -X PUT "http://localhost:8000/documents/1" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "updated_name.pdf",
    "status": "ready"
  }'
```

**Response** (200 OK): Updated document object
**Error** (404 Not Found):
```json
{
  "detail": "Document with ID 999 not found",
  "error_code": "HTTP_404"
}
```

#### **6. Delete Document (DELETE /documents/{id})**

Delete a document and its file.

**Request**:
```bash
curl -X DELETE "http://localhost:8000/documents/1"
```

**Response** (204 No Content): Empty response
**Error** (404 Not Found): If document doesn't exist

---

### 📘 Interactive API Docs

* Link : `http://localhost:8000/docs`


---

