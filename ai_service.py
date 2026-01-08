"""
RAG Service - Document Processing, Embeddings, and Question Answering
"""
import os
import logging
import time
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGService:
    
    def __init__(self):
        # Configuration
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "200"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.top_k = int(os.getenv("TOP_K_RETRIEVAL", "3"))
        
        # Models
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.llm_model_name = os.getenv("LLM_MODEL", "google/flan-t5-base")
        
        # Initialize models (lazy loading)
        self._embedding_model = None
        self._qa_model = None
        
        # Vector store
        self.index = None
        self.chunks_metadata = []  # Store chunk text and metadata
        
        logger.info("RAG Service initialized")
    
    @property
    def embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
    
    @property
    def qa_model(self):
        """Lazy load QA model"""
        if self._qa_model is None:
            logger.info(f"Loading QA model: {self.llm_model_name}")
            self._qa_model = pipeline(
                "text2text-generation",
                model=self.llm_model_name,
                max_length=512,
                device=-1 
            )
        return self._qa_model
    
    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, int]:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted text, number of pages)
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    # Add page number marker for tracking
                    text += f"\n[PAGE {page_num + 1}]\n{page_text}"
                
                logger.info(f"Extracted {len(text)} characters from {num_pages} pages")
                return text, num_pages
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def chunk_text(self, text: str, document_id: int, filename: str) -> List[Dict]:
        """
            text: Full document text
            document_id: Database ID of document
            filename: Original filename
        """
        chunks = []
        
        # Split by pages first
        pages = text.split('PAGE')
        
        for page_section in pages[1:]:  # Skip empty first split
            # Extract page number
            page_num_end = page_section.find(']')
            page_num = int(page_section[:page_num_end])
            page_text = page_section[page_num_end + 1:].strip()
            
            # Split page into chunks
            words = page_text.split()
            
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page_num,
                        "chunk_index": len(chunks)
                    })
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for text chunks
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_index(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Build FAISS index for similarity search
        """
        dimension = embeddings.shape[1]
        
        # Create FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.chunks_metadata = chunks
        
        logger.info(f"Built FAISS index with {len(chunks)} vectors")
    
    async def process_document(
        self,
        file_path: str,
        document_id: int,
        filename: str
    ) -> Dict:
        """
            file_path: Path to uploaded file
            document_id: Database ID
            filename: Original filename
        """
        try:
            # Extract text
            text, num_pages = self.extract_text_from_pdf(file_path)
            
            # Chunk text
            chunks = self.chunk_text(text, document_id, filename)
            
            # Generate embeddings
            embeddings = self.create_embeddings(chunks)
            
            # Build index (append to existing)
            if self.index is None:
                self.build_index(embeddings, chunks)
            else:
                # Add to existing index
                self.index.add(embeddings.astype('float32'))
                self.chunks_metadata.extend(chunks)
            
            return {
                "num_pages": num_pages,
                "num_chunks": len(chunks),
                "status": "ready"
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                "num_pages": 0,
                "num_chunks": 0,
                "status": "failed"
            }
    
    async def answer_question(
        self,
        question: str,
        document_id: Optional[int] = None
    ) -> Dict:
        """
        Dictionary with answer and metadata
        """
        start_time = time.time()
        
        try:
            if self.index is None or len(self.chunks_metadata) == 0:
                return {
                    "answer": "No documents have been uploaded yet. Please upload a document first.",
                    "retrieved_chunks": [],
                    "confidence_score": 0.0,
                    "response_time": time.time() - start_time,
                    "sources": []
                }
            
            # Generate question embedding
            question_embedding = self.embedding_model.encode([question])[0]
            
            # Search in FAISS
            distances, indices = self.index.search(
                question_embedding.reshape(1, -1).astype('float32'),
                min(self.top_k, len(self.chunks_metadata))
            )
            
            # Retrieve chunks
            retrieved_chunks = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.chunks_metadata):
                    chunk = self.chunks_metadata[idx]
                    
                    # Convert L2 distance to similarity score (0-1)
                    similarity_score = 1 / (1 + distance)
                    
                    retrieved_chunks.append({
                        "text": chunk["text"],
                        "document_name": chunk["filename"],
                        "page_number": chunk["page_number"],
                        "similarity_score": float(similarity_score)
                    })
            
            if not retrieved_chunks:
                return {
                    "answer": "No relevant information found in the documents.",
                    "retrieved_chunks": [],
                    "confidence_score": 0.0,
                    "response_time": time.time() - start_time,
                    "sources": []
                }
            
            # Prepare context for LLM
            context = "\n\n".join([
                f"[From {chunk['document_name']}, Page {chunk['page_number']}]\n{chunk['text']}"
                for chunk in retrieved_chunks[:3]
            ])
            
            # Generate answer using LLM
            prompt = f"""Context: {context}

Question: {question}

Answer the question based only on the context provided. If the answer is not in the context, say "The information is not available in the documents."""
            
            answer = self.qa_model(prompt, max_length=200, do_sample=False)[0]["generated_text"]
            
            # Extract sources
            sources = list(set([
                f"{chunk['document_name']} (Page {chunk['page_number']})"
                for chunk in retrieved_chunks
            ]))
            
            # Calculate confidence based on similarity scores
            avg_similarity = np.mean([chunk["similarity_score"] for chunk in retrieved_chunks])
            
            response_time = time.time() - start_time
            
            return {
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "confidence_score": float(avg_similarity),
                "response_time": response_time,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": "An error occurred while processing your question. Please try again.",
                "retrieved_chunks": [],
                "confidence_score": 0.0,
                "response_time": time.time() - start_time,
                "sources": []
            }


# Singleton instance
rag_service = RAGService()
