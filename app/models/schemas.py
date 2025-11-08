"""Pydantic schemas for API requests and responses"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


# ==================== Ingestion Schemas ====================

class DocumentMetadata(BaseModel):
    """Metadata for a document chunk"""
    doc_id: str = Field(..., description="Unique document identifier")
    source: str = Field(..., description="Source file name or path")
    chunk_index: int = Field(..., description="Index of the chunk within the document")
    date: Optional[str] = Field(None, description="Document date (ISO format)")
    topic: Optional[str] = Field(None, description="Document topic or category")
    custom_fields: Optional[Dict[str, Any]] = Field(None, description="Additional custom metadata")


class IngestRequest(BaseModel):
    """Request model for document ingestion"""
    text: Optional[str] = Field(None, description="Raw text to ingest")
    file_path: Optional[str] = Field(None, description="Path to PDF file in the upload directory")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata (topic, date, etc.)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a sample document to be ingested into the RAG system.",
                "metadata": {
                    "topic": "machine_learning",
                    "date": "2025-11-03",
                    # "author": "John Doe"
                }
            }
        }


class IngestResponse(BaseModel):
    """Response model for document ingestion"""
    success: bool = Field(..., description="Whether ingestion was successful")
    message: str = Field(..., description="Status message")
    doc_id: str = Field(..., description="Generated document ID")
    num_chunks: int = Field(..., description="Number of chunks created")
    num_tokens: int = Field(..., description="Approximate total token count")
    embedding_dimension: int = Field(..., description="Dimension of embedding vectors")
    chunk_ids: List[str] = Field(..., description="IDs of stored chunks")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Document ingested successfully",
                "doc_id": "doc_12345",
                "num_chunks": 5,
                "num_tokens": 2500,
                "embedding_dimension": 384,
                "chunk_ids": ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
            }
        }


# ==================== Query Schemas ====================

class QueryRequest(BaseModel):
    """Request model for query"""
    question: str = Field(..., description="User's question")
    k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top results to retrieve"
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata filters (e.g., {'topic': 'machine_learning'})"
    )
    include_sources: Optional[bool] = Field(
        default=True,
        description="Whether to include source documents in response"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is machine learning?",
                "k": 5,
                "metadata_filter": {"topic": "machine_learning"},
                "include_sources": True
            }
        }


class ContextChunk(BaseModel):
    """A retrieved context chunk"""
    text: str = Field(..., description="Chunk text")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")
    similarity_score: float = Field(..., description="Similarity score (0-1, higher is better)")
    chunk_id: str = Field(..., description="Chunk ID")


class QueryResponse(BaseModel):
    """Response model for query"""
    success: bool = Field(..., description="Whether query was successful")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    context_chunks: List[ContextChunk] = Field(
        ...,
        description="Retrieved context chunks with sources"
    )
    num_chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence... [Source: doc_12345]",
                "context_chunks": [
                    {
                        "text": "Machine learning is a method of data analysis...",
                        "metadata": {
                            "doc_id": "doc_12345",
                            "source": "ml_basics.pdf",
                            "topic": "machine_learning"
                        },
                        "similarity_score": 0.92,
                        "chunk_id": "chunk_1"
                    }
                ],
                "num_chunks_retrieved": 5
            }
        }

# ==================== System Info Schemas ====================

class SystemInfo(BaseModel):
    """System information"""
    status: str = Field(..., description="System status")
    embedding_model: Dict[str, Any] = Field(..., description="Embedding model information")
    vector_store: Dict[str, Any] = Field(..., description="Vector store statistics")
    configuration: Dict[str, Any] = Field(..., description="RAG configuration")


