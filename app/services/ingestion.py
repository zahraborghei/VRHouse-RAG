"""Document ingestion service"""
from typing import Dict, Any, List
import os
import uuid
from datetime import datetime
from pypdf import PdfReader

from app.core.config import get_settings
from app.core.embeddings import get_embedding_service
from app.core.vector_store import get_vector_store
from app.core.rag import get_text_chunker


class IngestionService:
    """
    Service for ingesting documents into the RAG system
    
    Handles:
    - PDF text extraction
    - Text chunking
    - Embedding generation
    - Vector storage with metadata
    """
    
    def __init__(self):
        """Initialize the ingestion service"""
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        self.text_chunker = get_text_chunker()
    
    def ingest_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest raw text into the RAG system
        
        Args:
            text: Raw text to ingest
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary with ingestion results
        """
        # Generate document ID
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not provided
        if 'date' not in metadata:
            metadata['date'] = datetime.now().isoformat()
        
        # Chunk the text
        chunks = self.text_chunker.chunk_text(text)
        
        if not chunks:
            raise ValueError("No valid chunks generated from the text")
        
        # Generate embeddings for all chunks
        embeddings = self.embedding_service.generate_embeddings(chunks)
        
        # Prepare metadata for each chunk
        chunk_metadatas = []
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            # Build metadata, filtering out None values (ChromaDB doesn't accept None)
            chunk_metadata = {
                "doc_id": doc_id,
                "source": metadata.get("source", "direct_input"),
                "chunk_index": i,
            }
            
            # Add optional fields only if they exist
            if metadata.get("date"):
                chunk_metadata["date"] = metadata["date"]
            if metadata.get("topic"):
                chunk_metadata["topic"] = metadata["topic"]
            
            # Add any other custom metadata fields (excluding None values)
            for k, v in metadata.items():
                if k not in ["source", "date", "topic", "doc_id", "chunk_index"] and v is not None:
                    chunk_metadata[k] = v
            
            chunk_metadatas.append(chunk_metadata)
        
        # Store in vector database
        self.vector_store.add_documents(
            texts=chunks,
            embeddings=embeddings,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )
        
        # Calculate approximate token count (rough estimate: 1 token ≈ 4 characters)
        total_chars = sum(len(chunk) for chunk in chunks)
        approx_tokens = total_chars // 4
        
        return {
            "success": True,
            "message": "Document ingested successfully",
            "doc_id": doc_id,
            "num_chunks": len(chunks),
            "num_tokens": approx_tokens,
            "embedding_dimension": self.embedding_service.dimension,
            "chunk_ids": chunk_ids
        }
    
    def ingest_pdf(
        self,
        file_path: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest a PDF file into the RAG system
        
        Args:
            file_path: Path to PDF file (relative to upload directory or absolute)
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary with ingestion results
        """
        # Resolve file path
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.settings.pdf_upload_path, file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Extract text from PDF
        text = self._extract_text_from_pdf(file_path)
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        # Add source to metadata
        if metadata is None:
            metadata = {}
        
        metadata["source"] = os.path.basename(file_path)
        
        # Ingest the extracted text
        return self.ingest_text(text, metadata)
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            reader = PdfReader(file_path)
            text_parts = []
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            return "\n\n".join(text_parts)
        
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics
        
        Returns:
            Dictionary with statistics
        """
        return self.vector_store.get_collection_stats()


# Singleton instance
_ingestion_service: IngestionService = None


def get_ingestion_service() -> IngestionService:
    """Get or create the ingestion service singleton"""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service

