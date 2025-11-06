"""Vector database integration using ChromaDB"""
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import get_settings
import uuid


class VectorStore:

    def __init__(self, collection_name: str = "documents"):
        """
        Initialize the vector store
        """
        settings = get_settings()
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents with their embeddings to the vector store
        
        Args:
            texts: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs (generated if not provided)
            
        Returns:
            List of document IDs
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
            
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform similarity search for the query embedding
        
        Args:
            query_embedding: Query vector
            k: Number of top results to return
            metadata_filter: Optional metadata filter (ChromaDB where clause)
            
        Returns:
            Dictionary containing:
                - documents: List of matched document texts
                - metadatas: List of metadata for each match
                - distances: List of similarity distances
                - ids: List of document IDs
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "name": self.collection.name,
            "document_count": count,
            "metadata": self.collection.metadata
        }
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(self.collection.name)
    
    def reset(self):
        """Reset the vector store (delete all documents)"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the vector store singleton"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

