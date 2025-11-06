"""Embedding generation service using sentence-transformers"""
from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings


class EmbeddingService:
    """Service for generating embeddings from text"""
    
    def __init__(self):
        """Initialize the embedding model"""
        settings = get_settings()
        self.model_name = settings.embedding_model_name
        self.dimension = settings.embedding_dimension
        self.model = SentenceTransformer(self.model_name)
        
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of float values representing the embedding vector
        """
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing)
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return [emb.tolist() for emb in embeddings]
    
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_sequence_length": self.model.max_seq_length
        }


# Singleton instance
_embedding_service: EmbeddingService = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

