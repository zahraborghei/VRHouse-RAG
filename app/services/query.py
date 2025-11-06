"""Query service for RAG retrieval and answer generation"""
from typing import Dict, Any, List, Optional

from app.core.embeddings import get_embedding_service
from app.core.vector_store import get_vector_store
from app.core.rag import get_llm_service
from app.core.config import get_settings


class QueryService:
    """
    Service for querying the RAG system.
    Handles query embedding, vector search, and LLM answer generation.
    """
    
    def __init__(self):
        """Initialize the query service"""
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        self.llm_service = get_llm_service()
    
    def query(
        self,
        question: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process query through RAG pipeline: embedding → similarity search → LLM answer generation
        
        Args:
            question: User's question
            k: Number of top results to retrieve
            metadata_filter: Optional metadata filter for retrieval
            
        Returns:
            Dictionary with answer and sources
        """
        # Use default k if not provided
        if k is None:
            k = self.settings.top_k_results
        
        # Step 1: Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(question)
        
        # Step 2: Perform similarity search
        search_results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=k,
            metadata_filter=metadata_filter
        )
        
        # Step 3: Process and filter results
        context_chunks = self._process_search_results(search_results)
        
        # Step 4: Generate answer using LLM
        if context_chunks:
            answer = self.llm_service.generate_answer(
                question=question,
                context_chunks=context_chunks
            )
        else:
            answer = "No relevant context found to answer this question."
        
        return {
            "success": True,
            "question": question,
            "answer": answer,
            "context_chunks": context_chunks,
            "num_chunks_retrieved": len(context_chunks)
        }
    
    def _process_search_results(
        self,
        search_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process search results: apply similarity threshold and format context chunks
        
        Args:
            search_results: Raw results from vector store
            
        Returns:
            List of processed context chunks
        """
        processed_chunks = []
        
        documents = search_results.get("documents", [])
        metadatas = search_results.get("metadatas", [])
        distances = search_results.get("distances", [])
        ids = search_results.get("ids", [])
        
        for i, (doc, metadata, distance, chunk_id) in enumerate(
            zip(documents, metadatas, distances, ids)
        ):
            # Convert distance to similarity score (ChromaDB uses cosine distance)
            # Cosine distance = 1 - cosine similarity
            # So similarity = 1 - distance
            similarity_score = 1.0 - distance
            
            # Apply similarity threshold
            if similarity_score >= self.settings.similarity_threshold:
                processed_chunks.append({
                    "text": doc,
                    "metadata": metadata,
                    "similarity_score": round(similarity_score, 4),
                    "chunk_id": chunk_id
                })
        
        return processed_chunks
    
    def retrieve_only(
        self,
        question: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks without LLM answer generation (retrieval only)
        
        Args:
            question
            k: Number of top results to retrieve
            metadata_filter: Optional metadata filter
            
        Returns:
            List of retrieved context chunks
        """
        if k is None:
            k = self.settings.top_k_results
        
        query_embedding = self.embedding_service.generate_embedding(question)
        
        search_results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=k,
            metadata_filter=metadata_filter
        )
        
        return self._process_search_results(search_results)


# Singleton instance
_query_service: QueryService = None


def get_query_service() -> QueryService:
    """Get or create the query service singleton"""
    global _query_service
    if _query_service is None:
        _query_service = QueryService()
    return _query_service

