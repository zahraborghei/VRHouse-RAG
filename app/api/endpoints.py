"""FastAPI endpoint implementations"""
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from app.models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ContextChunk,
    SystemInfo
)
from app.services.ingestion import get_ingestion_service
from app.services.query import get_query_service
from app.core.embeddings import get_embedding_service
from app.core.vector_store import get_vector_store
from app.core.config import get_settings


router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest documents",
    description="""
    Ingest documents into the RAG system.
    
    You can provide either:
    - **text**: Raw text to ingest directly
    - **file_path**: Path to a PDF file in the upload directory

    Returns a summary including document ID, number of chunks, and embedding dimensions.
    """
)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    """
    Ingest a document into the RAG system
    
    Args:
        request: Ingestion request with text or file path
        
    Returns:
        Ingestion response with summary
    """
    try:
        ingestion_service = get_ingestion_service()
        
        # Validate input
        if not request.text and not request.file_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'text' or 'file_path' must be provided"
            )
        
        if request.text and request.file_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide either 'text' or 'file_path', not both"
            )
        
        # Process based on input type
        if request.text:
            result = ingestion_service.ingest_text(
                text=request.text,
                metadata=request.metadata
            )
        else:
            result = ingestion_service.ingest_pdf(
                file_path=request.file_path,
                metadata=request.metadata
            )
        
        return IngestResponse(**result)
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG system",
    description="""
    Query the RAG system to get answers based on ingested documents.
        
    Parameters:
    - **question**: Your question
    - **k**: Number of chunks to retrieve (default: 5, max: 20)
    - **metadata_filter**: Optional filters (e.g., {"topic": "machine_learning"})
    - **include_sources**: Whether to include source chunks in response
    """
)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG system
    
    Args:
        request: Query request with question and parameters
        
    Returns:
        Query response with answer and sources
    """
    try:
        query_service = get_query_service()
        
        # Process query
        result = query_service.query(
            question=request.question,
            k=request.k,
            metadata_filter=request.metadata_filter
        )
        
        # Format context chunks
        context_chunks = [
            ContextChunk(**chunk)
            for chunk in result["context_chunks"]
        ]
        
        # Optionally exclude sources from response
        if not request.include_sources:
            context_chunks = []
        
        return QueryResponse(
            success=result["success"],
            question=result["question"],
            answer=result["answer"],
            context_chunks=context_chunks,
            num_chunks_retrieved=result["num_chunks_retrieved"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )

@router.get(
    "/info",
    response_model=SystemInfo,
    summary="Get system information",
    description="Get information about the RAG system configuration and statistics"
)
async def get_system_info() -> SystemInfo:
    """
    Get system information and statistics
    
    Returns:
        System information including model details and vector store stats
    """
    try:
        settings = get_settings()
        embedding_service = get_embedding_service()
        vector_store = get_vector_store()
        
        return SystemInfo(
            status="operational",
            embedding_model=embedding_service.get_model_info(),
            vector_store=vector_store.get_collection_stats(),
            configuration={
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k_results": settings.top_k_results,
                "similarity_threshold": settings.similarity_threshold,
                "embedding_dimension": settings.embedding_dimension
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system info: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health check",
    description="Check if the API is running"
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint
    
    Returns:
        Status message
    """
    return {"status": "healthy", "message": "RAG System API is operational"}


