"""Main FastAPI application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from app.core.config import get_settings


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Returns:
        Configured FastAPI application
    """
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="""
        # RAG System API
        
        A Retrieval-Augmented Generation (RAG) system that converts text into embeddings,
        stores them in a vector database, and answers queries using retrieved context.
        
        ## Features
        
        - **Document Ingestion**: Upload PDFs or raw text
        - **Intelligent Chunking**: Overlap-based text splitting
        - **Semantic Search**: ChromaDB-powered similarity search
        - **LLM Integration**: Google Gemini for answer generation
        - **Metadata Filtering**: Filter by topic, date, source, etc.
        - **Source Citations**: Answers include source references
        
        ## Architecture
        
        1. **Embedding Layer**: sentence-transformers/all-MiniLM-L6-v2
        2. **Vector Store**: ChromaDB with cosine similarity
        3. **LLM Layer**: Google Gemini (complete separation from retrieval)
        
        ## Getting Started
        
        1. POST to `/ingest` to add documents
        2. POST to `/query` to ask questions
        3. GET `/info` for system statistics
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API router
    app.include_router(router, prefix="/api/v1", tags=["RAG"])
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup"""
        print(f"Starting {settings.api_title} v{settings.api_version}")
        print(f"Embedding model: {settings.embedding_model_name}")
        print(f"Vector DB: ChromaDB at {settings.chroma_persist_directory}")
        print(f"Chunk size: {settings.chunk_size} (overlap: {settings.chunk_overlap})")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        print("Shutting down RAG System API")
    
    return app


# Create app instance
app = create_application()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

