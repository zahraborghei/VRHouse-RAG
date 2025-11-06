# RAG System - Retrieval-Augmented Generation

A **Retrieval-Augmented Generation (RAG)** system built with FastAPI, ChromaDB, and Google Gemini. Ingest documents, store embeddings in a vector database, and answer questions using retrieved context with source citations.

## Documentation

Detailed documentation is available in the `docs/` folder:

- **`database_choice.md`** - Justifies ChromaDB selection based on efficiency, scalability, and developer experience
- **`data_modeling.md`** - Explains the document chunk structure and how metadata improves filtering, retrieval speed, and RAG quality
- **`prompt_engineering.md`** - Describes prompt design strategies to reduce hallucinations and force source citations

---

## Prerequisites

- Python 3.9+
- Google Gemini API key ([Get one free](https://makersuite.google.com/app/apikey))

---

## Installation

1. **Clone and navigate to the project**
   ```bash
   git clone https://github.com/yourusername/VRHouse.git
   cd VRHouse
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   # Required
   GEMINI_API_KEY=your_api_key_here
   
   # Optional (defaults shown)
   CHROMA_PERSIST_DIRECTORY=./chroma_db
   EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   CHUNK_SIZE=500
   CHUNK_OVERLAP=50
   TOP_K_RESULTS=5
   SIMILARITY_THRESHOLD=0.5
   ```

5. **Start the server**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

   API available at: `http://localhost:8000`  
   Interactive docs: `http://localhost:8000/docs`

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | - | Google Gemini API key |
| `CHROMA_PERSIST_DIRECTORY` | No | `./chroma_db` | Vector database storage path |
| `EMBEDDING_MODEL_NAME` | No | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` | No | `500` | Text chunk size (characters) |
| `CHUNK_OVERLAP` | No | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | No | `5` | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | No | `0.5` | Minimum similarity score (0-1) |
| `EMBEDDING_DIMENSION` | No | `384` | Embedding vector dimension |

---

## API Endpoints

### 1. Health Check

**`GET /api/v1/health`**

**Response:**
```json
{
  "status": "healthy",
  "message": "RAG System API is operational"
}
```

---

### 2. System Info

**`GET /api/v1/info`**

**Response:**
```json
{
  "status": "operational",
  "embedding_model": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
    "max_sequence_length": 256
  },
  "vector_store": {
    "name": "documents",
    "document_count": 150
  },
  "configuration": {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k_results": 5,
    "similarity_threshold": 0.5
  }
}
```

---

### 3. Ingest Documents

**`POST /api/v1/ingest`**

**Request (Raw Text):**
```json
{
  "text": "Python is a high-level programming language created by Guido van Rossum in 1991.",
  "metadata": {
    "topic": "programming",
    "author": "John Doe",
    "date": "2024-01-15"
  }
}
```

**Request (PDF File):**
```json
{
  "file_path": "research_paper.pdf",
  "metadata": {
    "topic": "machine_learning",
    "date": "2024-01-15"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Document ingested successfully",
  "doc_id": "doc_abc12345",
  "num_chunks": 12,
  "num_tokens": 3000,
  "embedding_dimension": 384,
  "chunk_ids": [
    "doc_abc12345_chunk_0",
    "doc_abc12345_chunk_1",
    "doc_abc12345_chunk_2"
  ]
}
```

---

### 4. Query System

**`POST /api/v1/query`**

**Request:**
```json
{
  "question": "Who created Python?",
  "k": 5,
  "metadata_filter": {
    "topic": "programming"
  },
  "include_sources": true
}
```

**Response:**
```json
{
  "success": true,
  "question": "Who created Python?",
  "answer": "Python was created by Guido van Rossum in 1991. [Source: doc_abc12345]",
  "context_chunks": [
    {
      "text": "Python is a high-level programming language created by Guido van Rossum in 1991.",
      "metadata": {
        "doc_id": "doc_abc12345",
        "source": "programming_basics.txt",
        "topic": "programming",
        "chunk_index": 0
      },
      "similarity_score": 0.92,
      "chunk_id": "doc_abc12345_chunk_0"
    }
  ],
  "num_chunks_retrieved": 5
}
```

**Parameters:**
- `question` (required): Your question
- `k` (optional): Number of chunks to retrieve (default: 5)
- `metadata_filter` (optional): Filter by metadata fields
- `include_sources` (optional): Include source chunks (default: true)

---

## Sample Usage

### Using cURL

**Ingest a document (text):**
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Python is a high-level programming language.",
    "metadata": {"topic": "programming"}
  }'
```

**Ingest a document (PDF):**
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "research_paper.pdf",
    "metadata": {"topic": "machine_learning", "date": "2024-01-15"}
  }'
```

**Query the system:**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Python?",
    "k": 3
  }'
```

---

### Using Python

**Ingest and query:**
```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Ingest document
ingest_response = requests.post(
    f"{BASE_URL}/ingest",
    json={
        "text": "Python was created by Guido van Rossum in 1991.",
        "metadata": {"topic": "programming", "language": "python"}
    }
)
print(ingest_response.json())

# Query
query_response = requests.post(
    f"{BASE_URL}/query",
    json={
        "question": "Who created Python?",
        "k": 3,
        "metadata_filter": {"topic": "programming"}
    }
)

result = query_response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['num_chunks_retrieved']} chunks")
```

---

### Python Client Class

```python
import requests
from typing import Dict, Any, Optional

class RAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def ingest(
        self, 
        text: Optional[str] = None, 
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest text or PDF document"""
        payload = {"metadata": metadata or {}}
        if text:
            payload["text"] = text
        elif file_path:
            payload["file_path"] = file_path
        else:
            raise ValueError("Must provide either text or file_path")
        
        response = requests.post(f"{self.base_url}/api/v1/ingest", json=payload)
        response.raise_for_status()
        return response.json()
    
    def query(
        self, 
        question: str, 
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Query the RAG system"""
        response = requests.post(
            f"{self.base_url}/api/v1/query",
            json={
                "question": question,
                "k": k,
                "metadata_filter": metadata_filter,
                "include_sources": include_sources
            }
        )
        response.raise_for_status()
        return response.json()
    
    def health(self) -> Dict[str, Any]:
        """Check system health"""
        response = requests.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()

# Usage
client = RAGClient()

# Ingest
client.ingest(
    text="Python emphasizes code readability and simplicity.",
    metadata={"topic": "programming", "language": "python"}
)

# Query
result = client.query("What does Python emphasize?", k=3)
print(result["answer"])
```
