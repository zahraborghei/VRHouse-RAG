# Quick Start Guide

Get the RAG System up and running in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- Google Gemini API key ([Get one free here](https://makersuite.google.com/app/apikey))

## Installation Steps

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd VRHouse

# Create virtual environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Create .env file
cat > .env << EOF
GEMINI_API_KEY=your_actual_api_key_here
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
EOF

# Or copy from template
# See env_config.txt for all options
```

### 3. Start the Server

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at: http://localhost:8000

### 4. Test the System

```bash
# In a new terminal
python test_api.py
```

If all tests pass ✅, you're ready to go!

## First Steps

### View API Documentation

Open your browser:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Try Example Usage

```bash
python example_usage.py
```

This will walk you through:
- Ingesting documents
- Querying the system
- Using metadata filters
- Processing results

## Quick API Examples

### Ingest a Document

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Python is a programming language created by Guido van Rossum in 1991.",
    "metadata": {"topic": "programming"}
  }'
```

### Query the System

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who created Python?",
    "k": 3
  }'
```

### Check System Status

```bash
curl http://localhost:8000/api/v1/info
```

## Python Client Example

```python
import requests

# Ingest
response = requests.post(
    "http://localhost:8000/api/v1/ingest",
    json={
        "text": "Machine learning is a subset of AI.",
        "metadata": {"topic": "AI"}
    }
)
print("Ingested:", response.json()["doc_id"])

# Query
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={"question": "What is machine learning?"}
)
print("Answer:", response.json()["answer"])
```

## Common Issues

### "No module named 'app'"
Run from project root: `python -m uvicorn app.main:app --reload`

### "Connection refused"
Make sure the server is running on port 8000

### "GEMINI_API_KEY not found"
Check your .env file and ensure the API key is set

### "Import error: sentence_transformers"
Reinstall dependencies: `pip install -r requirements.txt`

## Next Steps

1. **Read the full README**: [README.md](README.md)
2. **Understand the architecture**: [docs/database_choice.md](docs/database_choice.md)
3. **Learn about data modeling**: [docs/data_modeling.md](docs/data_modeling.md)
4. **Try ingesting PDFs**: Place PDFs in `data/pdfs/` and use the `/ingest` endpoint



