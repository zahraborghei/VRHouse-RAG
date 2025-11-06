# PDF Upload Directory

Place your PDF files here for ingestion into the RAG system.

## Usage

1. Copy your PDF files to this directory:
   ```bash
   cp ~/Downloads/document.pdf ./data/pdfs/
   ```

2. Ingest via API:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/ingest" \
     -H "Content-Type: application/json" \
     -d '{
       "file_path": "document.pdf",
       "metadata": {
         "topic": "your_topic",
         "date": "2025-11-03"
       }
     }'
   ```

3. Or use the Python client:
   ```python
   from example_usage import RAGClient
   
   client = RAGClient()
   result = client.ingest_pdf("document.pdf", metadata={"topic": "research"})
   print(result)
   ```

## Supported PDF Types

- Text-based PDFs (preferred)
- Searchable PDFs

## Tip

- Organize PDFs by topic for better metadata management

