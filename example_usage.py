"""
Example usage of the RAG System API

This script demonstrates how to:
1. Ingest documents (text and PDFs)
2. Query the system
3. Use metadata filtering
4. Process results
"""

import requests
import json
from typing import Dict, Any, Optional


class RAGClient:
    """Client for interacting with the RAG System API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy"""
        response = requests.get(f"{self.api_url}/health")
        return response.json()
    
    def get_info(self) -> Dict[str, Any]:
        """Get system information"""
        response = requests.get(f"{self.api_url}/info")
        return response.json()
    
    def ingest_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest raw text into the system
        
        Args:
            text: Document text
            metadata: Optional metadata (topic, date, etc.)
        
        Returns:
            Ingestion response
        """
        payload = {
            "text": text,
            "metadata": metadata or {}
        }
        response = requests.post(
            f"{self.api_url}/ingest",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def ingest_pdf(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a PDF file
        
        Args:
            file_path: Path to PDF file (relative to data/pdfs/)
            metadata: Optional metadata
        
        Returns:
            Ingestion response
        """
        payload = {
            "file_path": file_path,
            "metadata": metadata or {}
        }
        response = requests.post(
            f"{self.api_url}/ingest",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def query(
        self,
        question: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
            metadata_filter: Optional metadata filter
            include_sources: Whether to include source chunks
        
        Returns:
            Query response with answer and sources
        """
        payload = {
            "question": question,
            "k": k,
            "metadata_filter": metadata_filter,
            "include_sources": include_sources
        }
        response = requests.post(
            f"{self.api_url}/query",
            json=payload
        )
        response.raise_for_status()
        return response.json()


def example_1_basic_ingestion():
    """Example 1: Basic text ingestion"""
    print("\n" + "="*60)
    print("Example 1: Basic Text Ingestion")
    print("="*60)
    
    client = RAGClient()
    
    # Check health
    health = client.health_check()
    print(f"API Status: {health['status']}")
    
    # Ingest a document about Python
    text = """
    Python is a high-level, interpreted programming language known for its 
    simplicity and readability. Created by Guido van Rossum and first released 
    in 1991, Python emphasizes code readability with its notable use of 
    significant indentation.
    
    Python supports multiple programming paradigms, including structured, 
    object-oriented, and functional programming. It features dynamic typing 
    and automatic memory management, making it easy to learn for beginners 
    while powerful enough for experts.
    
    Popular use cases include web development (Django, Flask), data science 
    (pandas, NumPy), machine learning (TensorFlow, PyTorch), automation, 
    and scripting.
    """
    
    result = client.ingest_text(
        text=text,
        metadata={
            "topic": "programming",
            "language": "python",
            "date": "2025-11-03"
        }
    )
    
    print(f"\n✅ Ingestion successful!")
    print(f"   Document ID: {result['doc_id']}")
    print(f"   Chunks created: {result['num_chunks']}")
    print(f"   Total tokens: {result['num_tokens']}")
    print(f"   Embedding dimension: {result['embedding_dimension']}")


def example_2_basic_query():
    """Example 2: Basic query"""
    print("\n" + "="*60)
    print("Example 2: Basic Query")
    print("="*60)
    
    client = RAGClient()
    
    # Query about Python
    result = client.query(
        question="What is Python and who created it?",
        k=3
    )
    
    print(f"\n❓ Question: {result['question']}")
    print(f"\n💡 Answer:\n{result['answer']}")
    print(f"\n📚 Sources: {result['num_chunks_retrieved']} chunks retrieved")
    
    if result['context_chunks']:
        print("\n📄 Top Source:")
        top_chunk = result['context_chunks'][0]
        print(f"   Similarity: {top_chunk['similarity_score']:.3f}")
        print(f"   Source: {top_chunk['metadata'].get('language', 'N/A')}")
        print(f"   Text preview: {top_chunk['text'][:100]}...")


def example_3_metadata_filtering():
    """Example 3: Query with metadata filtering"""
    print("\n" + "="*60)
    print("Example 3: Metadata Filtering")
    print("="*60)
    
    client = RAGClient()
    
    # First, ingest documents with different topics
    documents = [
        {
            "text": "Machine learning is a subset of artificial intelligence...",
            "metadata": {"topic": "machine_learning", "date": "2024-01-15"}
        },
        {
            "text": "Web development involves creating websites and web applications...",
            "metadata": {"topic": "web_development", "date": "2024-02-20"}
        },
        {
            "text": "Neural networks are computing systems inspired by biological neural networks...",
            "metadata": {"topic": "machine_learning", "date": "2024-03-10"}
        }
    ]
    
    print("\n📥 Ingesting documents with different topics...")
    for doc in documents:
        result = client.ingest_text(doc["text"], doc["metadata"])
        print(f"   ✓ Ingested: {result['doc_id']} (topic: {doc['metadata']['topic']})")
    
    # Query with metadata filter
    print("\n🔍 Querying with metadata filter (topic='machine_learning')...")
    result = client.query(
        question="Tell me about neural networks",
        k=2,
        metadata_filter={"topic": "machine_learning"}
    )
    
    print(f"\n💡 Answer:\n{result['answer']}")
    print(f"\n📊 Retrieved {result['num_chunks_retrieved']} chunks")
    print("\n   All chunks have topic='machine_learning' ✓")


def example_4_multiple_queries():
    """Example 4: Multiple queries on the same dataset"""
    print("\n" + "="*60)
    print("Example 4: Multiple Queries")
    print("="*60)
    
    client = RAGClient()
    
    # Ingest a longer document
    text = """
    Artificial Intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans.
    
    Machine Learning is a subset of AI that enables systems to learn and 
    improve from experience without being explicitly programmed. It focuses 
    on the development of computer programs that can access data and use it 
    to learn for themselves.
    
    Deep Learning is a subset of machine learning that uses neural networks 
    with multiple layers (deep neural networks). These networks can learn 
    complex patterns in large amounts of data.
    
    Natural Language Processing (NLP) is a branch of AI that helps computers 
    understand, interpret and manipulate human language. It combines 
    computational linguistics with machine learning.
    
    Computer Vision is a field of AI that trains computers to interpret and 
    understand the visual world. Using digital images and deep learning models, 
    machines can accurately identify and classify objects.
    """
    
    print("\n📥 Ingesting AI knowledge base...")
    client.ingest_text(
        text=text,
        metadata={"topic": "artificial_intelligence", "category": "overview"}
    )
    print("   ✓ Ingestion complete")
    
    # Ask multiple questions
    questions = [
        "What is Machine Learning?",
        "Explain Deep Learning",
        "What does NLP stand for?",
        "What is the relationship between AI and Machine Learning?"
    ]
    
    print("\n❓ Asking multiple questions...\n")
    for i, question in enumerate(questions, 1):
        result = client.query(question=question, k=2)
        print(f"{i}. Q: {question}")
        print(f"   A: {result['answer'][:150]}...")
        print()


def example_5_system_info():
    """Example 5: Get system information"""
    print("\n" + "="*60)
    print("Example 5: System Information")
    print("="*60)
    
    client = RAGClient()
    
    info = client.get_info()
    
    print("\n📊 System Status:", info['status'])
    
    print("\n🧠 Embedding Model:")
    model_info = info['embedding_model']
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    print("\n💾 Vector Store:")
    vs_info = info['vector_store']
    for key, value in vs_info.items():
        print(f"   {key}: {value}")
    
    print("\n⚙️  Configuration:")
    config = info['configuration']
    for key, value in config.items():
        print(f"   {key}: {value}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("RAG SYSTEM API - USAGE EXAMPLES")
    print("="*60)
    print("\nMake sure the API server is running:")
    print("  python -m uvicorn app.main:app --reload")
    print("\nPress Enter to continue...")
    input()
    
    try:
        # Run examples
        example_1_basic_ingestion()
        example_2_basic_query()
        example_3_metadata_filtering()
        example_4_multiple_queries()
        example_5_system_info()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("   Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()

