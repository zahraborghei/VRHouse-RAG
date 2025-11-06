"""
Simple API test script to verify the RAG system is working
"""

import requests
import sys


def test_api():
    """Test the RAG System API"""
    
    base_url = "http://localhost:8000/api/v1"
    
    print("🧪 Testing RAG System API\n")
    
    # Test 1: Health Check
    print("1️⃣  Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("   ✅ Health check passed")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Could not connect to API: {e}")
        print("   💡 Make sure the server is running: python -m uvicorn app.main:app --reload")
        return False
    
    # Test 2: System Info
    print("\n2️⃣  Testing info endpoint...")
    try:
        response = requests.get(f"{base_url}/info")
        if response.status_code == 200:
            info = response.json()
            print("   ✅ Info endpoint working")
            print(f"      Model: {info['embedding_model']['model_name']}")
            print(f"      Documents in DB: {info['vector_store']['document_count']}")
        else:
            print(f"   ❌ Info endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Ingestion
    print("\n3️⃣  Testing document ingestion...")
    try:
        test_doc = {
            "text": """
            The Quick Brown Fox Test Document.
            This is a simple test document to verify the ingestion pipeline.
            It contains information about testing, validation, and verification.
            The RAG system should be able to chunk this text, generate embeddings,
            and store it in the vector database successfully.
            """,
            "metadata": {
                "topic": "testing",
                "type": "test_document",
                "date": "2025-11-03"
            }
        }
        
        response = requests.post(f"{base_url}/ingest", json=test_doc)
        if response.status_code == 201:
            result = response.json()
            print("   ✅ Ingestion successful")
            print(f"      Doc ID: {result['doc_id']}")
            print(f"      Chunks: {result['num_chunks']}")
            print(f"      Tokens: {result['num_tokens']}")
            doc_id = result['doc_id']
        else:
            print(f"   ❌ Ingestion failed: {response.status_code}")
            print(f"      {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 4: Query
    print("\n4️⃣  Testing query endpoint...")
    try:
        query = {
            "question": "What is this document about?",
            "k": 3,
            "include_sources": True
        }
        
        response = requests.post(f"{base_url}/query", json=query)
        if response.status_code == 200:
            result = response.json()
            print("   ✅ Query successful")
            print(f"      Question: {result['question']}")
            print(f"      Chunks retrieved: {result['num_chunks_retrieved']}")
            print(f"      Answer preview: {result['answer'][:100]}...")
            
            if result['context_chunks']:
                top_chunk = result['context_chunks'][0]
                print(f"      Top similarity: {top_chunk['similarity_score']:.3f}")
        else:
            print(f"   ❌ Query failed: {response.status_code}")
            print(f"      {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 5: Metadata Filtering
    print("\n5️⃣  Testing metadata filtering...")
    try:
        query = {
            "question": "Tell me about testing",
            "k": 2,
            "metadata_filter": {"topic": "testing"}
        }
        
        response = requests.post(f"{base_url}/query", json=query)
        if response.status_code == 200:
            result = response.json()
            print("   ✅ Metadata filtering works")
            print(f"      Filtered chunks: {result['num_chunks_retrieved']}")
        else:
            print(f"   ❌ Filtering failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("✅ All tests passed! RAG System is working correctly.")
    print("="*60)
    print("\n💡 Next steps:")
    print("   - Try the example_usage.py script for more examples")
    print("   - Visit http://localhost:8000/docs for API documentation")
    print("   - Start ingesting your own documents!")
    
    return True


if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1)

