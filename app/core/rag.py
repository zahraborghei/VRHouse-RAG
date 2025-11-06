"""Core RAG functionality including text chunking and LLM integration"""
from typing import List, Dict, Any, Optional
import re
import google.generativeai as genai
from app.core.config import get_settings


class TextChunker:
    """
    Handles text chunking with overlap for context preservation.
    Uses fixed-size chunks with word-boundary awareness.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Clean the text
        text = self._clean_text(text)
        
        if len(text) <= self.chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        prev_start = -1
        
        while start < len(text):
            # Avoid infinite loop - if we haven't moved forward, break
            if start == prev_start:
                break
            prev_start = start
            
            # Calculate end position
            end = start + self.chunk_size
            
            # If not at the end, try to break at a word boundary
            if end < len(text):
                # Look for the last space before the end position
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position (with overlap)
            next_start = end - self.chunk_overlap
            
            # Ensure we're making progress
            if next_start <= start:
                start = end
            else:
                start = next_start
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing excessive whitespace and normalizing
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text


class LLMService:
    """
    Service for interacting with LLM (Google Gemini)
    
    """
    
    def __init__(self):
        """Initialize the LLM service"""
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        # self.model = genai.GenerativeModel('gemini-pro')
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        # System prompt template for RAG
        self.system_prompt = """You are a helpful AI assistant that answers questions based ONLY on the provided context.

CRITICAL RULES:
1. ONLY use information from the context provided below
2. If the answer is not in the context, clearly state "I cannot answer this question based on the provided context"
3. ALWAYS cite the source document ID for each piece of information you use
4. DO NOT use any external knowledge or make assumptions
5. Be concise and accurate in your responses

Context Sources:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the information from the context above
- Cite sources using [Source: doc_id] format
- If the context doesn't contain relevant information, explicitly say so
- Provide a clear, well-structured answer

Answer:"""
    
    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        """        
        Args:
            question
            context_chunks: List of retrieved context chunks with metadata
            
        Returns:
            Generated answer
        """
        # Format context with sources
        context_text = self._format_context(context_chunks)
        
        # Build the prompt
        prompt = self.system_prompt.format(
            context=context_text,
            question=question
        )
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Format context chunks into a readable string with source information."""
        formatted_parts = []
        
        for i, chunk in enumerate(context_chunks):
            doc_id = chunk.get('metadata', {}).get('doc_id', f'doc_{i}')
            source = chunk.get('metadata', {}).get('source', 'Unknown')
            text = chunk.get('text', '')
            
            formatted_parts.append(
                f"[Source ID: {doc_id}, File: {source}]\n{text}\n"
            )
        
        return "\n---\n".join(formatted_parts)


# Singleton instances
_text_chunker: Optional[TextChunker] = None
_llm_service: Optional[LLMService] = None


def get_text_chunker() -> TextChunker:
    """Get or create the text chunker singleton"""
    global _text_chunker
    if _text_chunker is None:
        settings = get_settings()
        _text_chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    return _text_chunker


def get_llm_service() -> LLMService:
    """Get or create the LLM service singleton"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

