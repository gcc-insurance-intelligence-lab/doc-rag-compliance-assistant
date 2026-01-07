"""
RAG Engine Utilities
Provides core retrieval and generation functions for the compliance assistant.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter


class DocumentStore:
    """Simple document storage and retrieval."""
    
    def __init__(self):
        self.documents = {}
        self.metadata = {}
    
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None):
        """
        Add a document to the store.
        
        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata (title, date, category, etc.)
        """
        self.documents[doc_id] = content
        self.metadata[doc_id] = metadata or {}
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Retrieve a document by ID."""
        return self.documents.get(doc_id)
    
    def get_all_documents(self) -> Dict[str, str]:
        """Get all documents."""
        return self.documents.copy()
    
    def get_metadata(self, doc_id: str) -> Dict:
        """Get metadata for a document."""
        return self.metadata.get(doc_id, {})
    
    def remove_document(self, doc_id: str):
        """Remove a document from the store."""
        if doc_id in self.documents:
            del self.documents[doc_id]
        if doc_id in self.metadata:
            del self.metadata[doc_id]
    
    def clear(self):
        """Clear all documents."""
        self.documents.clear()
        self.metadata.clear()
    
    def __len__(self):
        return len(self.documents)


class KeywordRetriever:
    """Keyword-based document retrieval."""
    
    def __init__(self, document_store: DocumentStore, min_keyword_length: int = 3):
        """
        Initialize retriever.
        
        Args:
            document_store: DocumentStore instance
            min_keyword_length: Minimum length for keywords to consider
        """
        self.store = document_store
        self.min_keyword_length = min_keyword_length
        self.stopwords = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'from', 'that', 'this',
            'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must',
            'can', 'what', 'when', 'where', 'who', 'how', 'why'
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Convert to lowercase
        text_lower = text.lower()
        
        # Extract words (alphanumeric only)
        words = re.findall(r'\b[a-z0-9]+\b', text_lower)
        
        # Filter by length and stopwords
        keywords = [
            word for word in words 
            if len(word) >= self.min_keyword_length and word not in self.stopwords
        ]
        
        return keywords
    
    def calculate_score(self, query_keywords: List[str], doc_content: str, 
                       doc_id: str) -> float:
        """
        Calculate relevance score for a document.
        
        Args:
            query_keywords: Keywords from query
            doc_content: Document content
            doc_id: Document identifier
            
        Returns:
            Relevance score
        """
        doc_lower = doc_content.lower()
        score = 0.0
        
        # Count keyword matches in content
        for keyword in query_keywords:
            if keyword in doc_lower:
                # Count occurrences
                count = doc_lower.count(keyword)
                score += count
        
        # Boost if keywords appear in document ID/title
        doc_id_lower = doc_id.lower()
        for keyword in query_keywords:
            if keyword in doc_id_lower:
                score += 5.0
        
        return score
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Retrieve relevant documents.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of (doc_id, content, score) tuples
        """
        # Extract query keywords
        query_keywords = self.extract_keywords(query)
        
        if not query_keywords:
            return []
        
        # Score all documents
        scores = []
        for doc_id, content in self.store.get_all_documents().items():
            score = self.calculate_score(query_keywords, content, doc_id)
            scores.append((doc_id, content, score))
        
        # Sort by score (descending) and return top_k
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]


class BM25Retriever:
    """BM25 ranking algorithm for document retrieval."""
    
    def __init__(self, document_store: DocumentStore, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.
        
        Args:
            document_store: DocumentStore instance
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.store = document_store
        self.k1 = k1
        self.b = b
        self.keyword_retriever = KeywordRetriever(document_store)
        
        # Precompute document statistics
        self._compute_stats()
    
    def _compute_stats(self):
        """Compute document statistics for BM25."""
        docs = self.store.get_all_documents()
        self.num_docs = len(docs)
        
        # Document lengths
        self.doc_lengths = {}
        total_length = 0
        
        for doc_id, content in docs.items():
            keywords = self.keyword_retriever.extract_keywords(content)
            length = len(keywords)
            self.doc_lengths[doc_id] = length
            total_length += length
        
        # Average document length
        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0
        
        # Document frequency (DF) for each term
        self.doc_freq = Counter()
        for doc_id, content in docs.items():
            keywords = set(self.keyword_retriever.extract_keywords(content))
            for keyword in keywords:
                self.doc_freq[keyword] += 1
    
    def calculate_idf(self, term: str) -> float:
        """Calculate inverse document frequency."""
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return np.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
    
    def calculate_score(self, query_keywords: List[str], doc_id: str, 
                       doc_content: str) -> float:
        """Calculate BM25 score."""
        score = 0.0
        doc_keywords = self.keyword_retriever.extract_keywords(doc_content)
        doc_length = self.doc_lengths.get(doc_id, 0)
        
        # Term frequencies in document
        term_freq = Counter(doc_keywords)
        
        for term in query_keywords:
            if term not in term_freq:
                continue
            
            # IDF
            idf = self.calculate_idf(term)
            
            # Term frequency
            tf = term_freq[term]
            
            # Length normalization
            norm = 1 - self.b + self.b * (doc_length / self.avg_doc_length)
            
            # BM25 formula
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
        
        return score
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Retrieve documents using BM25."""
        query_keywords = self.keyword_retriever.extract_keywords(query)
        
        if not query_keywords:
            return []
        
        scores = []
        for doc_id, content in self.store.get_all_documents().items():
            score = self.calculate_score(query_keywords, doc_id, content)
            scores.append((doc_id, content, score))
        
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]


class AnswerGenerator:
    """Generate answers from retrieved documents."""
    
    def __init__(self, max_context_length: int = 500):
        """
        Initialize generator.
        
        Args:
            max_context_length: Maximum characters to use from each document
        """
        self.max_context_length = max_context_length
    
    def extract_relevant_section(self, content: str, query: str) -> str:
        """
        Extract most relevant section from document.
        
        Args:
            content: Document content
            query: User query
            
        Returns:
            Relevant section
        """
        # Simple approach: return first N characters
        # Production would use more sophisticated extraction
        if len(content) <= self.max_context_length:
            return content
        
        return content[:self.max_context_length] + "..."
    
    def generate(self, query: str, retrieved_docs: List[Tuple[str, str, float]]) -> str:
        """
        Generate answer from retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of (doc_id, content, score) tuples
            
        Returns:
            Generated answer
        """
        if not retrieved_docs or retrieved_docs[0][2] == 0:
            return "No relevant information found."
        
        answer_parts = []
        
        for doc_id, content, score in retrieved_docs:
            if score > 0:
                section = self.extract_relevant_section(content, query)
                answer_parts.append(f"**From: {doc_id}**\n\n{section}\n")
        
        return "\n---\n\n".join(answer_parts)


class RAGPipeline:
    """Complete RAG pipeline."""
    
    def __init__(self, document_store: DocumentStore, 
                 retriever_type: str = "keyword"):
        """
        Initialize RAG pipeline.
        
        Args:
            document_store: DocumentStore instance
            retriever_type: 'keyword' or 'bm25'
        """
        self.store = document_store
        
        if retriever_type == "bm25":
            self.retriever = BM25Retriever(document_store)
        else:
            self.retriever = KeywordRetriever(document_store)
        
        self.generator = AnswerGenerator()
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve relevant documents
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        
        # Generate answer
        answer = self.generator.generate(question, retrieved)
        
        # Format sources
        sources = [
            {
                'doc_id': doc_id,
                'score': score,
                'preview': content[:200] + "..." if len(content) > 200 else content
            }
            for doc_id, content, score in retrieved
        ]
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'num_sources': len(sources)
        }
