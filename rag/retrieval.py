"""
RAG Retrieval System - Enhanced Version
======================================

This module handles the retrieval part of RAG with improved extraction,
chunking, and search capabilities for better keyword matching.
"""

import os
import json
import re
import math
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import pickle


class SimpleVectorStore:
    """Enhanced vector store with improved tokenization and search."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.vocabulary = {}
        self.idf_scores = {}
        self.metadata = []
    
    def add_document(self, text: str, metadata: Dict = None):
        """Add a document to the vector store."""
        self.documents.append(text)
        self.metadata.append(metadata or {})
    
    def build_index(self):
        """Build the TF-IDF index for all documents."""
        print("🔨 Building TF-IDF index...")
        
        # Build vocabulary from all documents
        word_counts = Counter()
        document_word_sets = []
        
        for doc in self.documents:
            words = self._tokenize(doc)
            document_word_sets.append(set(words))
            word_counts.update(words)
        
        # Create vocabulary (be more inclusive for small datasets)
        min_frequency = 1 if len(self.documents) <= 5 else 2
        
        # Important Python keywords that should always be included
        python_keywords = {
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally',
            'import', 'from', 'as', 'return', 'yield', 'break', 'continue', 'pass',
            'true', 'false', 'none', 'and', 'or', 'not', 'in', 'is', 'lambda',
            'global', 'nonlocal', 'with', 'assert', 'del', 'raise', 'async', 'await'
        }
        
        self.vocabulary = {}
        for idx, (word, count) in enumerate(word_counts.items()):
            if (count >= min_frequency or 
                word.lower() in python_keywords or
                len(word) > 4 or 
                '_' in word):
                self.vocabulary[word] = idx
        
        print(f"📚 Vocabulary size: {len(self.vocabulary)} words")
        
        # Calculate IDF scores
        num_docs = len(self.documents)
        for word in self.vocabulary:
            doc_freq = sum(1 for word_set in document_word_sets if word in word_set)
            self.idf_scores[word] = math.log(num_docs / (doc_freq + 1))
        
        # Create TF-IDF vectors
        self.embeddings = []
        for doc in self.documents:
            vector = self._create_tfidf_vector(doc)
            self.embeddings.append(vector)
        
        print(f"✅ Index built with {len(self.embeddings)} document vectors")
    
    def _tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization for better keyword recognition."""
        original_text = text
        text = text.lower()
        
        all_tokens = []
        
        # Python keywords
        python_keywords = {
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally',
            'import', 'from', 'as', 'return', 'yield', 'break', 'continue', 'pass',
            'true', 'false', 'none', 'and', 'or', 'not', 'in', 'is', 'lambda',
            'global', 'nonlocal', 'with', 'assert', 'del', 'raise', 'async', 'await'
        }
        
        # Find all Python keywords
        for keyword in python_keywords:
            if keyword in text:
                all_tokens.append(keyword)
        
        # Find compound keywords (or_eq, and_eq, etc.)
        compound_keywords = re.findall(r'\\b\\w+(?:_\\w+)+\\b', text)
        all_tokens.extend(compound_keywords)
        
        # Find operators
        operators = re.findall(r'[|&^!<>=]=?|\\+\\+|--', text)
        all_tokens.extend(operators)
        
        # Find technical terms
        technical_terms = re.findall(r'\\b[a-zA-Z]\\w*\\b', text)
        all_tokens.extend(technical_terms)
        
        # Find keyword:description patterns
        keyword_desc_pairs = re.findall(r'([a-zA-Z]\\w*)\\s*:', text)
        all_tokens.extend(keyword_desc_pairs)
        
        # Remove duplicates
        seen = set()
        unique_words = []
        for word in all_tokens:
            word = word.strip().lower()
            if word and word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        # Filter stop words but keep programming terms
        stop_words = {
            'the', 'a', 'an', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        filtered_words = []
        for word in unique_words:
            if (word in python_keywords or
                '_' in word or
                any(op in word for op in ['=', '|', '&', '^', '!', '<', '>', '+', '-']) or
                (len(word) > 1 and word not in stop_words)):
                filtered_words.append(word)
        
        return filtered_words
    
    def _create_tfidf_vector(self, text: str) -> Dict[int, float]:
        """Create TF-IDF vector for document."""
        words = self._tokenize(text)
        word_counts = Counter(words)
        total_words = len(words)
        
        vector = {}
        
        for word, count in word_counts.items():
            if word in self.vocabulary:
                word_idx = self.vocabulary[word]
                tf = count / total_words
                tfidf = tf * self.idf_scores[word]
                vector[word_idx] = tfidf
        
        return vector
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Enhanced search with multiple matching strategies."""
        if not self.documents:
            return []
        
        # Strategy 1: Direct keyword matching for short queries
        if len(query.split()) <= 3:
            direct_matches = self._direct_keyword_search(query)
            if direct_matches:
                return direct_matches[:top_k]
        
        # Strategy 2: TF-IDF similarity search
        query_vector = self._create_tfidf_vector(query)
        similarities = []
        
        for i, doc_vector in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_vector, doc_vector)
            similarities.append((i, similarity))
        
        # Sort and filter
        similarities.sort(key=lambda x: x[1], reverse=True)
        min_threshold = 0.05 if len(query.split()) <= 2 else 0.1
        filtered_results = [(i, score) for i, score in similarities if score > min_threshold]
        
        # Convert to expected format
        results = []
        for i, score in filtered_results[:top_k]:
            results.append((i, score, self.documents[i]))
        
        return results
    
    def _direct_keyword_search(self, query: str) -> List[Tuple[int, float, str]]:
        """Direct keyword matching for precise searches."""
        query_terms = set(self._tokenize(query))
        matches = []
        
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            doc_terms = set(self._tokenize(doc))
            
            # Calculate matches
            exact_matches = len(query_terms.intersection(doc_terms))
            partial_matches = 0
            
            for query_term in query_terms:
                if any(query_term in doc_term for doc_term in doc_terms):
                    partial_matches += 0.5
                elif query_term in doc_lower:
                    partial_matches += 0.3
            
            # Score based on matches
            if exact_matches > 0 or partial_matches > 0:
                base_score = (exact_matches * 2 + partial_matches) / len(query_terms)
                
                # Get metadata for this document
                metadata = self.metadata[i] if hasattr(self, 'metadata') and i < len(self.metadata) else {}
                filename = metadata.get('filename', '').lower()
                source = metadata.get('source', '').lower()
                
                # Boost tutorial and educational content
                if 'tutorial' in filename or 'tutorial' in source:
                    base_score *= 3.0  # Strong boost for tutorial content
                elif 'introduction' in filename or 'introduction' in source:
                    base_score *= 2.5  # Boost for introduction content
                elif 'reference' in filename or 'reference' in source:
                    base_score *= 2.0  # Boost for reference content
                
                # Boost if the text contains explanatory phrases
                explanatory_patterns = [
                    'is a', 'means', 'used to', 'keyword', 'statement',
                    'expression', 'operator', 'built-in', 'represents'
                ]
                explanation_boost = 1.0
                for pattern in explanatory_patterns:
                    if pattern in doc_lower:
                        explanation_boost += 0.2
                
                final_score = base_score * explanation_boost
                matches.append((i, final_score, doc))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def _cosine_similarity(self, vec1: Dict[int, float], vec2: Dict[int, float]) -> float:
        """Calculate cosine similarity between sparse vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        # Calculate dot product
        dot_product = 0.0
        for idx in vec1:
            if idx in vec2:
                dot_product += vec1[idx] * vec2[idx]
        
        # Calculate norms
        norm1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        norm2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


def debug_search_system(query: str = "what is True") -> None:
    """Debug the search system to identify issues."""
    print(f"🔍 Debugging search system with query: '{query}'")
    
    try:
        with open('rag/vector_index.pkl', 'rb') as f:
            vector_store = pickle.load(f)
        print(f"✅ Loaded vector store with {len(vector_store.documents)} documents")
    except FileNotFoundError:
        print("❌ Vector store not found. Please build it first.")
        return
    
    # Debug tokenization
    query_tokens = vector_store._tokenize(query)
    print(f"🔤 Query tokens: {query_tokens}")
    
    # Check vocabulary
    matching_vocab = [word for word in query_tokens if word in vector_store.vocabulary]
    print(f"📚 Query tokens in vocabulary: {matching_vocab}")
    
    # Check documents for query terms
    print("\\n📄 Checking documents for query terms:")
    for i, doc in enumerate(vector_store.documents[:3]):
        doc_tokens = vector_store._tokenize(doc)
        matches = [token for token in query_tokens if token in doc_tokens]
        doc_preview = doc[:100].replace('\\n', ' ')
        print(f"   Doc {i+1}: {len(matches)} matches {matches} | Preview: {doc_preview}...")
    
    # Test search
    print(f"\\n🔍 Testing search for: '{query}'")
    results = vector_store.search(query, top_k=3)
    
    if results:
        print(f"✅ Found {len(results)} results:")
        for i, (doc_idx, score, doc_text) in enumerate(results):
            preview = doc_text[:100].replace('\\n', ' ')
            print(f"   {i+1}. Score: {score:.3f} | Doc {doc_idx}: {preview}...")
    else:
        print("❌ No results found")
    
    # Check for specific keywords in documents
    test_keywords = ['true', 'false', 'while', 'for', 'if', 'or', 'and']
    print(f"\\n🔍 Checking for specific keywords in documents:")
    for keyword in test_keywords:
        found_in_docs = []
        for i, doc in enumerate(vector_store.documents):
            if keyword in doc.lower():
                found_in_docs.append(i)
        if found_in_docs:
            print(f"   '{keyword}': Found in docs {found_in_docs}")
        else:
            print(f"   '{keyword}': Not found in any document")


# Initialize global vector store
vector_store = SimpleVectorStore()


def build_vector_database(force_rebuild: bool = False) -> bool:
    """Build vector database from ingested document chunks.
    
    Args:
        force_rebuild (bool): Whether to force rebuild even if exists
        
    Returns:
        bool: True if successful, False otherwise
    """
    global vector_store
    
    print("🚀 Building vector database from chunks...")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found")
        return False
    
    print(f"📂 Loading chunks from '{data_dir}' directory...")
    
    # Reset vector store
    vector_store = SimpleVectorStore()
    
    # Load all chunk files
    chunk_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not chunk_files:
        print(f"❌ No .txt files found in '{data_dir}'")
        return False
    
    sources = set()
    
    for filename in chunk_files:
        filepath = os.path.join(data_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract source URL
            lines = content.split('\\n')
            source_url = "unknown"
            if len(lines) > 0 and lines[0].startswith("Source:"):
                source_url = lines[0].replace("Source:", "").strip()
                sources.add(source_url)
            
            # Add document with metadata
            metadata = {
                'filename': filename,
                'source': source_url,
                'filepath': filepath
            }
            vector_store.add_document(content, metadata)
            
        except Exception as e:
            print(f"❌ Error reading {filepath}: {str(e)}")
            continue
    
    print(f"✅ Loaded {len(vector_store.documents)} chunks")
    
    if not vector_store.documents:
        print("❌ No documents loaded")
        return False
    
    # Build the index
    vector_store.build_index()
    
    # Save vector index
    os.makedirs("rag", exist_ok=True)
    index_file = "rag/vector_index.pkl"
    
    try:
        with open(index_file, 'wb') as f:
            pickle.dump(vector_store, f)
        
        print(f"💾 Vector index saved to: {index_file}")
        print(f"🎉 Vector database built successfully!")
        print(f"   🧠 Documents: {len(vector_store.documents)}")
        print(f"   📚 Vocabulary: {len(vector_store.vocabulary)} words")  
        print(f"   🌐 Sources: {len(sources)} unique URLs")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving vector index: {str(e)}")
        return False


def retrieve_relevant_chunks(question: str, num_chunks: int = 3) -> List[Dict[str, any]]:
    """Retrieve the most relevant chunks for a question."""
    print(f"🔍 Retrieving relevant chunks for question: '{question[:50]}...'")
    
    global vector_store
    
    # Load vector store if not already loaded
    if not hasattr(vector_store, 'documents') or not vector_store.documents:
        try:
            with open('rag/vector_index.pkl', 'rb') as f:
                vector_store = pickle.load(f)
        except FileNotFoundError:
            print("❌ Vector database not found. Building it now...")
            if not build_vector_database():
                return []
    
    # Search for relevant documents
    if 'what is' in question.lower():
        search_k = min(num_chunks * 2, len(vector_store.documents))
    else:
        search_k = num_chunks
    
    results = vector_store.search(question, top_k=search_k)
    
    if not results:
        print("❌ No relevant chunks found")
        return []
    
    print(f"✅ Retrieved {len(results)} relevant chunks")
    
    # Format results
    formatted_chunks = []
    for i, (doc_idx, similarity, doc_text) in enumerate(results):
        metadata = vector_store.metadata[doc_idx] if hasattr(vector_store, 'metadata') and doc_idx < len(vector_store.metadata) else {}
        
        chunk_info = {
            'text': doc_text,
            'similarity_score': similarity,
            'source': metadata.get('source', 'unknown'),
            'filename': metadata.get('filename', 'unknown'),
            'chunk_id': metadata.get('filename', f'chunk_{doc_idx}')  # Add chunk_id
        }
        formatted_chunks.append(chunk_info)
        
        print(f"   {i+1}. {chunk_info['filename']} (relevance: {similarity*100:.1f}%)")
    
    return formatted_chunks[:num_chunks]


def search_documents(query: str, top_k: int = 5) -> Dict[str, any]:
    """Search documents using the vector database."""
    results = retrieve_relevant_chunks(query, top_k)
    
    return {
        'query': query,
        'total_results': len(results),
        'chunks': results
    }


def get_database_stats() -> Dict[str, any]:
    """Get statistics about the vector database."""
    global vector_store
    
    try:
        # Load vector store if needed
        if not hasattr(vector_store, 'documents') or not vector_store.documents:
            with open('rag/vector_index.pkl', 'rb') as f:
                vector_store = pickle.load(f)
        
        # Count unique sources
        sources = set()
        for metadata in vector_store.metadata:
            sources.add(metadata.get('source', 'Unknown'))
        
        stats = {
            'total_documents': len(vector_store.documents),
            'vocabulary_size': len(vector_store.vocabulary),
            'unique_sources': len(sources),
            'sources': list(sources),
            'average_doc_length': sum(len(doc.split()) for doc in vector_store.documents) / len(vector_store.documents) if vector_store.documents else 0
        }
        
        return stats
        
    except Exception as e:
        return {'error': f'Failed to get database stats: {str(e)}'}


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing RAG Retrieval System")
    
    if build_vector_database():
        test_queries = [
            "what is True",
            "how to define a function", 
            "python keywords",
            "while loop"
        ]
        
        for query in test_queries:
            print(f"\\n❓ Testing query: '{query}'")
            results = search_documents(query, top_k=3)
            print(f"Results: {results['total_results']}")
            
            for i, chunk in enumerate(results['chunks'], 1):
                print(f"\\nResult {i}:")
                print(f"  Source: {chunk['source']}")
                print(f"  Relevance: {chunk['similarity_score']:.2f}")
                print(f"  Preview: {chunk['text'][:100]}...")
    
    else:
        print("❌ Failed to build vector database")