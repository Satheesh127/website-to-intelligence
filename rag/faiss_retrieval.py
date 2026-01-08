"""
FAISS Vector Retrieval System
============================

This module implements a vector retrieval system using FAISS for efficient
similarity search. Replaces ChromaDB to solve file locking issues.

Features:
- FAISS for fast similarity search
- SentenceTransformers for embeddings
- Simple pickle file storage (no SQLite)
- No file locking issues
- Easy reset functionality
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import uuid

class FAISSVectorStore:
    """FAISS-based vector store for document retrieval"""
    
    def __init__(self, persist_directory: str = "vector_store", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize FAISS vector store
        
        Args:
            persist_directory: Directory to store FAISS index and metadata
            model_name: SentenceTransformer model for embeddings
        """
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.model = None
        self.index = None
        self.documents = []
        self.metadata = []
        self.embeddings = None
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # File paths
        self.index_path = os.path.join(persist_directory, "faiss_index.bin")
        self.metadata_path = os.path.join(persist_directory, "metadata.pkl")
        self.documents_path = os.path.join(persist_directory, "documents.pkl")
        
        # Load existing index if available
        self._load_model()
        self._load_index()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            print(f"[INFO] Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"[SUCCESS] Embedding model loaded")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            if (os.path.exists(self.index_path) and 
                os.path.exists(self.metadata_path) and 
                os.path.exists(self.documents_path)):
                
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                # Load documents
                with open(self.documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
                
                print(f"[SUCCESS] Loaded existing index with {len(self.documents)} documents")
            else:
                print("[INFO] No existing index found, will create new one")
                self.index = None
                self.documents = []
                self.metadata = []
                
        except Exception as e:
            print(f"[WARNING] Failed to load existing index: {e}")
            self.index = None
            self.documents = []
            self.metadata = []
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> bool:
        """
        Add documents to the vector store
        
        Args:
            texts: List of text documents
            metadatas: List of metadata dictionaries
            
        Returns:
            bool: Success status
        """
        try:
            if not texts:
                return False
            
            if metadatas is None:
                metadatas = [{"id": str(uuid.uuid4())} for _ in texts]
            
            print(f"[INFO] Adding {len(texts)} documents to vector store...")
            
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            embeddings = embeddings.astype('float32')  # FAISS requires float32
            
            # Create or update FAISS index
            if self.index is None:
                # Create new index
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                
                # Add embeddings
                self.index.add(embeddings)
                
                # Store documents and metadata
                self.documents = texts.copy()
                self.metadata = metadatas.copy()
            else:
                # Add to existing index
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings)
                
                # Append documents and metadata
                self.documents.extend(texts)
                self.metadata.extend(metadatas)
            
            # Save to disk
            self._save_index()
            
            print(f"[SUCCESS] Added {len(texts)} documents. Total: {len(self.documents)}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to add documents: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of document results with scores and metadata
        """
        try:
            if self.index is None or len(self.documents) == 0:
                print("[WARNING] No documents in index")
                return []
            
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(k, len(self.documents))  # Don't search for more than we have
            scores, indices = self.index.search(query_embedding, k)
            
            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):  # Validate index
                    result = {
                        'text': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'similarity_score': float(score),
                        'rank': i + 1
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            # Save documents
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            print(f"[SUCCESS] Index saved to {self.persist_directory}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.index.d if self.index else 0,
            'model_name': self.model_name,
            'storage_path': self.persist_directory
        }
    
    def delete_all(self) -> bool:
        """Delete all documents and reset the index"""
        try:
            # Clear in-memory data
            self.index = None
            self.documents = []
            self.metadata = []
            
            # Delete files
            files_to_delete = [self.index_path, self.metadata_path, self.documents_path]
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"[SUCCESS] Deleted {file_path}")
            
            print("[SUCCESS] Vector store reset complete")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to delete vector store: {e}")
            return False


# Global vector store instance
vector_store = None

def initialize_vector_store(force_rebuild: bool = False) -> bool:
    """
    Initialize the global vector store
    
    Args:
        force_rebuild: Whether to rebuild the entire database
        
    Returns:
        bool: True if successful, False otherwise
    """
    global vector_store
    
    try:
        print("[INFO] Initializing FAISS vector store...")
        
        # Ensure vector_store directory exists
        vector_dir = os.getenv("VECTOR_STORE_DIRECTORY", "vector_store")
        os.makedirs(vector_dir, exist_ok=True)
        
        vector_store = FAISSVectorStore(persist_directory=vector_dir)
        
        # Check if we need to build/rebuild
        stats = vector_store.get_stats()
        if stats['total_documents'] == 0 or force_rebuild:
            print("[INFO] Building vector database from documents...")
            return build_vector_database(force_rebuild)
        else:
            print(f"[SUCCESS] Vector store ready with {stats['total_documents']} documents")
            return True
            
    except Exception as e:
        print(f"[ERROR] Failed to initialize vector store: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def build_vector_database(force_rebuild: bool = False) -> bool:
    """
    Build vector database from document chunks
    
    Args:
        force_rebuild: Whether to force rebuild
        
    Returns:
        bool: True if successful, False otherwise
    """
    global vector_store
    
    try:
        if vector_store is None:
            print("[ERROR] Vector store not initialized")
            return False
        
        if force_rebuild:
            vector_store.delete_all()
            vector_store = FAISSVectorStore(persist_directory=vector_store.persist_directory)
        
        # Read document chunks from data directory
        data_dir = "data"
        if not os.path.exists(data_dir):
            print(f"[WARNING] Data directory {data_dir} does not exist")
            return False
        
        chunks = []
        metadatas = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Extract source URL from content
                        lines = content.split('\n')
                        source_url = "unknown"
                        for line in lines:
                            if line.startswith("Source: "):
                                source_url = line.replace("Source: ", "").strip()
                                break
                        
                        # Get the main text (skip header)
                        text_start = content.find("-" * 50)
                        if text_start != -1:
                            text = content[text_start + 52:].strip()
                        else:
                            text = content
                        
                        if len(text) > 50:  # Only add substantial chunks
                            chunks.append(text)
                            metadatas.append({
                                'source': source_url,
                                'filename': filename,
                                'chunk_id': len(chunks)
                            })
                
                except Exception as e:
                    print(f"[WARNING] Failed to read {filename}: {e}")
                    continue
        
        if not chunks:
            print("[WARNING] No document chunks found")
            return False
        
        # Add documents to vector store
        success = vector_store.add_documents(chunks, metadatas)
        
        if success:
            stats = vector_store.get_stats()
            print(f"[SUCCESS] Built vector database with {stats['total_documents']} documents")
            return True
        else:
            print("[ERROR] Failed to build vector database")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to build vector database: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def retrieve_relevant_chunks(question: str, num_chunks: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks for a question
    
    Args:
        question: User's question
        num_chunks: Number of chunks to retrieve
        
    Returns:
        List of relevant chunks
    """
    global vector_store
    
    try:
        if vector_store is None:
            print("[ERROR] Vector store not initialized")
            return []
        
        results = vector_store.search(question, k=num_chunks)
        
        # Format for compatibility with existing code
        formatted_results = []
        for result in results:
            formatted_results.append({
                'text': result['text'],
                'source': result['metadata'].get('source', 'unknown'),
                'similarity_score': result['similarity_score'],
                'metadata': result['metadata']
            })
        
        return formatted_results
        
    except Exception as e:
        print(f"[ERROR] Failed to retrieve chunks: {e}")
        return []

def get_database_stats() -> Dict[str, Any]:
    """Get database statistics"""
    global vector_store
    
    if vector_store is None:
        return {
            'total_documents': 0,
            'vocabulary_size': 0,
            'unique_sources': 0
        }
    
    stats = vector_store.get_stats()
    
    # Count unique sources
    unique_sources = set()
    for metadata in vector_store.metadata:
        source = metadata.get('source', 'unknown')
        if source != 'unknown':
            unique_sources.add(source)
    
    return {
        'total_documents': stats['total_documents'],
        'vocabulary_size': len(set(' '.join(vector_store.documents).split())),
        'unique_sources': len(unique_sources)
    }

def reset_vector_database() -> bool:
    """Reset the vector database completely"""
    global vector_store
    
    try:
        if vector_store is not None:
            success = vector_store.delete_all()
            vector_store = None
            return success
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to reset vector database: {e}")
        return False