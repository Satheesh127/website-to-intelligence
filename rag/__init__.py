"""
RAG Module
==========

This module handles Retrieval-Augmented Generation (RAG) functionality.

Modern Implementation:
- ChromaDB for vector storage
- SentenceTransformers for embeddings  
- LLM integration for answer generation

Legacy Implementation (backup):
- Custom TF-IDF vector store
- Heuristic answer generation
"""

# Modern RAG components (recommended)
try:
    from .chroma_retrieval import (
        initialize_vector_store, 
        build_vector_database, 
        retrieve_relevant_chunks,
        search_documents,
        get_database_stats,
        ChromaVectorStore
    )
    from .llm_answering import (
        answer_question,
        initialize_answer_generator,
        LLMAnswerGenerator
    )
    MODERN_RAG_AVAILABLE = True
except ImportError:
    MODERN_RAG_AVAILABLE = False

# Legacy RAG components (fallback)
try:
    from .retrieval import (
        retrieve_relevant_chunks as legacy_retrieve,
        build_vector_database as legacy_build_db,
        search_documents as legacy_search
    )
    from .answering import answer_question as legacy_answer
    LEGACY_RAG_AVAILABLE = True
except ImportError:
    LEGACY_RAG_AVAILABLE = False

# Provide fallback imports
if not MODERN_RAG_AVAILABLE:
    if LEGACY_RAG_AVAILABLE:
        retrieve_relevant_chunks = legacy_retrieve
        build_vector_database = legacy_build_db
        search_documents = legacy_search
        answer_question = legacy_answer
    else:
        raise ImportError("No RAG implementation available")