"""
Modern RAG Answer Generation with LLM Integration
=================================================

This module handles answer generation using Large Language Models (LLMs).
It integrates with OpenAI's API to generate grounded, contextual responses.

Features:
- OpenAI API integration
- Context-aware prompt engineering
- Response validation and grounding
- Fallback mechanisms
- Cost-efficient token management
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import openai
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Import our retrieval system
from rag.chroma_retrieval import retrieve_relevant_chunks


class LLMAnswerGenerator:
    """LLM-based answer generator for RAG system."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", max_tokens: int = 500, temperature: float = 0.3):
        """
        Initialize the LLM answer generator.
        
        Args:
            model (str): OpenAI model to use
            max_tokens (int): Maximum tokens for response
            temperature (float): Response creativity (0.0-1.0)
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Verify API key
        if not openai.api_key:
            print("⚠️ Warning: OPENAI_API_KEY not found in environment variables")
            print("   Create a .env file with: OPENAI_API_KEY=your_api_key_here")
    
    def create_context_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a context-aware prompt for the LLM.
        
        Args:
            question (str): User's question
            retrieved_chunks (List[Dict]): Retrieved document chunks
            
        Returns:
            str: Formatted prompt for LLM
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get('source', 'Unknown')
            text = chunk.get('text', '').strip()
            
            if text:
                context_parts.append(f"[Source {i}: {source}]\\n{text}\\n")
        
        context = "\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Create the prompt
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context. 

**IMPORTANT INSTRUCTIONS:**
1. Answer using ONLY the information provided in the context below
2. If the context doesn't contain enough information, say "The provided documentation doesn't contain sufficient information to answer this question."
3. Be specific and cite which source(s) you're using
4. Provide practical, actionable answers when possible
5. If multiple sources provide different information, mention the differences
6. Keep your answer concise but complete

**CONTEXT:**
{context}

**QUESTION:** {question}

**ANSWER:**"""
        
        return prompt
    
    def generate_answer(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer using the LLM.
        
        Args:
            question (str): User's question
            retrieved_chunks (List[Dict]): Retrieved document chunks
            
        Returns:
            Dict[str, Any]: Generated answer with metadata
        """
        if not openai.api_key:
            return {
                'answer': "LLM service is not configured. Please set OPENAI_API_KEY environment variable.",
                'confidence': 0.0,
                'sources': [],
                'method': 'error',
                'tokens_used': 0,
                'cost_estimate': 0.0
            }
        
        if not retrieved_chunks:
            return {
                'answer': "No relevant information found in the knowledge base.",
                'confidence': 0.0,
                'sources': [],
                'method': 'no_context',
                'tokens_used': 0,
                'cost_estimate': 0.0
            }
        
        try:
            # Create prompt
            prompt = self.create_context_prompt(question, retrieved_chunks)
            
            print(f"[AI] Generating answer using {self.model}...")
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise, helpful assistant that answers questions based strictly on provided documentation context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9
            )
            
            # Extract answer
            answer = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            
            # Estimate cost (approximate rates for GPT-3.5-turbo)
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost_estimate = (input_tokens * 0.0015 + output_tokens * 0.002) / 1000  # Per 1K tokens
            
            # Calculate confidence based on context relevance
            confidence = self._calculate_confidence(answer, retrieved_chunks)
            
            # Extract sources
            sources = [
                {
                    'source': chunk.get('source', 'Unknown'),
                    'filename': chunk.get('filename', 'Unknown'),
                    'similarity': chunk.get('similarity_score', 0.0)
                }
                for chunk in retrieved_chunks
            ]
            
            return {
                'answer': answer,
                'confidence': confidence,
                'sources': sources,
                'method': 'llm_generation',
                'tokens_used': tokens_used,
                'cost_estimate': cost_estimate,
                'model_used': self.model
            }
        
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'method': 'error',
                'tokens_used': 0,
                'cost_estimate': 0.0
            }
    
    def _calculate_confidence(self, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for the answer.
        
        Args:
            answer (str): Generated answer
            retrieved_chunks (List[Dict]): Source chunks
            
        Returns:
            float: Confidence score (0.0-1.0)
        """
        # Check for uncertainty indicators
        uncertainty_phrases = [
            "doesn't contain sufficient information",
            "not mentioned",
            "no information",
            "cannot determine",
            "unclear",
            "not specified"
        ]
        
        answer_lower = answer.lower()
        
        # If answer indicates uncertainty, low confidence
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            return 0.2
        
        # Base confidence on chunk similarity scores
        if retrieved_chunks:
            avg_similarity = sum(chunk.get('similarity_score', 0.0) for chunk in retrieved_chunks) / len(retrieved_chunks)
            
            # Adjust based on answer length and specificity
            if len(answer.split()) < 10:  # Very short answer
                confidence_modifier = 0.8
            elif len(answer.split()) > 100:  # Very detailed answer
                confidence_modifier = 1.1
            else:
                confidence_modifier = 1.0
            
            confidence = min(1.0, avg_similarity * confidence_modifier)
        else:
            confidence = 0.1
        
        return round(confidence, 2)


# Global answer generator instance
answer_generator: Optional[LLMAnswerGenerator] = None


def initialize_answer_generator(model: str = "gpt-3.5-turbo") -> bool:
    """
    Initialize the global answer generator.
    
    Args:
        model (str): LLM model to use
        
    Returns:
        bool: True if successful
    """
    global answer_generator
    
    try:
        print(f"[AI] Initializing LLM answer generator with {model}...")
        answer_generator = LLMAnswerGenerator(model=model)
        print("✅ Answer generator initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize answer generator: {str(e)}")
        return False


def answer_question(question: str, num_chunks: int = 5) -> Dict[str, Any]:
    """
    Main function to answer user questions using RAG with LLM.
    
    Args:
        question (str): User's question
        num_chunks (int): Number of document chunks to retrieve
        
    Returns:
        Dict[str, Any]: Complete answer with metadata
    """
    global answer_generator
    
    if answer_generator is None:
        if not initialize_answer_generator():
            # Fallback to simple keyword matching if LLM fails
            return fallback_answer(question, num_chunks)
    
    print(f"❓ Processing question: '{question}'")
    print("=" * 60)
    
    # Step 1: Retrieve relevant chunks
    print("🔍 Step 1: Retrieving relevant context...")
    retrieved_chunks = retrieve_relevant_chunks(question, num_chunks)
    
    if not retrieved_chunks:
        return {
            'answer': "No relevant information found in the knowledge base for your question.",
            'confidence': 0.0,
            'sources': [],
            'method': 'no_retrieval',
            'tokens_used': 0,
            'cost_estimate': 0.0
        }
    
    print(f"📚 Found {len(retrieved_chunks)} relevant chunks")
    
    # Step 2: Generate answer using LLM
    print("[AI] Step 2: Generating answer with LLM...")
    result = answer_generator.generate_answer(question, retrieved_chunks)
    
    # Step 3: Format final response
    print("📝 Step 3: Formatting response...")
    formatted_result = format_answer_response(result, question)
    
    print(f"✅ Answer generated successfully (Confidence: {result['confidence']:.2f})")
    return formatted_result


def fallback_answer(question: str, num_chunks: int = 5) -> Dict[str, Any]:
    """
    Fallback answer generation without LLM.
    
    Args:
        question (str): User's question
        num_chunks (int): Number of chunks to retrieve
        
    Returns:
        Dict[str, Any]: Answer using simple extraction
    """
    print("⚠️ Using fallback answer generation (no LLM)")
    
    retrieved_chunks = retrieve_relevant_chunks(question, num_chunks)
    
    if not retrieved_chunks:
        return {
            'answer': "The knowledge base does not contain information relevant to your question.",
            'confidence': 0.0,
            'sources': [],
            'method': 'fallback_no_context',
            'tokens_used': 0,
            'cost_estimate': 0.0
        }
    
    # Simple extraction from most relevant chunk
    best_chunk = retrieved_chunks[0]
    text = best_chunk['text']
    
    # Extract relevant sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
    
    # Find sentences with question keywords
    question_words = set(re.findall(r'\\b\\w+\\b', question.lower()))
    question_words.discard('what')
    question_words.discard('how')
    question_words.discard('when')
    question_words.discard('where')
    question_words.discard('why')
    
    relevant_sentences = []
    for sentence in sentences[:5]:  # Check first 5 sentences
        sentence_words = set(re.findall(r'\\b\\w+\\b', sentence.lower()))
        overlap = len(question_words & sentence_words)
        if overlap > 0:
            relevant_sentences.append((sentence, overlap))
    
    if relevant_sentences:
        # Sort by overlap and take best sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        answer = '. '.join([sent[0] for sent in relevant_sentences[:3]])
        confidence = min(0.7, best_chunk['similarity_score'])
    else:
        # Use first few sentences
        answer = '. '.join(sentences[:2])
        confidence = 0.3
    
    return {
        'answer': answer,
        'confidence': confidence,
        'sources': [
            {
                'source': best_chunk['source'],
                'filename': best_chunk['filename'],
                'similarity': best_chunk['similarity_score']
            }
        ],
        'method': 'fallback_extraction',
        'tokens_used': 0,
        'cost_estimate': 0.0
    }


def format_answer_response(result: Dict[str, Any], question: str) -> Dict[str, Any]:
    """
    Format the final answer response.
    
    Args:
        result (Dict): Raw answer result
        question (str): Original question
        
    Returns:
        Dict[str, Any]: Formatted response
    """
    # Add metadata
    result['question'] = question
    result['timestamp'] = datetime.now().isoformat()
    
    # Format sources for display
    if result.get('sources'):
        source_list = []
        for i, source in enumerate(result['sources'], 1):
            source_list.append(f"{i}. {source['source']} (similarity: {source['similarity']:.3f})")
        result['source_list'] = source_list
    
    return result


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing LLM Answer Generation")
    print("=" * 50)
    
    # Test questions
    test_questions = [
        "What is a Python function?",
        "How do I create a variable in Python?",
        "What are Python loops?",
        "How to handle errors in programming?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\\n🔍 Test {i}: {question}")
        print("-" * 40)
        
        result = answer_question(question)
        
        print(f"📝 Answer: {result['answer'][:200]}...")
        print(f"🎯 Confidence: {result['confidence']:.2f}")
        print(f"🔧 Method: {result['method']}")
        
        if result['tokens_used'] > 0:
            print(f"💰 Tokens used: {result['tokens_used']}")
            print(f"💲 Est. cost: ${result['cost_estimate']:.6f}")
        
        if result['sources']:
            print("📚 Sources:")
            for source in result.get('source_list', []):
                print(f"   {source}")