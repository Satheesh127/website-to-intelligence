"""
FREE Groq LLM Answering Module
==============================

This module provides FREE AI answering using Groq API.
Groq offers free access to powerful models like Llama!

Features:
- FREE API access (no costs!)
- Fast inference (much faster than OpenAI)
- Powerful Llama models
- Easy integration

Usage:
    from rag.groq_answering import generate_groq_answer
    
    chunks = [...] # Retrieved document chunks
    answer = generate_groq_answer("Your question", chunks)
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import tiktoken  # For accurate token counting

# Fallback token estimation if tiktoken fails
def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 0.75 words)"""
    try:
        # Try to use tiktoken for accurate counting
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except:
        # Fallback to word-based estimation
        words = len(text.split())
        return int(words * 0.75)

def remove_code_examples(text: str) -> str:
    """Remove code blocks and examples to save tokens"""
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '[Code example removed]', text)
    text = re.sub(r'`[^`]+`', '[Code removed]', text)
    
    # Remove common code patterns
    text = re.sub(r'#include.*?\n', '', text)
    text = re.sub(r'import.*?;?\n', '', text)
    text = re.sub(r'using namespace.*?;', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text.strip()

def extract_key_sentences(text: str, max_sentences: int = 3) -> str:
    """Extract most relevant sentences from text"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # Prioritize sentences with key terms
    key_terms = ['graph', 'vertex', 'edge', 'adjacency', 'matrix', 'list', 'algorithm', 'data structure']
    
    # Score sentences based on key terms
    scored_sentences = []
    for sentence in sentences:
        score = sum(1 for term in key_terms if term.lower() in sentence.lower())
        if score > 0 or len(scored_sentences) < max_sentences:
            scored_sentences.append((score, sentence))
    
    # Sort by score and take top sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    top_sentences = [sentence for _, sentence in scored_sentences[:max_sentences]]
    
    return '. '.join(top_sentences) + '.' if top_sentences else text[:200] + '...'

def smart_truncate(text: str, max_chars: int = 600) -> str:
    """Intelligently truncate text while preserving meaning"""
    if len(text) <= max_chars:
        return text
    
    # Try to cut at sentence boundaries
    sentences = text.split('.')
    result = ""
    
    for sentence in sentences:
        if len(result + sentence + '.') <= max_chars:
            result += sentence + '.'
        else:
            break
    
    if not result:
        # If no complete sentences fit, cut at word boundary
        words = text[:max_chars].split()
        result = ' '.join(words[:-1]) + '...'
    
    return result.strip()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
    # Note: Groq library imported successfully
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: Groq library not found. Install with: pip install groq")

# Load environment variables
load_dotenv()

# Token counting and text processing utilities
def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 0.75 words)"""
    try:
        # Try to use tiktoken for accurate counting (if available)
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except:
        # Fallback to word-based estimation
        words = len(text.split())
        return int(words * 0.75)

def remove_code_examples(text: str) -> str:
    """Remove code blocks and examples to save tokens"""
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '[Code example removed]', text)
    text = re.sub(r'`[^`]+`', '[Code removed]', text)
    
    # Remove common code patterns
    text = re.sub(r'#include.*?\n', '', text)
    text = re.sub(r'import.*?;?\n', '', text)
    text = re.sub(r'using namespace.*?;', '', text)
    text = re.sub(r'class\s+\w+.*?{.*?}', '[Class definition removed]', text, flags=re.DOTALL)
    text = re.sub(r'function.*?{.*?}', '[Function removed]', text, flags=re.DOTALL)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text.strip()

def extract_key_sentences(text: str, max_sentences: int = 3) -> str:
    """Extract most relevant sentences from text"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # Prioritize sentences with key terms
    key_terms = ['graph', 'vertex', 'edge', 'adjacency', 'matrix', 'list', 'algorithm', 'data structure', 
                'represent', 'connection', 'node', 'implementation', 'time complexity', 'space complexity']
    
    # Score sentences based on key terms
    scored_sentences = []
    for sentence in sentences:
        score = sum(1 for term in key_terms if term.lower() in sentence.lower())
        if score > 0 or len(scored_sentences) < max_sentences:
            scored_sentences.append((score, sentence))
    
    # Sort by score and take top sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    top_sentences = [sentence for _, sentence in scored_sentences[:max_sentences]]
    
    return '. '.join(top_sentences) + '.' if top_sentences else text[:200] + '...'

def smart_truncate(text: str, max_chars: int = 600) -> str:
    """Intelligently truncate text while preserving meaning"""
    if len(text) <= max_chars:
        return text
    
    # Try to cut at sentence boundaries
    sentences = text.split('.')
    result = ""
    
    for sentence in sentences:
        if len(result + sentence + '.') <= max_chars:
            result += sentence + '.'
        else:
            break
    
    if not result:
        # If no complete sentences fit, cut at word boundary
        words = text[:max_chars].split()
        result = ' '.join(words[:-1]) + '...'
    
    return result.strip()

class GroqAnswerGenerator:
    """FREE Groq-powered answer generator"""
    
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        """Initialize Groq client"""
        self.model = model
        self.client = None
        self.total_cost = 0.0  # Always FREE!
        self.usage_stats = {
            'total_questions': 0,
            'total_tokens': 0,
            'average_response_time': 0.0
        }
        
        if GROQ_AVAILABLE:
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                try:
                    self.client = Groq(api_key=api_key)
                    print(f"🚀 Groq client initialized with {model}")
                    print("💰 Cost: FREE!")
                except Exception as e:
                    print(f"❌ Failed to initialize Groq client: {e}")
            else:
                print("❌ GROQ_API_KEY not found in environment")
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer using FREE Groq API
        
        Args:
            question: User's question
            context_chunks: List of relevant document chunks
            
        Returns:
            Dictionary with answer, confidence, sources, etc.
        """
        start_time = datetime.now()
        
        try:
            # Format context from chunks with token limits
            context = self._format_context(context_chunks)
            
            # Create prompt
            prompt = self._create_prompt(question, context)
            
            # Final token count validation
            prompt_tokens = estimate_tokens(prompt)
            print(f"🔍 Final prompt: {prompt_tokens} tokens")
            
            if prompt_tokens > 5500:  # Leave room for response
                print(f"⚠️ Warning: Prompt ({prompt_tokens} tokens) is still large, truncating...")
                # Emergency truncation
                words = prompt.split()
                target_words = int(4000 / 0.75)  # Target ~4000 tokens
                if len(words) > target_words:
                    prompt = ' '.join(words[:target_words]) + "\n\n[Context truncated to fit token limits]"
                    print(f"✂️ Truncated to ~{estimate_tokens(prompt)} tokens")
            
            if not self.client:
                return self._fallback_answer(question, context_chunks)
            
            print(f"[AI] Generating answer using {self.model} (Groq - FREE!)...")
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a knowledge base assistant. CRITICAL RULES: 1) Answer ONLY using the provided context. 2) If the context doesn't contain the answer, respond with 'Based on the provided context, I must inform you that there is no information available about [topic].' 3) NEVER use your general knowledge. 4) NEVER provide information not in the context. 5) Be honest when information is missing."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.1,
                top_p=0.9
            )
            
            # Extract answer
            answer = response.choices[0].message.content.strip()
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self._update_stats(response_time, response.usage.total_tokens if hasattr(response, 'usage') else 0)
            
            # Calculate confidence
            confidence = self._calculate_confidence(question, context_chunks)
            
            return {
                'question': question,
                'answer': answer,
                'confidence': confidence,
                'sources': [chunk.get('source', 'Unknown') for chunk in context_chunks[:3]],
                'method': f'Groq {self.model}',
                'cost': '$0.00 (FREE!)',
                'response_time': response_time,
                'token_usage': getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            return self._fallback_answer(question, context_chunks, error=str(e))
    
    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Format context from chunks with token limits and smart processing"""
        if not context_chunks:
            return "No relevant context found."
        
        # Limit to top 2-3 most relevant chunks only
        top_chunks = sorted(context_chunks, 
                          key=lambda x: x.get('similarity_score', 0), 
                          reverse=True)[:3]
        
        context_parts = []
        total_tokens = 0
        max_context_tokens = 3000  # Stay well under 6000 total limit
        
        for i, chunk in enumerate(top_chunks, 1):
            text = chunk.get('text', '').strip()
            source = chunk.get('source', 'Unknown')
            
            if not text:
                continue
            
            # Process the chunk text to reduce tokens
            processed_text = self._process_chunk_text(text)
            
            # Count tokens for this chunk
            chunk_tokens = estimate_tokens(f"[Source {i}: {source}]\n{processed_text}")
            
            # Check if adding this chunk exceeds limit
            if total_tokens + chunk_tokens > max_context_tokens:
                # Try to truncate this chunk to fit
                remaining_tokens = max_context_tokens - total_tokens
                if remaining_tokens > 100:  # Only add if meaningful space left
                    # Estimate characters for remaining tokens
                    remaining_chars = remaining_tokens * 4  # Rough: 4 chars per token
                    processed_text = smart_truncate(processed_text, remaining_chars)
                    context_parts.append(f"[Source {i}: {source}]\n{processed_text}")
                break
            
            context_parts.append(f"[Source {i}: {source}]\n{processed_text}")
            total_tokens += chunk_tokens
        
        result = "\n\n".join(context_parts)
        final_tokens = estimate_tokens(result)
        
        print(f"📊 Context processed: {final_tokens} tokens from {len(top_chunks)} chunks")
        
        return result
    
    def _process_chunk_text(self, text: str) -> str:
        """Process chunk text to reduce token count while preserving key information"""
        # Step 1: Remove code examples (major token saver)
        text = remove_code_examples(text)
        
        # Step 2: Extract key sentences
        text = extract_key_sentences(text, max_sentences=4)
        
        # Step 3: Smart truncation to max characters
        text = smart_truncate(text, max_chars=600)
        
        # Step 4: Clean up formatting
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\[.*?\]', '', text)  # Remove reference brackets
        
        return text.strip()
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for Groq"""
        return f"""**CONTEXT:**
{context}

**QUESTION:** {question}

**CRITICAL INSTRUCTIONS:**
1. Answer ONLY using the provided context above
2. If the context does not contain information about the question, respond EXACTLY with: "Based on the provided context, I must inform you that there is no information available about [the topic]."
3. DO NOT use any general knowledge or information outside the context
4. DO NOT provide explanations, tutorials, or guidance not present in the context
5. If answering from context, cite the specific sources
6. Be precise and factual

**ANSWER:**"""
    
    def _calculate_confidence(self, question: str, context_chunks: List[Dict]) -> float:
        """Calculate confidence score"""
        if not context_chunks:
            return 0.0
        
        # Base confidence on similarity scores
        avg_similarity = sum(chunk.get('similarity_score', 0) for chunk in context_chunks) / len(context_chunks)
        
        # Boost confidence based on number of relevant chunks
        chunk_bonus = min(len(context_chunks) * 0.1, 0.3)
        
        # Question-context relevance
        question_words = set(question.lower().split())
        context_text = ' '.join(chunk.get('text', '') for chunk in context_chunks).lower()
        context_words = set(context_text.split())
        
        overlap = len(question_words.intersection(context_words))
        relevance_bonus = min(overlap * 0.05, 0.2)
        
        final_confidence = min(avg_similarity + chunk_bonus + relevance_bonus, 0.95)
        return round(final_confidence, 2)
    
    def _update_stats(self, response_time: float, tokens: int):
        """Update usage statistics"""
        self.usage_stats['total_questions'] += 1
        self.usage_stats['total_tokens'] += tokens
        
        # Update average response time
        current_avg = self.usage_stats['average_response_time']
        total_questions = self.usage_stats['total_questions']
        self.usage_stats['average_response_time'] = (current_avg * (total_questions - 1) + response_time) / total_questions
    
    def _fallback_answer(self, question: str, context_chunks: List[Dict], error: str = None) -> Dict[str, Any]:
        """Fallback to rule-based answering if Groq fails"""
        try:
            from rag.free_llm_answering import generate_free_answer
            print("🔄 Falling back to free rule-based answering...")
            return generate_free_answer(question, context_chunks)
        except ImportError:
            context = self._format_context(context_chunks)
            fallback_answer = f"Based on the available context: {context[:300]}..." if context else "No relevant information found."
            
            return {
                'question': question,
                'answer': fallback_answer,
                'confidence': 0.5 if context else 0.0,
                'sources': [chunk.get('source', 'Unknown') for chunk in context_chunks[:3]],
                'method': 'Fallback (Rule-based)',
                'cost': '$0.00 (FREE!)',
                'error': error,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            **self.usage_stats,
            'total_cost': '$0.00 (FREE!)',
            'model': self.model,
            'provider': 'Groq'
        }

# Global instance
groq_generator = None

def initialize_groq_generator(model: str = None) -> GroqAnswerGenerator:
    """Initialize the global Groq generator"""
    global groq_generator
    
    if model is None:
        model = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
    
    print(f"[AI] Initializing Groq answer generator with {model}...")
    groq_generator = GroqAnswerGenerator(model=model)
    print("✅ Groq answer generator ready!")
    
    return groq_generator

def generate_groq_answer(question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to generate answers using FREE Groq API
    
    Args:
        question: User's question
        context_chunks: List of relevant document chunks
        
    Returns:
        Answer dictionary with text, confidence, sources, etc.
    """
    global groq_generator
    
    if groq_generator is None:
        groq_generator = initialize_groq_generator()
    
    return groq_generator.generate_answer(question, context_chunks)

def format_groq_response(response: Dict[str, Any]) -> str:
    """Format Groq response for display"""
    answer = response.get('answer', 'No answer generated.')
    confidence = response.get('confidence', 0.0)
    sources = response.get('sources', [])
    cost = response.get('cost', '$0.00 (FREE!)')
    response_time = response.get('response_time', 0.0)
    
    formatted = f"📝 **Answer:**\n{answer}\n\n"
    
    if sources:
        formatted += "📚 **Sources:**\n"
        for i, source in enumerate(sources, 1):
            formatted += f"  {i}. {source}\n"
        formatted += "\n"
    
    formatted += f"📊 **Details:**\n"
    formatted += f"  🎯 Confidence: {confidence:.1%}\n"
    formatted += f"  ⚡ Response Time: {response_time:.2f}s\n"
    formatted += f"  💰 Cost: {cost}\n"
    formatted += f"  🤖 Method: {response.get('method', 'Groq AI')}"
    
    return formatted

def get_usage_stats() -> Dict[str, Any]:
    """Get Groq usage statistics"""
    global groq_generator
    
    if groq_generator is None:
        return {"error": "Groq generator not initialized"}
    
    return groq_generator.get_stats()

# Test function
def test_groq_answering():
    """Test the Groq answering system"""
    test_chunks = [
        {
            'text': 'A graph is a data structure consisting of vertices (nodes) connected by edges. It represents relationships between entities.',
            'source': 'graph_tutorial.txt',
            'similarity_score': 0.85
        },
        {
            'text': 'The two main ways to represent graphs are adjacency matrix and adjacency list.',
            'source': 'graph_representations.txt',
            'similarity_score': 0.75
        }
    ]
    
    test_question = "What is a graph data structure?"
    
    print("🧪 Testing Groq Answer Generation")
    print("=" * 50)
    
    response = generate_groq_answer(test_question, test_chunks)
    formatted = format_groq_response(response)
    
    print(formatted)
    print("\n" + "=" * 50)
    print("🎉 Groq test completed!")

if __name__ == "__main__":
    test_groq_answering()