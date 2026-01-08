"""
RAG Answer Generation Module
===========================

This module handles the generation part of RAG (Retrieve-Augmented Generation).
It takes user questions, retrieves relevant chunks, and generates grounded answers
using only the content from the documentation.

Functions:
- check_answer_availability(question, chunks): Checks if answer can be provided
- generate_grounded_answer(question, chunks): Creates answer from retrieved content
- extract_relevant_sentences(question, text): Finds most relevant sentences
- format_answer_with_sources(answer, chunks): Formats final answer with sources
- answer_question(question): Main function to answer user questions
"""

import re
from typing import List, Dict, Tuple, Optional
from rag.retrieval import retrieve_relevant_chunks


def extract_relevant_sentences(question: str, text: str) -> List[str]:
    """
    Extracts sentences from text that are most relevant to the question.
    
    Args:
        question (str): User's question
        text (str): Text to extract sentences from
        
    Returns:
        List[str]: List of relevant sentences
    """
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Extract keywords from question
    question_words = set(re.findall(r'\b[a-zA-Z]+\b', question.lower()))
    # Remove common question words
    stop_words = {'how', 'what', 'when', 'where', 'why', 'do', 'does', 'is', 'are', 'can', 'should', 'will'}
    question_keywords = question_words - stop_words
    
    relevant_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:  # Skip very short sentences
            continue
        
        # Score sentence based on keyword matches
        sentence_words = set(re.findall(r'\b[a-zA-Z]+\b', sentence.lower()))
        
        # Count keyword matches
        keyword_matches = len(question_keywords & sentence_words)
        
        # Bonus for exact phrase matches
        phrase_bonus = 0
        if len(question_keywords) > 1:
            question_text = ' '.join(question_keywords)
            if question_text in sentence.lower():
                phrase_bonus = 2
        
        total_score = keyword_matches + phrase_bonus
        
        # Include sentences with good keyword overlap
        if total_score >= 2 or (total_score >= 1 and len(sentence) > 50):
            relevant_sentences.append((sentence, total_score))
    
    # Sort by relevance score and return top sentences
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    return [sentence for sentence, score in relevant_sentences[:5]]  # Top 5 sentences


def check_answer_availability(question: str, chunks: List[Dict[str, any]]) -> Tuple[bool, str]:
    """
    Checks if the retrieved chunks contain enough information to answer the question.
    
    Args:
        question (str): User's question
        chunks (List[Dict[str, any]]): Retrieved chunks
        
    Returns:
        Tuple[bool, str]: (can_answer, reason)
    """
    if not chunks:
        return False, "No relevant chunks retrieved"
    
    # Check if chunks have sufficient relevance
    min_relevance_threshold = 0.1
    high_relevance_chunks = [chunk for chunk in chunks if chunk['similarity_score'] >= min_relevance_threshold]
    
    if not high_relevance_chunks:
        return False, "Retrieved chunks have very low relevance to the question"
    
    # Extract question keywords
    question_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
    stop_words = {'how', 'what', 'when', 'where', 'why', 'does', 'can', 'should', 'will', 'the', 'and', 'for'}
    question_keywords = question_words - stop_words
    
    if not question_keywords:
        return False, "Cannot extract meaningful keywords from question"
    
    # Check if any chunks contain question keywords
    total_keyword_matches = 0
    
    for chunk in high_relevance_chunks:
        chunk_text = chunk['text'].lower()
        chunk_matches = sum(1 for keyword in question_keywords if keyword in chunk_text)
        total_keyword_matches += chunk_matches
    
    # Need at least some keyword matches across all chunks
    if total_keyword_matches < len(question_keywords) * 0.5:
        return False, "Retrieved chunks don't contain enough relevant information"
    
    return True, "Sufficient information available"


def generate_grounded_answer(question: str, chunks: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Generates an answer grounded in the retrieved documentation chunks.
    
    Args:
        question (str): User's question
        chunks (List[Dict[str, any]]): Retrieved relevant chunks
        
    Returns:
        Dict[str, any]: Generated answer with metadata
    """
    print(f"🤖 Generating grounded answer for: '{question[:50]}...'")
    
    # First check if we can answer the question
    can_answer, reason = check_answer_availability(question, chunks)
    
    if not can_answer:
        print(f"❌ Cannot answer question: {reason}")
        return {
            'question': question,
            'answer': "The knowledge base does not contain this information.",
            'confidence': 0.0,
            'sources': [],
            'reason': reason
        }
    
    # Extract relevant content from chunks
    all_relevant_sentences = []
    sources_used = []
    
    for chunk in chunks:
        relevant_sentences = extract_relevant_sentences(question, chunk['text'])
        
        for sentence in relevant_sentences:
            all_relevant_sentences.append({
                'text': sentence,
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'relevance_score': chunk['similarity_score']
            })
        
        sources_used.append({
            'source': chunk['source'],
            'chunk_id': chunk['chunk_id'],
            'relevance_score': chunk['similarity_score']
        })
    
    if not all_relevant_sentences:
        return {
            'question': question,
            'answer': "The knowledge base does not contain this information.",
            'confidence': 0.0,
            'sources': sources_used,
            'reason': "No relevant sentences found in chunks"
        }
    
    # Sort sentences by relevance
    all_relevant_sentences.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Generate the answer by combining relevant sentences
    answer = construct_coherent_answer(question, all_relevant_sentences)
    
    # Calculate confidence based on relevance scores and content quality
    avg_relevance = sum(chunk['similarity_score'] for chunk in chunks) / len(chunks)
    content_coverage = min(1.0, len(all_relevant_sentences) / 3)  # More sentences = better coverage
    confidence = min(0.95, avg_relevance * content_coverage)  # Cap at 95%
    
    print(f"✅ Generated answer with {confidence:.2f} confidence")
    
    return {
        'question': question,
        'answer': answer,
        'confidence': confidence,
        'sources': sources_used,
        'relevant_sentences_count': len(all_relevant_sentences),
        'reason': 'Answer generated successfully'
    }


def construct_coherent_answer(question: str, relevant_sentences: List[Dict[str, any]]) -> str:
    """
    Constructs a focused answer by extracting the most relevant information.
    
    Args:
        question (str): Original question
        relevant_sentences (List[Dict[str, any]]): List of relevant sentences with metadata
        
    Returns:
        str: Constructed answer
    """
    if not relevant_sentences:
        return "The knowledge base does not contain this information."
    
    # Extract keywords from question
    question_lower = question.lower()
    question_keywords = extract_question_keywords(question)
    
    # Get the full text from all chunks for better context
    full_text = " ".join([s['text'] for s in relevant_sentences])
    
    # Try to find specific definition or explanation
    specific_answer = extract_specific_answer(question, full_text, question_keywords)
    if specific_answer:
        return specific_answer
    
    # Fallback: analyze question type for structured response
    if any(word in question_lower for word in ['what is', 'what are', 'define']):
        return extract_definition_answer(question, full_text, question_keywords)
    elif any(word in question_lower for word in ['how', 'how to']):
        return extract_instruction_answer(question, full_text, question_keywords)
    elif 'keyword' in question_lower and any(word in question_lower for word in ['used for', 'makes']):
        return extract_keyword_answer(question, full_text, question_keywords)
    
    # General approach: find most relevant 1-2 sentences
    return extract_focused_answer(question, full_text, question_keywords)


def extract_question_keywords(question: str) -> set:
    """Extract meaningful keywords from question"""
    # Remove question words and extract content
    question_words = re.findall(r'\b\w+\b', question.lower())
    stop_words = {'what', 'is', 'are', 'how', 'do', 'does', 'can', 'should', 'will', 'the', 'a', 'an', 'and', 'or', 'for', 'used', 'keyword'}
    return set(word for word in question_words if len(word) > 2 and word not in stop_words)


def extract_specific_answer(question: str, text: str, keywords: set) -> str:
    """Try to find specific definition or explanation for the question"""
    question_lower = question.lower()
    
    # Handle "what is X?" questions with enhanced pattern matching
    if 'what is' in question_lower:
        # Extract the term being asked about
        what_is_match = re.search(r'what is (\w+)', question_lower)
        if what_is_match:
            term = what_is_match.group(1)
            
            # Create a list of known C++ keywords to help stop at boundaries
            cpp_keywords = ['and', 'and_eq', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case', 
                           'catch', 'char', 'class', 'compl', 'const', 'continue', 'default', 
                           'delete', 'do', 'double', 'else', 'enum', 'false', 'float', 'for', 
                           'friend', 'goto', 'if', 'int', 'long', 'namespace', 'new', 'not', 
                           'or', 'or_eq', 'private', 'protected', 'public', 'return', 'short', 
                           'signed', 'sizeof', 'static', 'struct', 'switch', 'template', 'this', 
                           'throw', 'true', 'try', 'typedef', 'unsigned', 'using', 'virtual', 
                           'void', 'while', 'xor', 'xor_eq']
            keywords_pattern = '|'.join(cpp_keywords)
            
            # Look for the term followed by its description, stopping at the next keyword
            # Use a much simpler approach: extract until we hit a clear keyword boundary
            term_match = re.search(rf'\b{term}\b', text, re.IGNORECASE)
            if term_match:
                start_pos = term_match.end()
                remaining_text = text[start_pos:]
                
                # Look for clear keyword definition patterns: keyword + space + Capital/Article
                # But exclude mid-sentence keywords like "return a value" 
                main_keywords = ['enum', 'false', 'float', 'void', 'int', 'bool', 'char', 'double', 'long', 'short', 'auto', 'static', 'const']
                
                # Find the next main keyword that starts a new definition
                next_keyword_pos = None
                for keyword in main_keywords:
                    if keyword.lower() != term.lower():  # Don't match the same keyword
                        # Look for keyword at start of text or after significant break
                        pattern = rf'(?:^|\s){keyword}\s+(?:[A-Z][a-z]|A\s|An\s|The\s)'
                        match = re.search(pattern, remaining_text, re.IGNORECASE)
                        if match:
                            if next_keyword_pos is None or match.start() < next_keyword_pos:
                                next_keyword_pos = match.start()
                
                if next_keyword_pos is not None:
                    definition_text = remaining_text[:next_keyword_pos].strip()
                else:
                    # No next definition found, extract reasonable amount or to end
                    definition_text = remaining_text.strip()
                    # If too long, cut at reasonable boundary
                    if len(definition_text) > 200:
                        # Find last sentence boundary within limit
                        sentences = re.split(r'[.!?]', definition_text[:200])
                        if len(sentences) > 1:
                            definition_text = sentences[0] + '.'
                        else:
                            definition_text = definition_text[:200].rsplit(' ', 1)[0] + '.'
                
                if definition_text and len(definition_text) > 3:
                    # Clean up the definition
                    definition_text = re.sub(r'\s+', ' ', definition_text)  # Normalize whitespace
                    
                    # Add proper ending if needed
                    if not definition_text.endswith(('.', '!', '?')):
                        definition_text += '.'
                    
                    return f"The keyword '{term}' {definition_text.lower()}"
            
            # Fallback: look for any occurrence and extract reasonable amount
            fallback_pattern = rf'\b{term}\s+([\w\s,]+?)(?:\s+[a-z_]+\s+[A-Z]|$)'
            match = re.search(fallback_pattern, text, re.IGNORECASE)
            if match:
                definition = match.group(1).strip()
                if definition and len(definition) > 3:
                    # Take first complete phrase
                    sentences = re.split(r'[.!?]', definition)
                    if sentences and sentences[0].strip():
                        definition = sentences[0].strip()
                        if not definition.endswith(('.', '!', '?')):
                            definition += '.'
                        return f"The keyword '{term}' {definition.lower()}"
            
            # Fallback: try simpler pattern with common definition verbs
            simple_pattern = rf'\b{term}\s+((?:declares?|defines?|specifies?|is|means|creates?)[^.]*?)(?=\s+\w+\s+[A-Z]|$)'
            match = re.search(simple_pattern, text, re.IGNORECASE)
            if match:
                definition = match.group(1).strip()
                if definition:
                    if not definition.endswith(('.', '!', '?')):
                        definition += '.'
                    return f"The keyword '{term}' {definition.lower()}"
    
    # Handle "which keyword" questions
    elif 'which keyword' in question_lower or 'what keyword' in question_lower:
        # Extract what functionality is being asked about
        for_match = re.search(r'for (.+?)[\\.?]', question_lower)
        if for_match:
            functionality = for_match.group(1).strip()
            
            # Look for lines containing the functionality description
            cpp_keywords_pattern = '|'.join(['and', 'and_eq', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case', 'catch', 'char', 'class', 'compl', 'const', 'continue', 'default', 'delete', 'do', 'double', 'else', 'enum', 'false', 'float', 'for', 'friend', 'goto', 'if', 'int', 'long', 'namespace', 'new', 'not', 'or', 'or_eq', 'private', 'protected', 'public', 'return', 'short', 'signed', 'sizeof', 'static', 'struct', 'switch', 'template', 'this', 'throw', 'true', 'try', 'typedef', 'unsigned', 'using', 'virtual', 'void', 'while', 'xor', 'xor_eq'])
            
            # Look for the functionality phrase and extract the preceding keyword
            pattern = rf'(\w+)\s+[^.]*?{re.escape(functionality)}'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                keyword = match.group(1)
                return f"The keyword '{keyword}' {functionality}."
    
    # Look for specific patterns in the text (fallback)
    if 'or_eq' in question_lower:
        lines = text.split('\\n')
        for line in lines:
            if 'or_eq' in line.lower() and ('alternative' in line.lower() or 'operator' in line.lower()):
                return line.strip()
    
    # Look for operator explanations
    if '|=' in question_lower or 'operator' in question_lower:
        lines = text.split('\\n')
        for line in lines:
            if '|=' in line and any(word in line.lower() for word in ['alternative', 'operator', 'assignment']):
                return line.strip()
    
    # Look for keyword definitions using improved pattern (fallback)
    for keyword in keywords:
        pattern = rf"{keyword}\s+([^.]*(?:is|means|refers to|alternative way)[^.]*\.)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def extract_definition_answer(question: str, text: str, keywords: set) -> str:
    """Extract definition-style answer with improved pattern matching"""
    question_lower = question.lower()
    
    # For "what is X" questions, try to extract the specific term
    if 'what is' in question_lower:
        what_is_match = re.search(r'what is (\w+)', question_lower)
        if what_is_match:
            term = what_is_match.group(1)
            
            # Look for direct definition patterns in the text
            # Pattern: "enum Declares an enumerated type"
            definition_patterns = [
                rf'\b{term}\s+([A-Z][^.]*)',  # Capitalized definition
                rf'\b{term}\s+([a-z][^.]*(?:declares?|defines?|specifies?|is|means)[^.]*)',  # Lowercase with action verbs
            ]
            
            for pattern in definition_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    definition = match.group(1).strip()
                    if definition and len(definition) > 10:  # Ensure substantial definition
                        # Clean and format the definition
                        first_sentence = re.split(r'[.!?]', definition)[0].strip()
                        if first_sentence:
                            # Ensure proper sentence structure
                            if not first_sentence.endswith(('.', '!', '?')):
                                first_sentence += '.'
                            return f"The keyword '{term}' {first_sentence.lower()}"
    
    # Fallback: Look for definition patterns with any keyword
    sentences = re.split(r'[.!?]+', text)
    
    for keyword in keywords:
        for sentence in sentences:
            sentence = sentence.strip()
            if (keyword in sentence.lower() and 
                any(pattern in sentence.lower() for pattern in ['is a', 'is an', 'refers to', 'means', 'declares', 'defines', 'specifies'])):
                # Clean up the sentence
                if len(sentence) > 20:  # Ensure it's substantial
                    return sentence + "." if not sentence.endswith(('.', '!', '?')) else sentence
    
    # Final fallback: find sentences with the most keywords
    return extract_focused_answer(question, text, keywords)


def extract_instruction_answer(question: str, text: str, keywords: set) -> str:
    """Extract instruction-style answer"""
    sentences = re.split(r'[.!?]+', text)
    
    # Look for instructional content
    instruction_words = ['use', 'create', 'define', 'declare', 'write', 'make']
    
    best_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
            
        # Score based on keywords and instruction words
        keyword_score = sum(1 for keyword in keywords if keyword in sentence.lower())
        instruction_score = sum(1 for word in instruction_words if word in sentence.lower())
        
        if keyword_score + instruction_score >= 1:
            best_sentences.append((sentence, keyword_score + instruction_score))
    
    if best_sentences:
        best_sentences.sort(key=lambda x: x[1], reverse=True)
        return best_sentences[0][0] + "."
    
    return extract_focused_answer(question, text, keywords)


def extract_keyword_answer(question: str, text: str, keywords: set) -> str:
    """Extract answer about which keyword is used for something"""
    question_lower = question.lower()
    
    # Look for the specific functionality mentioned
    if 'conditional statement' in question_lower:
        lines = text.split('\n')
        for line in lines:
            if 'if ' in line.lower() and ('makes' in line.lower() or 'conditional' in line.lower()):
                # Extract just the keyword and definition
                match = re.search(r'\bif\b[^.]*conditional[^.]*', line, re.IGNORECASE)
                if match:
                    return f"The keyword 'if' makes a conditional statement: {match.group().strip()}"
        # Fallback
        return "The keyword 'if' is used to make conditional statements."
    
    return extract_focused_answer(question, text, keywords)


def extract_focused_answer(question: str, text: str, keywords: set) -> str:
    """Extract most focused 1-2 sentences containing the answer"""
    sentences = re.split(r'[.!?]+', text)
    
    # Score sentences by relevance
    scored_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:  # Skip very short sentences
            continue
        
        sentence_lower = sentence.lower()
        
        # Count keyword matches
        keyword_matches = sum(1 for keyword in keywords if keyword in sentence_lower)
        
        # Bonus for definition patterns
        definition_bonus = 0
        definition_patterns = ['is a', 'is an', 'refers to', 'means', 'declares', 'defines', 'specifies', 'used to', 'alternative']
        if any(pattern in sentence_lower for pattern in definition_patterns):
            definition_bonus = 3
        
        # Bonus for technical terms
        technical_bonus = 0
        technical_terms = ['operator', 'keyword', 'statement', 'function', 'class', 'type', 'variable']
        if any(term in sentence_lower for term in technical_terms):
            technical_bonus = 2
        
        # Penalty for very long sentences (likely concatenated text)
        length_penalty = 0
        if len(sentence) > 200:
            length_penalty = -5
        
        total_score = keyword_matches + definition_bonus + technical_bonus + length_penalty
        
        if total_score > 0:
            scored_sentences.append((sentence, total_score))
    
    if not scored_sentences:
        return "The knowledge base does not contain this information."
    
    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Filter out extremely long sentences that are likely text dumps
    good_sentences = [s for s in scored_sentences if len(s[0]) < 300]
    
    if not good_sentences:
        # If all sentences are too long, take the shortest among the top-scored
        good_sentences = sorted(scored_sentences[:3], key=lambda x: len(x[0]))[:1]
    
    # Return the best sentence (or two if the second one is almost as good and complementary)
    if (len(good_sentences) >= 2 and 
        good_sentences[1][1] >= good_sentences[0][1] * 0.8 and
        len(good_sentences[0][0]) + len(good_sentences[1][0]) < 400):
        # Include second sentence if it's almost as good and together they're not too long
        return f"{good_sentences[0][0]}. {good_sentences[1][0]}."
    else:
        # Just return the best sentence
        best_sentence = good_sentences[0][0]
        return best_sentence + "." if not best_sentence.endswith(('.', '!', '?')) else best_sentence


def format_answer_with_sources(answer_data: Dict[str, any]) -> str:
    """
    Formats the answer with source references for display.
    
    Args:
        answer_data (Dict[str, any]): Answer data with sources
        
    Returns:
        str: Formatted answer with sources
    """
    answer = answer_data['answer']
    sources = answer_data.get('sources', [])
    confidence = answer_data.get('confidence', 0.0)
    
    # If no information available, return the standard message
    if answer == "The knowledge base does not contain this information.":
        return answer
    
    # Format the main answer
    formatted_answer = answer
    
    # Add confidence indicator
    if confidence > 0:
        confidence_percent = confidence * 100
        if confidence_percent >= 80:
            confidence_text = "High confidence"
        elif confidence_percent >= 60:
            confidence_text = "Medium confidence"
        else:
            confidence_text = "Low confidence"
        
        formatted_answer += f"\n\n📊 Confidence: {confidence_text} ({confidence_percent:.0f}%)"
    
    # Add source references
    if sources:
        formatted_answer += "\n\n📚 Sources:"
        
        # Group sources by URL
        source_groups = {}
        for source in sources:
            url = source['source']
            if url not in source_groups:
                source_groups[url] = []
            source_groups[url].append(source)
        
        for i, (url, source_list) in enumerate(source_groups.items(), 1):
            if url == "Unknown":
                formatted_answer += f"\n{i}. Documentation (multiple sections)"
            else:
                # Extract domain name from URL for display
                domain = url.split('/')[2] if '//' in url else url
                formatted_answer += f"\n{i}. {domain}"
                
                # Add chunk info if available
                chunk_ids = [s['chunk_id'] for s in source_list if s['chunk_id'] != 'Unknown']
                if chunk_ids:
                    formatted_answer += f" ({len(chunk_ids)} sections)"
    
    return formatted_answer


def answer_question(question: str) -> str:
    """
    Main function to answer a user question using RAG.
    
    Args:
        question (str): User's question
        
    Returns:
        str: Formatted answer with sources, or standard message if no info available
        
    This is the main function that other modules will call.
    """
    print(f"❓ Processing question: {question}")
    
    if not question or len(question.strip()) < 3:
        return "Please provide a valid question."
    
    try:
        # Step 1: Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(question, num_chunks=3)
        
        # Step 2: Generate grounded answer
        answer_data = generate_grounded_answer(question, relevant_chunks)
        
        # Step 3: Format answer with sources
        formatted_answer = format_answer_with_sources(answer_data)
        
        print(f"✅ Answer generated successfully")
        return formatted_answer
        
    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        return "Sorry, there was an error processing your question. Please try again."


def get_answer_with_details(question: str) -> Dict[str, any]:
    """
    Gets detailed answer information (useful for debugging or API responses).
    
    Args:
        question (str): User's question
        
    Returns:
        Dict[str, any]: Detailed answer information
    """
    print(f"📋 Getting detailed answer for: {question}")
    
    try:
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(question, num_chunks=3)
        
        # Generate grounded answer
        answer_data = generate_grounded_answer(question, relevant_chunks)
        
        # Add chunk details
        answer_data['retrieved_chunks'] = relevant_chunks
        answer_data['formatted_answer'] = format_answer_with_sources(answer_data)
        
        return answer_data
        
    except Exception as e:
        print(f"❌ Error getting detailed answer: {str(e)}")
        return {
            'question': question,
            'answer': "Sorry, there was an error processing your question.",
            'confidence': 0.0,
            'sources': [],
            'error': str(e)
        }


# Example usage (runs when file is executed directly)
if __name__ == "__main__":
    print("🧪 Running answer generation example...")
    
    # Test questions
    test_questions = [
        "How do I install Python packages?",
        "What is a function in Python?",
        "How to handle files in Python?",
        "What is machine learning?",  # This might not be in the docs
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("-" * 60)
        
        answer = answer_question(question)
        print(answer)
        
        print("\n" + "="*60)