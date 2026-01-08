"""
Document Ingestion Module
=========================

This module handles downloading and processing documentation from web sources.
It cleans HTML content, extracts readable text, and splits it into manageable chunks.

Functions:
- download_webpage(url): Downloads HTML content from a URL
- clean_html_content(html): Removes navigation, ads, scripts from HTML
- extract_text_content(soup): Extracts clean text from BeautifulSoup object
- split_into_chunks(text): Splits text into 400-700 word chunks
- save_chunks_to_files(chunks, source_url): Saves chunks as text files
- process_documentation(urls): Main function to process multiple URLs
"""

import requests
from bs4 import BeautifulSoup
import os
import re
import time
from urllib.parse import urlparse
from typing import List, Dict


def extract_w3schools_keywords(soup: BeautifulSoup) -> str:
    """
    Extract keyword-description pairs from W3Schools keyword pages.
    
    Args:
        soup (BeautifulSoup): Parsed HTML soup
        
    Returns:
        str: Structured keyword descriptions
    """
    content = []
    
    # Look for keyword table
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows[1:]:  # Skip header row
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                keyword = cells[0].get_text().strip()
                description = cells[1].get_text().strip()
                if keyword and description and len(keyword) < 50:  # Valid keyword-description pair
                    content.append(f"{keyword}: {description}")
    
    # If no table found, look for definition lists
    if not content:
        dl_elements = soup.find_all('dl')
        for dl in dl_elements:
            dt_elements = dl.find_all('dt')
            dd_elements = dl.find_all('dd')
            for dt, dd in zip(dt_elements, dd_elements):
                keyword = dt.get_text().strip()
                description = dd.get_text().strip()
                if keyword and description:
                    content.append(f"{keyword}: {description}")
    
    # Fallback: look for any structured keyword content
    if not content:
        # Look for pattern: keyword followed by description
        main_div = soup.find('div', id='main') or soup.find('div', class_='w3-col l10 m12')
        if main_div:
            paragraphs = main_div.find_all(['p', 'div', 'span'])
            for p in paragraphs:
                text = p.get_text().strip()
                # Look for keyword: description pattern
                if ':' in text and len(text) < 200:
                    lines = text.split('\n')
                    for line in lines:
                        if ':' in line and len(line.strip()) > 10:
                            content.append(line.strip())
    
    return '\n'.join(content) if content else soup.get_text()


def extract_w3schools_content(soup: BeautifulSoup) -> str:
    """
    Extract content from general W3Schools pages.
    
    Args:
        soup (BeautifulSoup): Parsed HTML soup
        
    Returns:
        str: Clean content
    """
    # Find main content area
    main_content = soup.find('div', id='main') or soup.find('div', class_='w3-col l10 m12')
    
    if main_content:
        # Remove navigation and ads within main content
        for unwanted in main_content.find_all(['nav', 'aside']):
            unwanted.decompose()
        for ad in main_content.find_all(class_=lambda x: x and ('ad' in str(x).lower() or 'w3-sidebar' in str(x).lower()) if x else False):
            ad.decompose()
        
        # Extract educational content (examples, explanations, code)
        content_elements = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre', 'code', 'li', 'table', 'blockquote'])
        text = ' '.join([elem.get_text() for elem in content_elements])
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    return soup.get_text()


def debug_extraction(url: str) -> None:
    """
    Debug what content is being extracted from a URL.
    
    Args:
        url (str): URL to debug
    """
    print(f"🔍 Debugging extraction for: {url}")
    
    try:
        # Download and parse
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=30)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check for main content areas
        main_div = soup.find('div', id='main')
        if main_div:
            print("✅ Found main content div")
            
            # Look for keyword table
            table = main_div.find('table')
            if table:
                print("✅ Found keyword table")
                rows = table.find_all('tr')
                print(f"📊 Table has {len(rows)} rows")
                
                # Show first few keywords
                for i, row in enumerate(rows[1:6]):  # Skip header, show 5
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        keyword = cells[0].get_text().strip()
                        desc = cells[1].get_text().strip()[:50]
                        print(f"   {i+1}. {keyword}: {desc}...")
            else:
                print("❌ No keyword table found")
        else:
            print("❌ No main content div found")
            
        # Try W3Schools specific extraction
        if 'w3schools.com' in url.lower():
            extracted = extract_w3schools_keywords(soup) if 'keywords' in url.lower() else extract_w3schools_content(soup)
            print(f"📄 Extracted content length: {len(extracted)} characters")
            print(f"📄 First 200 characters: {extracted[:200]}...")
            
    except Exception as e:
        print(f"❌ Debug failed: {str(e)}")


def download_webpage(url: str, timeout: int = 30) -> str:
    """
    Downloads the HTML content from a given URL.
    
    Args:
        url (str): The URL to download
        timeout (int): Request timeout in seconds
        
    Returns:
        str: HTML content as string, empty if failed
        
    Example:
        html = download_webpage("https://docs.python.org/3/")
    """
    try:
        print(f"📥 Downloading: {url}")
        
        # Set headers to mimic a real browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Make the HTTP request
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raises exception for bad status codes
        
        print(f"✅ Successfully downloaded {len(response.text)} characters")
        return response.text
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {url}: {str(e)}")
        return ""


def clean_html_content(html: str) -> BeautifulSoup:
    """
    Cleans HTML by removing unwanted elements like navigation, ads, scripts.
    
    Args:
        html (str): Raw HTML content
        
    Returns:
        BeautifulSoup: Cleaned BeautifulSoup object
        
    What gets removed:
    - Navigation menus
    - Scripts and styles
    - Advertisements
    - Headers and footers
    - Sidebar content
    """
    print("🧹 Cleaning HTML content...")
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove definitely unwanted elements (but be more conservative)
    for element in soup(['script', 'style']):
        element.decompose()
    
    # Only remove obvious navigation/ads, not all elements with these classes
    unwanted_selectors = [
        'topnav', 'sidenav', 'sidebar', 'breadcrumb',
        'advertisement', 'banner', 'social-share'
    ]
    
    for selector in unwanted_selectors:
        # Remove by class (only exact matches)
        for element in soup.find_all(class_=selector):
            element.decompose()
        # Remove by id (only exact matches)
        for element in soup.find_all(id=selector):
            element.decompose()
    
    # Remove specific navigation elements but preserve content divs
    for element in soup.find_all(['header', 'footer']):
        element.decompose()
    
    print(f"✅ Cleaned HTML - removed navigation and ads")
    return soup


def extract_text_content(soup: BeautifulSoup, url: str = "") -> str:
    """
    Extracts clean, readable text from the cleaned HTML with enhanced site-specific extraction.
    
    Args:
        soup (BeautifulSoup): Cleaned BeautifulSoup object
        url (str): Source URL for site-specific extraction
        
    Returns:
        str: Clean text content
    """
    print("📝 Extracting text content...")
    
    # Enhanced site-specific extraction
    if 'w3schools.com' in url.lower() and 'keywords' in url.lower():
        return extract_w3schools_keywords(soup)
    elif 'w3schools.com' in url.lower():
        return extract_w3schools_content(soup)
    
    # Try to find main content areas with site-specific selectors
    main_content = (
        soup.find('main') or 
        soup.find('article') or 
        soup.find('div', class_='w3-col l10 m12') or  # W3Schools specific main content
        soup.find('div', id='main') or
        soup.find('div', class_=lambda x: x and 'content' in str(x).lower() if x else False) or
        soup.find('div', class_=lambda x: x and 'tutorial' in str(x).lower() if x else False)
    )
    
    if main_content:
        # Remove navigation within main content but keep educational content
        for nav in main_content.find_all(['nav', 'aside']):
            nav.decompose()
        # Remove obvious ads but keep content
        for ad in main_content.find_all(class_=lambda x: x and ('ad' in str(x).lower() or 'banner' in str(x).lower()) if x else False):
            ad.decompose()
        text = main_content.get_text()
    else:
        # Fallback: focus on paragraphs, headings, and code examples
        content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'pre', 'code', 'blockquote'])
        text = ' '.join([elem.get_text() for elem in content_elements])
        
        # If still no good content, get all text with minimal filtering
        if len(text.strip()) < 100:
            text = soup.get_text()
    
    # Clean the text but don't be too aggressive
    # 1. Remove excessive repetition but keep educational content
    text = re.sub(r'(Tutorial\s+){4,}', 'Tutorial ', text, flags=re.IGNORECASE)  # Remove excessive "Tutorial"
    text = re.sub(r'(Examples?\s+){4,}', 'Examples ', text, flags=re.IGNORECASE)  # Remove excessive "Examples"
    text = re.sub(r'(Reference\s+){4,}', 'Reference ', text, flags=re.IGNORECASE)  # Remove excessive "Reference"
    
    # 2. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Remove multiple newlines
    text = re.sub(r'\n+', ' ', text)
    
    # 3. Strip leading/trailing whitespace
    text = text.strip()
    
    print(f"✅ Extracted {len(text)} characters of clean text")
    return text


def split_into_chunks(text: str, min_chars: int = 300, max_chars: int = 600, overlap_chars: int = 80, url: str = "") -> List[str]:
    """
    Splits text into smaller, content-aware chunks for better retrieval precision.
    
    Args:
        text (str): The text to split
        min_chars (int): Minimum characters per chunk (default: 300)
        max_chars (int): Maximum characters per chunk (default: 600)
        overlap_chars (int): Characters to overlap between chunks (default: 80)
        url (str): Source URL for content-aware splitting
        
    Returns:
        List[str]: List of text chunks
        
    Enhanced chunking strategy:
    - Smaller chunks for better precision (300-600 chars instead of 400-800)
    - Content-aware splitting for structured data
    - Smart boundary detection for technical articles
    - Preserve keyword-description pairs
    """
    print(f"Splitting text into chunks ({min_chars}-{max_chars} characters with {overlap_chars} char overlap)...")
    
    # Content-aware chunking for keyword pages (very specific detection)
    is_keyword_page = ('keywords' in url.lower() or 
                      ('reference' in url.lower() and text.count(':') > 5 and text.count('\n') > 10))
    if is_keyword_page:
        return split_keyword_content(text, min_chars, max_chars)
    
    # For structured content, ALWAYS try to split by logical sections (even if small)
    section_markers = ['\n\n', '===', '---', 'Chapter', 'Section', 'Layer:', 'Types of', 'What is', 'Role:', 'Function:']
    if any(marker in text for marker in section_markers):
        result = split_by_sections(text, min_chars, max_chars, overlap_chars)
        # If section splitting worked (created multiple chunks), use it
        if len(result) > 1:
            return result
    
    # Simple, reliable chunking for all content
    if len(text) <= max_chars:
        print(f"Text is small ({len(text)} chars), creating 1 chunk")
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_chars, len(text))
        
        # Try to break at sentence end within last 100 chars
        if end < len(text):
            search_start = max(end - 100, start)
            for i in range(end - 1, search_start - 1, -1):
                if text[i] in '.!?\n' and i < len(text) - 1:
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        
        start = end - overlap_chars if end < len(text) else end
        
        # Safety check to avoid infinite loops
        if start >= len(text):
            break
    
    print(f"Created {len(chunks)} chunks")
    
    # Debug: show chunk sizes
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}: {len(chunk)} characters")
    
    return chunks


def split_keyword_content(text: str, min_chars: int, max_chars: int) -> List[str]:
    """
    Split keyword-description content preserving keyword-description pairs.
    
    Args:
        text (str): Text containing keyword descriptions
        min_chars (int): Minimum chunk size
        max_chars (int): Maximum chunk size
        
    Returns:
        List[str]: List of chunks with preserved keyword pairs
    """
    chunks = []
    lines = text.split('\n')
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        line_size = len(line)
        
        # If adding this line would exceed max size, finalize current chunk
        if current_size + line_size > max_chars and current_chunk:
            if current_size >= min_chars:
                chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size + 1  # +1 for newline
    
    # Add final chunk if it has content
    if current_chunk and current_size >= min_chars:
        chunks.append('\n'.join(current_chunk))
    elif current_chunk and chunks:  # Merge small final chunk with previous
        chunks[-1] += '\n' + '\n'.join(current_chunk)
    
    return chunks if chunks else [text]


def split_by_sections(text: str, min_chars: int, max_chars: int, overlap_chars: int) -> List[str]:
    """
    Split text by logical sections while maintaining context.
    
    Args:
        text (str): Text to split
        min_chars (int): Minimum chunk size
        max_chars (int): Maximum chunk size
        overlap_chars (int): Overlap between chunks
        
    Returns:
        List[str]: List of section-based chunks
    """
    # Try to identify section boundaries with enhanced markers for technical content
    section_markers = [
        '\n\n\n',  # Triple newlines (major sections)
        '\n\n',    # Double newlines (paragraphs)
        '===',     # Horizontal rules
        '---',     # Dashes
        'Layer:',  # Technical section headers
        'Role:',   # Role descriptions
        'Function:', # Function explanations
        'Types of', # Type listings
        'What is',  # Definitions
        '. ',       # Sentence boundaries as last resort
    ]
    best_marker = None
    
    for marker in section_markers:
        if marker in text and len(text.split(marker)) > 1:
            best_marker = marker
            break
    
    if not best_marker:
        # Fallback to standard chunking
        return standard_chunk_split(text, min_chars, max_chars, overlap_chars)
    
    sections = text.split(best_marker)
    chunks = []
    current_chunk = ""
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # For technical content with clear markers, prefer separate chunks
        # Only combine sections if current section is too small and won't create a valid chunk
        if current_chunk and len(section) >= min_chars:
            # Both current chunk and new section are valid sizes - keep them separate
            chunks.append(current_chunk)
            current_chunk = section
        elif len(current_chunk) + len(section) + len(best_marker) <= max_chars:
            # Either current chunk is empty, section is small, or combined size fits - combine them
            current_chunk = current_chunk + (best_marker if current_chunk else "") + section
        else:
            # Adding section would exceed max_chars - finalize current and start new
            if len(current_chunk) >= min_chars:
                chunks.append(current_chunk)
            current_chunk = section
            
            # If single section is too large, split it
            if len(section) > max_chars:
                sub_chunks = standard_chunk_split(section, min_chars, max_chars, overlap_chars)
                chunks.extend(sub_chunks)
                current_chunk = ""
    
    # Add final chunk
    if current_chunk and len(current_chunk) >= min_chars:
        chunks.append(current_chunk)
    elif current_chunk and chunks:  # Merge small final chunk
        chunks[-1] += best_marker + current_chunk
    
    return chunks if chunks else [text]


def standard_chunk_split(text: str, min_chars: int, max_chars: int, overlap_chars: int) -> List[str]:
    """
    Standard chunking algorithm with sentence boundary detection.
    
    Args:
        text (str): Text to split
        min_chars (int): Minimum chunk size
        max_chars (int): Maximum chunk size
        overlap_chars (int): Overlap between chunks
        
    Returns:
        List[str]: List of chunks
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        
        if end >= len(text):
            chunk = text[start:]
            if len(chunk.strip()) > 100:
                chunks.append(chunk.strip())
            break
        
        # Find sentence boundary
        chunk_text = text[start:end]
        sentence_endings = []
        
        for i, char in enumerate(chunk_text[-200:], len(chunk_text)-200):
            if char in '.!?' and i < len(chunk_text) - 1:
                if i + 1 < len(chunk_text) and chunk_text[i + 1] in ' \n\t':
                    sentence_endings.append(i)
        
        if sentence_endings:
            actual_end = sentence_endings[-1] + 1
            chunk = text[start:start + actual_end].strip()
        else:
            words = chunk_text.split()
            if len(words) > 1:
                chunk = ' '.join(words[:-1])
            else:
                chunk = chunk_text
        
        if len(chunk.strip()) >= min_chars:
            chunks.append(chunk.strip())
            start += max(len(chunk) - overlap_chars, len(chunk) // 2)
        else:
            start += max_chars // 2
    
    return chunks


def verify_chunks(data_dir: str = "data") -> None:
    """
    Verify chunk quality and content extraction.
    
    Args:
        data_dir (str): Directory containing chunk files
    """
    print("🔍 Verifying chunk quality...")
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found")
        return
    
    chunk_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not chunk_files:
        print("❌ No chunk files found")
        return
    
    for i, filename in enumerate(chunk_files[:3]):  # Check first 3 files
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print(f"\n📄 Chunk {i+1}: {filename}")
        print(f"📏 Size: {len(content)} characters")
        
        # Extract main content (skip metadata)
        lines = content.split('\n')
        main_content = '\n'.join(lines[3:]) if len(lines) > 3 else content
        
        print(f"📄 Preview (first 200 chars): {main_content[:200]}...")
        
        # Check for important keywords
        keywords = ['def', 'class', 'if', 'while', 'for', 'True', 'False', 'or', 'and', 'return', 'import']
        found = [kw for kw in keywords if kw.lower() in content.lower()]
        print(f"✅ Found keywords: {found[:5]}{'...' if len(found) > 5 else ''}")
        
        # Check for structured content (keyword:description pairs)
        colon_lines = [line.strip() for line in main_content.split('\n') if ':' in line and len(line.strip()) < 100]
        if colon_lines:
            print(f"🔑 Found {len(colon_lines)} keyword-description pairs")
            for j, line in enumerate(colon_lines[:3]):
                print(f"   {j+1}. {line[:80]}...")


def save_chunks_to_files(chunks: List[str], source_url: str, data_dir: str = "data") -> List[str]:
    """
    Saves text chunks as individual files in the data directory.
    
    Args:
        chunks (List[str]): List of text chunks to save
        source_url (str): Original URL source (for filename)
        data_dir (str): Directory to save files in
        
    Returns:
        List[str]: List of saved file paths
    """
    print(f"💾 Saving {len(chunks)} chunks to files...")
    
    # Ensure data directory exists (cloud deployment fix)
    try:
        os.makedirs(data_dir, exist_ok=True)
        print(f"✅ Directory ensured: {data_dir}")
    except Exception as e:
        print(f"❌ Failed to create directory {data_dir}: {e}")
        return []
    
    # Create filename from URL
    parsed_url = urlparse(source_url)
    domain = parsed_url.netloc.replace('.', '_')
    path = parsed_url.path.replace('/', '_').replace('.html', '')
    
    saved_files = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Create filename: domain_path_chunk001.txt
            filename = f"{domain}{path}_chunk{i+1:03d}.txt"
            filepath = os.path.join(data_dir, filename)
            
            # Save chunk to file with better error handling
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Source: {source_url}\n")
                f.write(f"Chunk {i+1}/{len(chunks)}\n")
                f.write("-" * 50 + "\n\n")
                f.write(chunk)
            
            saved_files.append(filepath)
            
        except Exception as e:
            print(f"❌ Error saving chunk {i+1}: {e}")
            continue
    
    print(f"✅ Saved {len(saved_files)} chunk files")
    return saved_files


def process_documentation(urls: List[str]) -> Dict[str, List[str]]:
    """
    Main function to process multiple documentation URLs.
    
    Args:
        urls (List[str]): List of URLs to process
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping URLs to their chunk file paths
        
    This is the main function you'll call to ingest documentation.
    """
    print("🚀 Starting documentation ingestion process...")
    print(f"📋 Processing {len(urls)} URLs")
    
    results = {}
    
    for i, url in enumerate(urls, 1):
        print(f"\n📄 Processing URL {i}/{len(urls)}: {url}")
        
        try:
            # Step 1: Download the webpage
            html = download_webpage(url)
            if not html:
                print(f"❌ Skipping {url} - download failed")
                continue
            
            # Step 2: Clean the HTML
            soup = clean_html_content(html)
            
            # Step 3: Extract text with URL-aware extraction
            text = extract_text_content(soup, url)
            if len(text) < 100:
                print(f"❌ Skipping {url} - too little content")
                continue
            
            # Step 4: Split into chunks with improved strategy
            chunks = split_into_chunks(text, url=url)
            
            # Step 5: Save chunks to files
            saved_files = save_chunks_to_files(chunks, url)
            results[url] = saved_files
            
            print(f"✅ Successfully processed {url}")
            
            # Debug: verify chunk quality for first URL
            if i == 1 and 'keywords' in url.lower():
                verify_chunks()
            
            # Small delay to be respectful to servers
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Error processing {url}: {str(e)}")
            continue
    
    print(f"\n🎉 Ingestion complete! Processed {len(results)} URLs successfully")
    return results


def get_all_chunks() -> List[str]:
    """
    Retrieves all text chunks from the data directory.
    
    Returns:
        List[str]: List of all chunk file contents
        
    This function is used by other modules to access the ingested data.
    """
    data_dir = "data"
    chunks = []
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found")
        return chunks
    
    # Read all .txt files in data directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    chunks.append(content)
            except Exception as e:
                print(f"❌ Error reading {filepath}: {str(e)}")
    
    print(f"📚 Retrieved {len(chunks)} chunks from data directory")
    return chunks


# Example usage (this runs only when file is executed directly)
if __name__ == "__main__":
    # Example URLs - replace these with your documentation URLs
    sample_urls = [
        "https://docs.python.org/3/tutorial/",
        "https://docs.python.org/3/library/os.html"
    ]
    
    print("🧪 Running ingestion example with sample URLs...")
    results = process_documentation(sample_urls)
    
    # Display results
    for url, files in results.items():
        print(f"\n📄 {url}: {len(files)} chunks saved")