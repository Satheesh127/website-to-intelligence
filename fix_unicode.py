#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fix Unicode characters in chroma_retrieval.py"""

import re

def fix_unicode_chars():
    file_path = "rag/chroma_retrieval.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define replacements
    replacements = [
        ('🗄️', '[DATABASE]'),
        ('📚', '[DOCS]'),
        ('🆕', '[NEW]'),
        ('💾', '[SAVE]'),
        ('🔍', '[SEARCH]'),
        ('🗑️', '[DELETE]'),
        ('📂', '[FOLDER]'),
        ('❌', '[ERROR]'),
        ('📄', '[DOC]'),
        ('❓', '[QUESTION]'),
        ('🧪', '[TEST]'),
    ]
    
    # Apply replacements
    for old, new in replacements:
        content = content.replace(old, new)
    
    # Remove any other Unicode characters (keep ASCII only)
    content = re.sub(r'[^\x00-\x7F]+', '[?]', content)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed Unicode characters in chroma_retrieval.py")

if __name__ == "__main__":
    fix_unicode_chars()