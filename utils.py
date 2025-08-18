import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
# from textsplit.tools import split_single


CHUNKSIZE = 200  # Maximum words per chunk


def chunk_text(text) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNKSIZE,
        chunk_overlap=20,
        length_function=lambda x: len(x.split()),
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def has_keyword(text, keywords):
    """
    Check if any keywords are found in the text and return the list of found keywords.
    Uses exact word matching (not substring matching).
    
    Args:
        text (str): The text to search in
        keywords (list): List of keywords to search for
        
    Returns:
        list: List of keywords found in the text (empty list if none found)
    """
    
    found_keywords = []
    for keyword in keywords:
        # Create a regex pattern with word boundaries for exact matching
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_keywords.append(keyword)
    
    return list(set(found_keywords))  # Remove duplicates and return as list


