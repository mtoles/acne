from flashtext import KeywordProcessor
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    
    Args:
        text (str): The text to search in
        keywords (list): List of keywords to search for
        
    Returns:
        list: List of keywords found in the text (empty list if none found)
    """
    kp = KeywordProcessor()
    kp.add_keywords_from_list(keywords)
    extracted = kp.extract_keywords(text)
    return list(set(extracted))  # Remove duplicates and return as list