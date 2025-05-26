from flashtext import KeywordProcessor
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from textsplit.tools import split_single


CHUNKSIZE = 100  # Maximum words per chunk

# def chunk_text(text) -> list[str]:
#     words = text.split()
#     if len(words) <= CHUNKSIZE:
#         return [text]
    
#     # Split by newlines first
#     lines = text.split('\n')
#     chunks = []
#     current_chunk = []
#     current_word_count = 0
    
#     for line in lines:
#         line_words = line.split()
#         if current_word_count + len(line_words) > CHUNKSIZE:
#             if current_chunk:
#                 chunks.append(' '.join(current_chunk))
#             current_chunk = line_words
#             current_word_count = len(line_words)
#         else:
#             current_chunk.extend(line_words)
#             current_word_count += len(line_words)
    
#     if current_chunk:
#         chunks.append(' '.join(current_chunk))
    
#     return chunks # list of strings

def chunk_text(text) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNKSIZE,
        chunk_overlap=0,
        length_function=lambda x: len(x.split()),
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)




def has_keyword(text, keywords):
    kp = KeywordProcessor()
    kp.add_keywords_from_list(keywords)
    extracted = kp.extract_keywords(text)
    return any(extracted)