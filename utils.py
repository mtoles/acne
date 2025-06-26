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
    kp = KeywordProcessor()
    kp.add_keywords_from_list(keywords)
    extracted = kp.extract_keywords(text)
    return any(extracted)