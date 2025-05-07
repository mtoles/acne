CHUNKSIZE = 500  # Maximum words per chunk

def chunk_text(text):
    words = text.split()
    if len(words) <= CHUNKSIZE:
        return [text]
    
    # Split by newlines first
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for line in lines:
        line_words = line.split()
        if current_word_count + len(line_words) > CHUNKSIZE:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = line_words
            current_word_count = len(line_words)
        else:
            current_chunk.extend(line_words)
            current_word_count += len(line_words)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


