from core.constants import (
    META_SUBJECT,
    META_SOURCE_ID,
    META_SOURCE_TITLE,
    META_PAGE,
    META_CHUNK_INDEX,
    META_START_CHAR,
    META_END_CHAR
)


class recursive_chunker:

    def __init__(self, chunk_size=500, chunk_overlap=120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text, subject, source, page):
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "id": f"{subject}_{source.id}_p{page}_c{idx}",
                    "text": chunk_text,
                    "metadata": {
                        META_SUBJECT: subject,
                        META_SOURCE_ID: source.id,
                        META_SOURCE_TITLE: source.title,
                        META_PAGE: page,
                        META_CHUNK_INDEX: idx,
                        META_START_CHAR: start,
                        META_END_CHAR: min(end, len(text))
                    }
                })
                idx += 1

            
            start += self.chunk_size - self.chunk_overlap

        return chunks
