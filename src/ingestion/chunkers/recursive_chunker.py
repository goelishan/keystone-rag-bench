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
        """
        Splits the input text into overlapping chunks of specified size 
        and attaches relevant metadata to each chunk.

        Args:
            text (str): The input text to be chunked.
            subject (str): Corpus subject for metadata.
            source (Any): Source object containing 'id' and 'title'.
            page (int): Page number within the source.

        Returns:
            List[dict]: List of chunk dictionaries with metadata.
        """
        if not isinstance(text, str):
            raise ValueError("Input 'text' must be a string.")
        if not text.strip():
            return []

        text_len = len(text)
        chunks = []
        start = 0
        idx = 0
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
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
                        META_END_CHAR: end
                    }
                })
                idx += 1
            start += step

        return chunks
