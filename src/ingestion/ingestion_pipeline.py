import json
from pathlib import Path

from core.constants import (
    SUBJECT_CLOUD_DEVOPS_DOCS_V1,
    CHUNKS_FILENAME
)
from ingestion.corpus import load_corpus
from ingestion.loaders.text_loader import load_pdf_by_page
from ingestion.chunkers.recursive_chunker import recursive_chunker


def run_ingestion(subject: str):
    out_dir = Path("data/processed") / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / CHUNKS_FILENAME

    chunker = recursive_chunker()
    all_chunks = []

    for source in load_corpus(subject):
        for page, text in load_pdf_by_page(source.path):
            chunks = chunker.chunk(
                text=text,
                subject=subject,
                source=source,
                page=page
            )
            all_chunks.extend(chunks)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"{subject} ingested {len(all_chunks)} chunks!")


if __name__ == "__main__":
    run_ingestion(SUBJECT_CLOUD_DEVOPS_DOCS_V1)
