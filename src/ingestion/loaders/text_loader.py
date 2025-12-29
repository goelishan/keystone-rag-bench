from pathlib import Path
from pypdf import PdfReader


def load_pdf_by_page(pdf_path):
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(path)

    page_count = len(reader.pages)
    print(f">>> PDF page count ({path.name}): {page_count}")

    if page_count == 0:
        raise ValueError(f"PDF contains no pages: {path}")

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
        except Exception:
            continue

        if not text or not text.strip():
            continue

        yield i, text
