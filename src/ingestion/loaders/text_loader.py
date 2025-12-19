from pathlib import Path
from PyPDF2 import PdfReader


def load_pdf_by_page(pdf_path: str):
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(path)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            yield i + 1, text
