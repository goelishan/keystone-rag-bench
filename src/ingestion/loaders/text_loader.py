from typing import Iterator, Tuple
from pathlib import Path
from PyPDF2 import PdfReader


def load_pdf_by_page(pdf_path: str | Path) -> Iterator[Tuple[int, str]]:
  pdf_path = Path(pdf_path)

  if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

  reader = PdfReader(str(pdf_path))

  for page_number, page in enumerate(reader.pages, start=1):
    try:
      text = page.extract_text()
        # Explicit failure visibility
    except Exception as e:
      raise RuntimeError(
        f"Failed to extract text from {pdf_path.name}, page {page_number}"
    ) from e

      if text and text.strip():
          yield page_number, text