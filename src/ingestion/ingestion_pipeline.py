import json
from pathlib import Path

from core.constants import SUBJECT_EVERYDAY_PHYSICS_V1,CHUNKS_FILENAME
from ingestion.corpus import load_corpus
from ingestion.loaders.text_loader import load_pdf_by_page
from ingestion.chunkers.recursive_chunker import recursive_chunker


def run_ingestion(subject: str):
  out_dir=Path(f"/content/keystone-rag-bench/data/processed/{subject}")
  out_dir.mkdir(parents=True,exist_ok=True)

  chunker=recursive_chunker()
  all_chunks=[]

  for source in load_corpus(subject):
    for page,text in load_pdf_by_page(source.Path):
      all_chunks.extend(
        chunker.chunk(
          text=text,
          subject=subject,
          source=source,
          page=page
        )
        )
  
  with open(out_dir,"w",encoding="utf-8") as f:
    json.dump(all_chunks,f,indent=2,ensure_ascii=True)
  
  print(f"{subject} ingested {len(all_chunks)} chunks!")


if __name__=="__main__":
  run_ingestion(SUBJECT_EVERYDAY_PHYSICS_V1)