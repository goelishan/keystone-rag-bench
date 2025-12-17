import json
from pathlib import Path
from dataclass import dataclass
from typing import List


@dataclass(frozen=True)
class corpus_source:
  name: str
  path: str
  description: str

def get_corpus(subject: str) -> List[corpus_source]:
  
  registry_path=Path("/content/keystone-rag-bench/docs/corpus_registry.json")

  with open(registry_path,"r",encoding="utf-8") as f:
    registry=json.load(f)

    if subject not in registry:
      raise ValueError(f"Unknown corpus subject {subject}")

    return[
      corpus_source(
        id=s["id"],
        title=s["title"],
        path=s["path"]
      )
      for s in registry[subject]["sources"]
    ]