import json
from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class corpus_source:
  id: str
  title: str
  path: str

def load_corpus(subject: str) -> List[corpus_source]:
    registry_path = Path("data/corpus_registry.json")
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Registry file not found at {registry_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode JSON from {registry_path}")

    if subject not in registry:
        raise ValueError(f"Unknown corpus subject {subject}")

    try:
        sources = registry[subject]["sources"]
        return [
            corpus_source(
                id=s["id"],
                title=s["title"],
                path=s["path"]
            )
            for s in sources
        ]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid format for sources in subject '{subject}': {e}")
