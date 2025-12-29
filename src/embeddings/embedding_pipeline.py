import json
import numpy as np
from pathlib import Path
from typing import List,Dict
import faiss

from core.constants import (SUBJECT_CLOUD_DEVOPS_DOCS_V1,CHUNKS_FILENAME)
from embeddings.openai_embedder import OpenAIEmbedder
from embeddings.normalize import l2_normalize

def load_chunks(subject:str)->List[Dict]:
    chunks_path=Path("data/processed")/subject/CHUNKS_FILENAME
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_faiss_index(vectors:np.ndarray)->faiss.Index:
    dim=vectors.shape[1]
    index=faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

    
def run_embeddings(subject:str):
    
    print(f">>> Running embedding pipeline for subject: {subject}")

    chunks=load_chunks(subject)

    texts=[c["text"] for c in chunks]
    metadata=[
        {
            "id":c["id"],
            **c["metadata"]
        } for c in chunks
    ]
    print(f">>> Loaded {len(texts)} chunks and {len(metadata)} metadata")

    embedder=OpenAIEmbedder()
    vectors=embedder.embed_documents(texts)

    print(f">>> Generated embeddings: shape={vectors.shape}")

    vectors=l2_normalize(vectors)

    index=build_faiss_index(vectors)

    out_dir = Path("data/processed") / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embeddings.npy", vectors)

    faiss.write_index(index, str(out_dir / "faiss.index"))

    with open(out_dir / "vector_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(">>> Embedding pipeline completed successfully")
    print(f">>> Vectors: {vectors.shape[0]}, Dim: {vectors.shape[1]}")

if __name__=="__main__":
    run_embeddings(SUBJECT_CLOUD_DEVOPS_DOCS_V1)
