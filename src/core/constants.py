from enum import Enum

class RetrievalMode(str, Enum):
  DENSE="dense"
  HYBRID="hybrid"

class TelemetryStage(str, Enum):
  RETRIEVAL="retrieval"
  RERANKING="reranking"
  CONTEXT="context"
  GENERATION="generation"
  EVALUATION="evaluation"

class MetaDataKeys(str, Enum):
  SOURCE="source"
  CHUNK_ID="chunk_id"
  SECTION="section"


## -------- Subject IDs --------
SUBJECT_CLOUD_DEVOPS_DOCS_V1 = "cloud_devops_docs_v1"

# -------- Chunking --------
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 120

# -------- Metadata keys (internal contract) --------
META_SUBJECT = "subject"
META_SOURCE_ID = "source_id"
META_SOURCE_TITLE = "source_title"
META_PAGE = "page"
META_CHUNK_INDEX = "chunk_index"
META_START_CHAR = "start_char"
META_END_CHAR = "end_char"

# -------- Filenames --------
CHUNKS_FILENAME = "chunks.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
METADATA_FILENAME = "metadata.json"
FAISS_INDEX_FILENAME = "faiss.index"
