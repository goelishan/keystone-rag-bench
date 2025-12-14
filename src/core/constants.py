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


