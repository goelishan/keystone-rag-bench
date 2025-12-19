import numpy as np
from typing import List
from abc import ABC,abstractmethod

class Base_Embedder(ABC):

  @abstractmethod
  def embed_texts(self, texts: List[str])->np.ndarray:
    #returns a np array of size(N,D)
    pass
  
  @staticmethod
  def L2_normalize(vectors:np.ndarray)->np.ndarray:
    #L2 normalize embeddings for cosime similarity

    norms=np.linalg.norm(vectors,axis=1,keepdims=True)
    if np.any(norms==0):
      raise ValueError("Zero embeddings encountered during normalization")

    return vectors,norms 