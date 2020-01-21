from abc import ABC, abstractmethod
from enum import Enum, auto
import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer

class EmbeddingType(Enum):
    BERTSENT = auto()
    GLOVE = auto()
    
class EmbeddingEncoder(ABC):
    
    def __init__(self,
                 type_: EmbeddingType,
                 dimension: int):
        self.type = type_
        self.dim = dimension
        self.encoder = self._load_model()
    
    @abstractmethod
    def _load_model(self) -> object:
        raise NotImplementedError
   
    @abstractmethod
    def encode_single_item(self, text: str) -> np.array:
        raise NotImplementedError
    
    @abstractmethod
    def encode_multiple_items(self, texts: List[str]) -> np.array:
        raise NotImplementedError
    
     
class BertSentEncoder(EmbeddingEncoder):
    
    def __init__(self):
        super(BertSentEncoder, self).__init__(type = EmbeddingType.BERTSENT,
                                              dim = 768)
    
    def _load_model(self) -> SentenceTransformer:
        return SentenceTransformer('embedding_weights/bert-base-nli-mean-tokens.zip')
        
    def encode_single_item(self, text: str) -> np.array():
        return np.array(self.encoder.encode([text]))
    
    def encode_multiplee_items(self, texts: List[str]) -> np.array():
        return np.array(self.encoder.encode(texts))


         
    