import faiss
from typing import List, Tuple

from embeddings import EmbeddingEncoder, EmbeddingType

DEFAULT_LOC = "index_cache/"

class IndexManager():
    
    def __init__(self):
        self.indices: List[SemanticIndex] = dict()
        self.index_names: List[str] = list(self.indices.keys())
        
    def create_new_index(self, 
                         index_name: str,
                         embedding_dim: int,
                         embedding_type: EmbeddingType):
        index = SemanticIndex(index_name, embedding_dim, embedding_type)
        self._add_index_to_manager(index_name, index)
        
            
    def _add_index_to_manager(self, index_name: str, index: SemanticIndex):
        self.indices[index_name] = index
        self.index_names.append(index_name)        
   
    
   def serialize_indices(self, location = DEFAULT_LOC) -> str:
       saved_index_names = list()
       
       for index_name:str , index: SemanticIndex in self.indices.items():
           
           
class SemanticIndex(object):
    
    def __init__(self, 
                 index_name: str,
                 embeddin_dim: int,
                 embedding_type: str):
        
        self.index_name = index_name
        self.last_id = 0
        self.index_map = dict()
        self.encoder = encoder
        self.embeddin_dim = embeddin_dim
        self.embeddin_type = embedding_type
        self.index: faiss.IndexIDMap = faiss.IndexIDMap(faiss.IndexFlatIP(embeddin_dim))

    def search_index(self, 
                     query: str, 
                     n_results: int = 4) -> List[Tuple[str, float]]:
        """ Method searches semantic index and returnneed to pass an np array to faisss a list retrieved items and
            similarity scores. The similarity score is how semantically similar
            the retrieved item is to the query. """     
        # 1. encode query
        encoded_query = self.encoder.encode_single_item(query)
        # 2. normalize query
        faiss.normalize_L2(encoded_query)
        # 3. Search
        results: Tuple[np.array, np.array] = self.index.search(encoded_query, 
                                                               n_results)
        # 4. Zip results 
        results = [(self.index_map[vec_id], round(score, 3)) 
                   for score, vec_id in zip(list(results[0][0]), 
                                            list(results[1][0]))]         
        return results

    def index_item(self, item: str, id_: int = -1):
        """ Update semantic index with an encoded item. Unencoded item is stored
            in index map property. Additionally, item id is derived from by
            incrementing the last used id by id by default. User can also 
            provide a custom integer id. """
        #  If user provides no id_, use sequentially generated id by index. 
        item_id = self.index_size if id_ == -1 else item_id
        item_id: np.array = np.array([item_id]) 
        if id_ == -1:
            self.last_id += 1
        
        # 1. Encode item and normalize
        encoded_item: np.array = self.encoder.encode([item])
        faiss.normalize_L2(encoded_item)
        
        # 2. Update semantic index
        self.index.add_with_ids(encoded_item, item_id)
        
        # 3. Update index map with unencoded item
        self.index_map[item_id] = item
    
    def index_multiple_items(self, items: List[str], ids: List[int] = None):
        """ Update semantic index with mulitple encoded items. Unencoded items
            are stored in index map property. """
        if ids is not None and len(items) != len(ids):
            raise ValueError(f"Error: Number of provided ids({len(ids)}) does"
                             f" not match length of items ({len(items)})")
        
        

    
    