import sys
import argparse
import faiss
sys.path.append(".")


from findexer import *
from fsearcher import *
from faissconfig import *


def test_indexer_trainer():

    try:
        index = faiss.read_index(INDEX_PATH)
        print(index.is_trained)
        assert index.ntotal > 0
    except:
        i = IndexerTrain()
        index_total = i.indexer()
        index = faiss.read_index(INDEX_PATH)
        assert index.ntotal > 0
    
def test_searcher():
    fs = FaissSearch()
    result = fs.search_by_image('search/avena.JPG')
    assert result.get('file_path')
    #print(result, type(result))
    
    
    
    