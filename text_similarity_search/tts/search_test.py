import numpy as np

from . import utils
from .search import VectorSimilaritySearch, MinHashSimilaritySearch


def test_vector_similarity_search():
    d = 8  # dimension
    nb = 10  # database size
    np.random.seed(1234)  # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xq = xb

    vss = VectorSimilaritySearch(xb, batch_size=200, metric="L2", num_candidate_neighbors=4, num_actual_neighbors=2,
                                 select_mode='sample')
    I = vss.search(xq)

    print(I)


def test_minhash_similarity_search():
    source_data = utils.read_list('source_data.txt')
    source_data_lemma = utils.read_list('source_data.txt_lemma')

    mss = MinHashSimilaritySearch(source_data, source_data_lemma,
                                  num_candidate_neighbors=20, num_actual_neighbors=10,
                                  num_threads=4, remove_self_result=True,
                                  select_mode='best', jaccard_threshold=0.4)

    pairs = mss.create_pair_dataset()
    pairs_str = [(source_data[i], source_data[j]) for i, j in pairs]

    from pprint import pprint
    pprint(pairs_str)


def test_minhash_similarity_search_query():
    source_data = utils.read_list('source_data.txt')
    source_data_lemma = utils.read_list('source_data.txt_lemma')

    query = source_data[:20]
    query_lemma = source_data_lemma[:20]

    mss = MinHashSimilaritySearch(source_data, source_data_lemma,
                                  num_candidate_neighbors=20, num_actual_neighbors=10,
                                  num_threads=4, remove_self_result=True, batch_size=20,
                                  select_mode='best', jaccard_threshold=0.2)

    non_batch_result = mss.search((query_lemma, query))

    mss = MinHashSimilaritySearch(source_data, source_data_lemma,
                                  num_candidate_neighbors=20, num_actual_neighbors=10,
                                  num_threads=4, remove_self_result=True, batch_size=4,
                                  select_mode='best', jaccard_threshold=0.2, create_new_minhash=False)

    mss.batch_size = 4
    batch_result = mss.search((query_lemma, query))

    # pairs_str = [(source_data[i], source_data[j]) for i, j in pairs]

    assert np.all(utils.flatten_result(batch_result) == utils.flatten_result(non_batch_result))

    # np.save('tmp22.npy', results)


if __name__ == '__main__':
    # test_vector_similarity_search()
    # test_minhash_similarity_search()
    test_minhash_similarity_search_query()
