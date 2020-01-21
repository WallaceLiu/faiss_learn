import fire

from text_similarity_search.tts.search import VectorSimilaritySearch, MinHashSimilaritySearch

all_methods = [VectorSimilaritySearch, MinHashSimilaritySearch]


def find_method_class(method_name):
    for cls in all_methods:
        if cls.__name__.lower().find(method_name.lower()) != -1:
            return cls


def create_method_instance(method_name, source_path, kwargs):
    method_cls = find_method_class(method_name)

    source_data = method_cls.read_source_data(source_path)
    if not isinstance(source_data, tuple):
        source_data = (source_data,)

    ss = method_cls(*source_data, **kwargs)

    return ss


class Runner:
    def __init__(self, method, source_path):
        self._source_path = source_path
        self._method_name = method

    def search(self, query_path, output_path, **kwargs):
        method_instance = create_method_instance(self._method_name, self._source_path, kwargs)

        query = find_method_class(self._method_name).read_query(query_path)
        result = method_instance.search(query)
        method_instance.save_search_result(result, output_path)

    def create_pair_dataset(self, output_path, **kwargs):
        method_instance = create_method_instance(self._method_name, self._source_path, kwargs)

        pairs = method_instance.create_pair_dataset()
        method_instance.save_pair_dataset(pairs, output_path)


if __name__ == '__main__':
    fire.Fire(Runner)
