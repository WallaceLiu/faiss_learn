import sys

import faiss
import numpy as np
# import psutil
import os
import platform
import time


def memory_usage_psutil():
    # return the memory usage in MB
    # return psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)
    os.system('ps -l -p %d' % os.getpid())


def generate_faiss_index(output_file, size=10000):
    if os.path.exists(output_file):
        os.unlink(output_file)
    vector_dimension = 1024
    database_size = size
    search_data = np.random.random((database_size, vector_dimension)).astype("float32")
    quantizer = faiss.IndexFlatL2(vector_dimension)
    index = faiss.IndexIVFFlat(quantizer, vector_dimension, 1)
    index.train(search_data)
    index.add(search_data)


    faiss.write_index(index, output_file)


def generate_file_name(ind):
    return "test_file_%s.index" % ind

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # generate 2 index files with specified size
        size1 = int(sys.argv[1])
        size2 = int(sys.argv[2])
        generate_faiss_index(generate_file_name(0), size1)
        generate_faiss_index(generate_file_name(1), size2)
    else:
        print("python version:%s " % platform.python_version())
        print("faiss version: %s" % faiss.__version__)
        time.sleep(3)
        # benchmark
        print("Start memory: %s" % memory_usage_psutil())
        index1 = faiss.read_index(generate_file_name(0))
        print("first index read: %s" % memory_usage_psutil())
        index1.own_invlists = True
        index1 = None
        print("dereference first index: %s" % memory_usage_psutil())
        index2 = faiss.read_index(generate_file_name(1))
        print("second index read: %s" % memory_usage_psutil())
        index2 = None
        print("dereference second index: %s" % memory_usage_psutil())
        # wait before
        for i in range(100):
            p=i*i
        time.sleep(5)
        print("final: %s" % memory_usage_psutil())