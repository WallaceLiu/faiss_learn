#coding=utf8
import numpy as np
import math
import time
import sys
import faiss


input_file = sys.argv[1]
print('loading data...............')
embedding_features = np.loadtxt(input_file,delimiter=' ').astype('float32')
print('training data shape ',embedding_features.shape)
print(embedding_features.dtype)

#building faiss and test
print('building faiss................')

d = 200
nlist = 512

quantizer = faiss.IndexFlatIP(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

ngpus = faiss.get_num_gpus()
print("number of GPUs:", ngpus)
gpu_index = faiss.index_cpu_to_all_gpus(index)

train_begin_time = time.time()
gpu_index.train(embedding_features)
train_end_time = time.time()
print('train cost time',train_end_time - train_begin_time)

#save index
print('writing faiss index..............')
gpu_index.add(embedding_features)
cpu_index = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(cpu_index, input_file + '_faiss_index')
print('finished.................')
