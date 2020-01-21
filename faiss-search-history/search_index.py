#!/bin/bash/python

import sys
import base64
import requests
import faiss
import numpy as np
import time
import os 

def load_kv_index(filename):
  index = {}
  for line in open(filename, 'r'):
    items = line.strip().split('\t')
    idx = int(items[1])
    key = items[0]
    index[idx] = key
  return index

def load_emb(filename, id_index):
  data = {}
  idx = 0
  for line in open(filename, 'r'):
    items = line.strip().split(' ')
    vector = []
    for val in items:
      vector.append(float(val))
    data[id_index[idx]] = np.array(vector)
    idx += 1
  return data

begin = time.time()

faiss_index_file = sys.argv[1]
id_index_file = sys.argv[2]
emb_data_file = sys.argv[3]

faiss_index = faiss.read_index(faiss_index_file)
print("load faiss index...")
id_index = load_kv_index(id_index_file)
print("load id index...")
emb_index = load_emb(emb_data_file, id_index)
print("load embedding index...")

cells_ser = 100
top_k = 200
dim = 200

res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 1, faiss_index)

gpu_index.nprobe = cells_ser
 
end = time.time()

print('tiem cost of load model: ' + str(end - begin))

dirname = os.path.dirname(emb_data_file)
basename = os.path.basename(emb_data_file)
result_file = open(os.path.join(dirname,'{}.result_top{}'.format(basename.partition('.')[0],top_k)),'w')

#while True:
#  item = raw_input()
#  if item not in emb_index:
#    print("[ERROR] item is not in dict...")
#    continue
#   
#  print("query: " + item)
k = 0
emb_line = len(emb_index)
for item in emb_index:
  k += 1
  begin = time.time()
  embedding_features = emb_index[item]
  embedding_features = np.reshape(embedding_features,(1,dim)).astype('float32')
  dis, induces = gpu_index.search(embedding_features,top_k)
  #print("induces: ",induces)
  result_file.write(str(item) + '\t')
  for i in range(top_k):
    if int(induces[0][i]) < 0:
      continue
    #begin = time.time()
    if i == top_k-1:
      result_file.write(id_index[int(induces[0][i])] + '\002' + str(dis[0][i]))
    else:
      result_file.write(id_index[int(induces[0][i])] + '\002' + str(dis[0][i]) + '\001')
    #result = "\t" + str(i) + '\t' + id_index[int(induces[0][i])] + '\t' + str(dis[0][i])
    #print("result: ",result)
    #end = time.time()

    #print('time cost of search:' + str(end - begin))
  result_file.write('\n')
  if k % 10000 == 0: 
    print('processed {} / {}'.format(k,emb_line))
