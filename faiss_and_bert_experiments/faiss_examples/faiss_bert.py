# coding: utf-8
import os
import json
import numpy as np
from tqdm import tqdm
import faiss

files = list(map(lambda x: os.path.join(".",x), os.scandir("enc_outputs/")))
files[0]
data = []

for file in tqdm(files):
    with open(file) as f:
        tmp = json.load(f)
        tmp = np.mean([d['layers'][0]['values'] for d in tmp['features']], axis=0)
        data.append(tmp)

index = faiss.IndexFlatL2(data[0].shape[0])
index.add(np.array(data, dtype=np.float32))
faiss.write_index(index, "faiss_bert.faiss")
