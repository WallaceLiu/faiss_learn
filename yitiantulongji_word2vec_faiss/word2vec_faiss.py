#   section_30.txt section_34.txt
# s01.txt        section_03.txt section_07.txt section_11.txt section_15.txt section_19.txt section_23.txt section_27.txt section_31.txt section_35.txt
# s02.txt        section_04.txt section_08.txt section_12.txt section_16.txt section_20.txt section_24.txt section_28.txt section_32.txt section_36.txt
# section_01.txt section_05.txt section_09.txt section_13.txt section_17.txt section_21.txt section_25.txt section_29.txt section_33.txt

file_path = ['data/section_02.txt',
             'data/section_06.txt',
             'data/section_10.txt',
             'data/section_14.txt',
             'data/section_18.txt',
             'data/section_22.txt',
             'data/section_26.txt',
             ]

# import os
#
# file_path = 'data/'
# doc_path = 'data/doc'

# dirs = os.listdir(file_path)
# ds = []
# for d in dirs:
#     if d.endswith('txt'):
#         ds.append(d)
#
#
# doc_arr = []
# for f in ds:
#     p = os.path.join(file_path, f)
#     with open(p, 'r') as fin:
#         lines = fin.readlines()
#         c = ' '.join(lines).replace('\n', ' ')
#         doc_arr.append(c)
#
# fou = open(doc_path, 'w', encoding='UTF-8')
# for d in doc_arr:
#     print(d, file=fou)
# fou.close()
#
# print(ds)

import jieba
import gensim.models.word2vec as w2v
from gensim import utils
import os

sentences = w2v.LineSentence('data/doc')
model = w2v.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
model.wv.get_vector('')