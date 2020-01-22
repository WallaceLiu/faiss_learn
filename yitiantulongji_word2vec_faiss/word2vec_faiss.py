import jieba
import gensim.models.word2vec as w2v
from gensim import utils
import os

sentences = w2v.LineSentence('data/doc')
model = w2v.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
model.wv.get_vector('')
