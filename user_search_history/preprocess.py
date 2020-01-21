import sys
import math

input_file = sys.argv[1]

def str2vector(strv):
  vec = []
  for val in strv:
    vec.append(float(val))
  return vec

def vec2str(vec):
  strv = []
  for val in vec:
    strv.append(str(val))
  return strv

def dot(vec1, vec2):
  val = 0.0
  for i in range(len(vec1)):
    val += vec1[i] * vec2[i]
  return val

def devide(vec, num):
  vec1 = []
  for val in vec:
    vec1.append(val/num)
  return vec1

model_out = open(input_file + "_norm", 'w')
index_out = open(input_file  + "_index", 'w')
idx = 0
for line in open(input_file, 'r'):
  kv = line.split('\t')
  key = kv[0]
  items = kv[1].split('\001')
  vector = str2vector(items[0].split('\002'))
  norm = math.sqrt(dot(vector, vector))
  norm_vector = devide(vector, norm)
  model_out.write((' ').join(vec2str(norm_vector)) + '\n')
  index_out.write(key + '\t' + str(idx) + '\n')

  idx += 1
  if idx % 10000 == 0:
    print("parsing with ", idx)
model_out.close()
index_out.close()
