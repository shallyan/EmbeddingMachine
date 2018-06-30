# -*- coding=utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import math
import numpy as np
from tqdm import tqdm
import json

def myLog2(x):
  return math.log(x) / math.log(2)

class TFIDFSum(object):
  def __init__(self, word_embedding_path):
    self.name = 'TFIDFSum'
    self.word_embedding_path = word_embedding_path 
    self.word_embedding = dict() 
    print('[{}] init done'.format(self.name), end='\n')

  def build(self):
    return self
  
  def compile(self, loss=None, optimizer=None):
    pass
  
  def fit(self, doc_data):
    self.word_idf = dict()
    doc_cnt = 0
    for doc in tqdm(doc_data):
      doc = doc.lower()
      doc_cnt += 1
      words_lst = doc.split()
      for word in set(words_lst):
        self.word_idf.setdefault(word, 0)
        self.word_idf[word] += 1 
    
    for word in self.word_idf:
      self.word_idf[word] = myLog2(doc_cnt * 1.0 / self.word_idf[word])

  def _normalize(self, vec):
    sum = 0.0 
    for e in vec:
      sum += e*e
    mode = math.sqrt(sum) 
    if mode > 1e-10:
      for i in range(len(vec)):
        vec[i] = vec[i] / mode
    return vec

  def load_word_embedding(self):
    print("Start to load word embedding")
    with open(self.word_embedding_path) as f:
      count_info = f.readline()
      all_word_num = int(count_info.split()[0])
      dim_num = int(count_info.split()[1])
      self.embedding_size = dim_num

      for line in tqdm(f):
        fields = line.split()
        word = fields[0].strip()
        vec = list()
        vec_dim = len(fields) - 1
        if vec_dim != dim_num:
          raise Exception('Error, word dim not match')
        for i in range(vec_dim):
          vec.append(float(fields[i+1].strip()))
        vec = self._normalize(vec)
        self.word_embedding[word] = np.array(vec)

    if all_word_num != len(self.word_embedding):
      raise Exception('Error, word num not match')
    print("Load word embedding done, word num: {}, embedding dim: {}".format(all_word_num, dim_num))

  def predict(self, doc):
    doc = doc.lower()
    if len(self.word_embedding) == 0:
      self.load_word_embedding()

    words_lst = doc.split()
    doc_vec = np.zeros(self.embedding_size)
    for word in words_lst:
      if word in self.word_embedding:
        word_weight = self.word_idf[word] if word in self.word_idf else 100.0
        doc_vec = doc_vec + self.word_embedding[word] * word_weight
    return doc_vec

  def save_weights(self, model_path):
    with open(model_path, 'w') as f:
      json.dump(self.word_idf, f)

  def load_weights(self, model_path):
    with open(model_path) as f:
      self.word_idf = json.load(f)
