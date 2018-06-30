# -*- coding: utf8 -*-

from __future__ import print_function
from __future__ import absolute_import
from embeddingmachine.models.weightedsum import TFIDFSum 

tfidfsum = TFIDFSum(word_embedding_path="data/word_rep")
model = tfidfsum.build()

# parameter of fit is iterator of documents, such as file, list, set etc.
# each document consists of words separated by blank.
with open("data/doc_data") as f:
  model.fit(f)

#optional. If load_word_embedding is not explictly called, predict will call it
#model.load_word_embedding()
vec = model.predict("paper model proposed")
print(vec)

# save and load
model.save_weights("tfidfsum_model")

tfidfsum_cpy = TFIDFSum(word_embedding_path="data/word_rep")
model_cpy = tfidfsum_cpy.build()
model_cpy.load_weights("tfidfsum_model")
vec_test = model_cpy.predict("paper model proposed")
print(vec_test)
