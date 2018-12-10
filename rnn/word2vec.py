from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
import numpy as np
import torch
import torch.nn as nn
import nltk
import gensim.models
import glob

# 读取文件至字符串形式： glob、with+open、
negnames = glob.glob('./train/neg/*.txt')
posnames = glob.glob('./train/pos/*.txt')
filenames = negnames + posnames
documents = []
for filename in filenames:
    with open(filename, encoding='utf-8') as document:
        documents.append(document.read())

# 分词： nltk
words_list = []
for document in documents:
    words_list.append(nltk.word_tokenize(document))

path = get_tmpfile('word2vec.model')

model = Word2Vec(words_list, size=100, window=5, min_count=1, workers=4)
model.save('./word2vec/word2vec.model')
