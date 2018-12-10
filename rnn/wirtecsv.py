from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
import numpy as np
import nltk
import glob
import csv
import pandas as pd

# 读取文件至字符串形式： glob、with+open、
negnames = glob.glob('./train/neg/*.txt')
posnames = glob.glob('./train/pos/*.txt')
filenames = negnames + posnames
negdocuments = []
posdocuments = []
for filename in negnames:
    with open(filename, encoding='utf-8') as document:
        negdocuments.append(document.read())
for filename in posnames:
    with open(filename, encoding='utf-8') as document:
        posdocuments.append(document.read())

# 写入csv
negdataframe = pd.DataFrame({'text:' : negdocuments, 'label:' : 0})
posdataframe = pd.DataFrame({'text:' : posdocuments, 'label:' : 1})
dataframe = negdataframe + posdataframe

# 分词： nltk
words_list = []
for document in documents:
    words_list.append(nltk.word_tokenize(document))
