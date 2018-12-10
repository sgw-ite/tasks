import warnings
warnings.filterwarnings(action='ignore', module='gensim', category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tudata
import torch.optim as optim
import torchtext
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile, common_texts
import nltk
import glob

EPOCH = 10
BATCH_SIZE = 1

# 读取文件至字符串形式： glob、with+open、
negnames = glob.glob('./train/neg/*.txt')
posnames = glob.glob('./train/pos/*.txt')
negdocuments = []
posdocuments = []
for negname in negnames:
    with open(negname, encoding='utf-8') as negdocument:
        negdocuments.append(negdocument.read())
for posname in posnames:
    with open(posname, encoding='utf-8') as posdocument:
        posdocuments.append(posdocument.read())

# 分词： nltk
negwords_list = []
for negdocument in negdocuments:
    negwords_list.append(nltk.word_tokenize(negdocument))
poswords_list = []
for posdocument in posdocuments:
    poswords_list.append(nltk.word_tokenize(posdocument))

# 读取训练好的模型
model = Word2Vec.load("./word2vec/word2vec.model")

# 建立数据集


class MYDATA():

    def __init__(self, negwords_list, poswords_list):
        self.datas = negwords_list + poswords_list
        self.targets = [0] * 12500 + [1] * 12500

    def __getitem__(self, index):
        datas = self.datas[index]
        targets = self.targets[index]
        return datas, targets

    def __len__(self):
        return len(self.datas)

# 建立rnn网络


class ViewClassifier(nn.Module):

    def __init__(self, embadding_dim=100, hidden_dim=100,
                 target_size=2, layer_num=1):
        super(ViewClassifier, self).__init__()
        self.embadding_dim = embadding_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.target_size = target_size
        self.lstm = nn.LSTM(self.embadding_dim, self.hidden_dim,
                            self.layer_num, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 2)

    def forward(self, input):
        x, (hn, cn) = self.lstm(input)
        x = x.view(-1, 100)
        x = self.fc(x)
        x = nn.LogSoftmax(x)
        return x, (hn, cn)

mydata = MYDATA(negwords_list, poswords_list)
mydataloader = tudata.DataLoader(mydata, batch_size=BATCH_SIZE, shuffle=True)
myclassifier = ViewClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(myclassifier.parameters(), lr=0.001)

for epoch in range(EPOCH):
    hn, cn = torch.zeros(1, -1, BATCH_SIZE), torch.zeros(1, -1, BATCH_SIZE)
    for i, data in enumerate(mydataloader):
        inputs, label = data
        label = label.float()
        optimizer.zero_grad()
        output, (hn, cn) = myclassifier(inputs, (hn, cn))
        loss = criterion(output, label)
        optimizer.step()
        running_loss += loss.item()
        if i % 500 == 499:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

torchtext.