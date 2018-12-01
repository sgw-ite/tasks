#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metrics #内置四个评估方式
import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from LClassifier import LClassifier

#特征工程相关

#用pandas预处理
df = pd.read_csv('./train.csv')
del df['Name'] #如何用index一次删除？
del df['Ticket']
del df['PassengerId']

#鉴于客舱有大量缺失，这里先采用把它扔掉的方法（当然，可以根据有无缺失分类，也可以把有缺失作为单独一类
del df['Cabin']

#对于Age的处理：这里先采取平均值填充，之后尝试一下根据名字中的mr ms来预测的方法
for i in ['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']:
    df[i] = df[i].astype(float)
df['Age'] = df['Age'].fillna(value=29.7)
df = df.dropna(axis=0)

#预处理初步成功，切换到numpy数组，提取X_train,y_train，再做进一步处理
ar = np.array(df)
y_train = ar[:,0]
X_train = ar[:,1:]
y_train = np.array([float(i) for i in y_train])

#处理值只有分类意义而无大小意义的特征
_Pclass = X_train[:,0]
_Sex = X_train[:,1]
_Embarked = X_train[:,-1]
Pclass, Sex, Embarked = [], [], []
for i in _Pclass: #这里有没有办法可以不写三遍？= =
    if i == 3:
        Pclass.append([0.,0.,1.])
    if i == 2:
        Pclass.append([0.,1.,0.])
    if i == 1:
        Pclass.append([1.,0.,0.])
for i in _Sex:
    if i == 'male':
        Sex.append([0.,1.])
    else:
        Sex.append([1.,0.])
for i in _Embarked:
    if i == 'S':
        Embarked.append([0.,0.,1.])
    elif i == 'C':
        Embarked.append([0.,1.,0.])
    else:
        Embarked.append([1.,0.,0.])
        
#把这三列的处理结果加入
X_train = np.hstack([X_train[:,2:6], np.array(Pclass).reshape(X_train.shape[0],-1),
                    np.array(Sex).reshape(X_train.shape[0],-1), np.array(Embarked).reshape(X_train.shape[0],-1)])
X_train = X_train.astype(float) #哇果然有一步到位的方法。。熬测的时候蠢飞

#归一化 经过测试均值方差归一化要优于最值归一化
for i in range(X_train.shape[1]):
    X_train[:,i] = (X_train[:,i] - np.mean(X_train[:,i])) / np.std(X_train[:,i])
#for i in range(X_train.shape[1]):
#    X_train[:,i] = (X_train[:,i] - np.min(X_train[:,i])) / (np.max(X_train[:,i]) - np.min(X_train[:,i]))

#自此特征处理结束，开始进行数据划分


# In[2]:


X_train, y_train, X_test, y_test = tts.holdout(X_train, y_train)
#X_train, y_train, X_test, y_test = tts.bootstrapping(X_train, y_train)
#res = tts.crossvalidation(X_train, y_train, myclassifier=LogisticRegression,k=10, p=10, scoring=metrics.acc_score)
#自此，数据划分结束，开始训练模型


lclf = LClassifier()
lclf.fit(X_train, y_train)
y_pred = lclf.predict(X_test)
#自此模型训练完毕，开始获取评分


# In[3]:


print(metrics.acc_score(y_test, y_pred), metrics.recall(y_test, y_pred), 
      metrics.precisionrate(y_test, y_pred), metrics.F1_score(y_test, y_pred))


# In[9]:


#绘制roc
TPR_history = []
FPR_history = []
for i in range(10000):
    y_pred = lclf.predict(X_test, rate=(i * 0.0001))
    TN, FN, TP, FP = metrics.confusion_metrix(y_pred, y_test)
    TPR = TP / (TP + FN)
    FPR = FP / (TN + FP)
    TPR_history.append(TPR)
    FPR_history.append(FPR)
print('TPR_history\'s shape:', np.array(TPR_history).shape)


# In[11]:


plt.plot(FPR_history, TPR_history)
plt.axis([0,1,0,1])
plt.show
#怎么这么不光滑= =


# In[15]:


#计算AUC
AUC = 0.5 * sum(([((TPR_history[i] + TPR_history[i+1]) * (FPR_history[i] - FPR_history[i+1])) for i in range(len(FPR_history) - 1)]))
print('AUC = ',AUC)

