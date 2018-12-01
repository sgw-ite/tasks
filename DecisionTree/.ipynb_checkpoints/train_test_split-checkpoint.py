#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np 


# In[ ]:


def holdout(X_train, y_train, rate = 0.7):
    np.random.seed(666)
    np.random.shuffle(X_train)
    np.random.seed(666)
    np.random.shuffle(y_train)
    _X_train = X_train[:(int(X_train.shape[0] * rate)), :]
    _X_test = X_train[(int(X_train.shape[0] * rate)):, :]
    _y_train = y_train[:(int(X_train.shape[0] * rate))]
    _y_test = y_train[(int(X_train.shape[0] * rate)):]
    return _X_train, _y_train, _X_test,  _y_test


# In[ ]:


def bootstrapping(X_train, y_train):
    _X_train = np.ones((1,X_train.shape[1]))
    _y_train = np.ones(1)
    X_test = X_train.copy()
    y_test = y_train.copy()
    ind = []
    for i in range(len(X_train)):
        _X_train = np.vstack([_X_train,(X_train[np.random.randint(0,(len(X_train) - 1))])])
        _y_train = np.append(_y_train.tolist(), y_train[np.random.randint(0,(len(X_train) - 1))].tolist())
    _X_train = _X_train[1:,:]
    _y_train = _y_train[1:]
    for i in range(len(_X_train)):
        for j in range(len(X_train)):
            sum = 0
            for k in range(X_train.shape[1]):
                if _X_train[i][k] == X_train[j][k]:
                    sum += 1
            if sum == X_train.shape[1]:
                ind.append(j)
    _X_test = np.delete(X_train,ind,0)
    _y_test = np.delete(y_train,ind,0)
    return _X_train, _y_train, _X_test, _y_test


# In[ ]:


def crossvalidation(X_train, y_train, k, p, myclassifier, scoring):
    '''因为返回的是很多组的X_train等（不想前面只返回一组，在主函数里引用也要重新写一段代码才可以利用，干脆就直接令他返回结果了'''
    ind = [i for i in range(k)]
    Score_history = []
    for j in range(p):
        np.random.seed(666)
        np.random.shuffle(X_train)
        np.random.seed(666)
        np.random.shuffle(y_train)
        num = X_train.shape[0] // k
        X = np.zeros((k*num,X_train.shape[1]))
        y = np.zeros(k*num)
        for i in range(k):
            X[i * num:(i+1) * num,:] = X_train[i * num:(i+1) * num,:]
            y[i * num:(i+1) * num] = y_train[i * num:(i+1) * num]
        #X[k-1] = X_train[(k-1) * num:, :]
        #y[k-1] = y_train[(k-1) * num:]
        clf = myclassifier()
        score_history = []
        
        for i in range(k-1):
            if i == 0:
                X_train_ = X[(i+1)*num:,:]
                y_train_ = y[(i+1)*num:]
            else:
                X_train_ = np.vstack([X[:(i*num),:], X[(i+1)*num:,:]])
                y_train_ = np.hstack([y[:(i*num)], y[(i+1)*num:]])
            X_test = X[(i*num):(i+1)*num,:]
            y_test = y[(i*num):(i+1)*num]
            clf.fit(X_train_,y_train_)
            y_pred = clf.predict(X_test)
            score_history.append(scoring(y_pred, y_test))
        Score_history.append(np.mean(score_history))
    return np.mean(Score_history)

