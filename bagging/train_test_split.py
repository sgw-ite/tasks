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
    _y_train = y_train[:(int(y_train.shape[0] * rate))]
    _y_test = y_train[(int(y_train.shape[0] * rate)):]
    return _X_train, _y_train, _X_test,  _y_test


# In[ ]:


def bootstrapping(X_train, y_train):
    Datas = np.concatenate([X_train,y_train.reshape(-1,1)],axis=1)
    ind1 = np.random.choice(range(0,X_train.shape[0]),size=X_train.shape[0])
    #for i in range(len(Datas)):
    #    train_Datas = np.vstack([train_Datas,Datas[np.random.randint(0,len(X_train))]])
    train_Datas = Datas[ind1]
    test_Datas = Datas.copy()
    #for i in range(len(train_Datas)):
    #    for j in range(len(Datas)):
     #       if (train_Datas[i]==Datas[j]).all():
    #            ind2.append(j)
    set1 = set(range(Datas.shape[0]))
    set2 = set(ind1)
    ind2 = list(set1 - (set1 & set2))
    test_Datas = test_Datas[ind]
    _X_train, _y_train = train_Datas[:,:(Datas.shape[1]-1)], train_Datas[:,-1].reshape(train_Datas.shape[0],)
    _X_test, _y_test = test_Datas[:,:(Datas.shape[1]-1)], test_Datas[:,-1].reshape(test_Datas.shape[0],)
    return _X_train, _y_train, _X_test, _y_test

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

