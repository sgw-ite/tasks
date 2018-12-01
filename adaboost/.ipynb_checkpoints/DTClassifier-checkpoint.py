#!/usr/bin/env python
# coding: utf-8


# $Gain(D,a) = Gini(D) - \sum_{v=1}^V(Gini(D^v)  \frac{|D^v|}{|D|}$

import numpy as np
from collections import Counter
import metrics

class DTClassifier():
    
    def __init__(self, max_gini=0.2, min_samples=9, weight=1):  
        self.Tree = None
        self.max_gini = max_gini
        self.min_samples = min_samples
        self.weight = weight #adaboost中的权重
        
    def _split(self, X, Y, d, value):
        '''依据d和value将某节点上的数据二分'''
        ind1 = (X[:,d] < value)
        ind2 = (X[:,d] >= value)
        return X[ind1], X[ind2], Y[ind1], Y[ind2] #X_l, X_r, y_l, y_r
    
    def _gini(self, Y):
        '''计算某节点上的数据的gini'''
        gini = 0.
        for i in range(len(Y)):
            r = Y[i][0] / len(Y)
            gini += (r**2)*(Y[i][1] * len(Y))
        gini = 1-gini
        return gini
    
    def _Searchdv(self, X, Y):
        '''计算某节点上最优的划分特征和二分点'''
        d, value, gini_best = 0, 0., 1e4
        for i in range(X.shape[1]): #遍历搜索全局最佳的Gain以得到最佳d，v
            ind = np.argsort(X[:,i]) #通过得到X排序的序列号以在下面的搜索中从小到大的搜索x
            for j in range(1,X.shape[0]):
                if X[ind[j],i] != X[ind[j-1],i]:
                    v= 0.5 * (X[ind[j],i] + X[ind[j-1],i])
                    X_l, X_r, Y_l, Y_r = self._split(X, Y, d=i, value=v)
                    gini = self._gini(Y_l) + self._gini(Y_r)
                    if gini < gini_best:
                        gini_best, d, value = gini, i, v
        return d, value
    
    def _CreateTree(self, X, Y):
        '''用字典中的key储存判断信息，用value储存子树或者最终结果'''
        d, value = self._Searchdv(X, Y)
        X_l, X_r, Y_l, Y_r = self._split(X, Y, d, value)
        if ((self._gini(Y_l) <= self.max_gini) or (len(Y_l)<=self.min_samples)):
            if ((self._gini(Y_r) <= self.max_gini) or (len(Y_r)<=self.min_samples)):
                Tree = {(d,value,0):int(np.median(Y_l[:,0])), (d,value,1):int(np.median(Y_l[:,0]))}
            else:
                Tree = {(d,value,0):int(np.median(Y_l[:,0])), (d,value,1):self._CreateTree(X_r, Y_r)}
        else:
            if ((self._gini(Y_r) <= self.max_gini) or (len(Y_r)<=self.min_samples)):
                Tree = {(d,value,0):self._CreateTree(X_l, Y_l), (d,value,1):int(np.median(Y_l[:,0]))}
            else:
                Tree = {(d,value,0):self._CreateTree(X_l, Y_l), (d,value,1):self._CreateTree(X_r, Y_r)}
        return Tree
        
    def fit(self, X_train, y_train):
        self.YandW = np.hstack([y_train.reshape(-1,1), self.weight.reshape(-1,1)])
        self.Tree = self._CreateTree(X_train, self.YandW)
        return self
             
    def score(self, X_test, y_test, scoring=metrics.acc_score):
        y_pred = self.predict(X_test)
        score = scoring(y_pred, y_test)
        return score
    
    def _predict_vec(self, x, Tree):
        for key in Tree.keys():
            if (x[key[0]] > key[1]) == key[2]:
                if type(Tree[key]) == dict:
                    y = self._predict_vec(x, Tree[key])
                else: y = Tree[key]
        return y
    
    def predict(self, X_test, Tree=None, negative='False'):
        y_pred = []
        if Tree == None:
            Tree = self.Tree
        for x in X_test:
            y_pred.append(self._predict_vec(x, Tree))
        y_pred =np.array(y_pred)
        if negative == 'Ture':
            for i in y_pred:
                if i==0:
                    i=-1
            return y_pred
        return y_pred