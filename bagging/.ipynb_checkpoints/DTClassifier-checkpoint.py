#!/usr/bin/env python
# coding: utf-8

# $Gini = 1 - \sum_i^m(p_i^2)$

# $Gain(D,a) = Gini(D) - \sum_{v=1}^V(Gini(D^v)  \frac{|D^v|}{|D|}$

import numpy as np
from collections import Counter
import metrics

class DTClassifier():
    
    def __init__(self, max_depth=2, criterion='gini'): #暂时先不剪枝 
        self.Tree = None

    def _split(self, X, y, d, value):
        '''依据d和value将某节点上的数据二分'''
        ind1 = (X[:,d] < value)
        ind2 = (X[:,d] >= value)
        return X[ind1], X[ind2], y[ind1], y[ind2] #X_l, X_r, y_l, y_r
    
    def _gini(self, y):
        '''计算某节点上的数据的gini'''
        counter = Counter(y)
        gini = 0.
        for i in counter.values():
            r = i / len(y) #r:值对应的概率
            gini += r ** 2
        gini = 1 - gini
        return gini
    
    def _Searchdv(self, X, y):
        '''计算某节点上最优的划分特征和二分点'''
        d, value, gini_best = 0, 0., float('inf')
        for i in range(X.shape[1]): #遍历搜索全局最佳的Gain以得到最佳d，v
            ind = np.argsort(X[:,i]) #通过得到X排序的序列号以在下面的搜索中从小到大的搜索x
            for j in range(1,X.shape[0]):
                if X[ind[j],i] != X[ind[j-1],i]:
                    v= 0.5 * (X[ind[j],i] + X[ind[j-1],i])
                    X_l, X_r, y_l, y_r = self._split(X, y, d=i, value=v)
                    gini = self._gini(y_l) + self._gini(y_r)
                    if gini < gini_best:
                        gini_best, d, value = gini, i, v
        return d, value
    
    def _CreateTree(self, X, y):
        '''用字典中的key储存判断信息，用value储存子树或者最终结果'''
        d, value = self._Searchdv(X, y)
        X_l, X_r, y_l, y_r = self._split(X, y, d, value)
        if ((self._gini(y_l) <= 0.3) or (len(y_l)<=3)):
            if ((self._gini(y_r) <= 0.3) or (len(y_r)<=3)):
                Tree = {(d,value,0):np.median(y_l), (d,value,1):np.median(y_r[0])}
            else:
                Tree = {(d,value,0):np.median(y_l[0]), (d,value,1):self._CreateTree(X_r, y_r)}
        else:
            if ((self._gini(y_r) <= 0.3) or (len(y_r)<=3)):
                Tree = {(d,value,0):self._CreateTree(X_l, y_l), (d,value,1):np.median(y_r[0])}
            else:
                Tree = {(d,value,0):self._CreateTree(X_l, y_l), (d,value,1):self._CreateTree(X_r, y_r)}
        return Tree
        
    def fit(self, X_train, y_train):
        self.Tree = self._CreateTree(X_train, y_train)
        return self
             
    def score(self, X_test, y_test, scoring=metrics.acc_score):
        y_pred = self.predict(X_test)
        score = scoring(y_pred, y_test)
        return score
    
    def get_params(self):
        return self.Tree
    
    def set_params(self, Tree=None):
        if Tree = None:
            Tree = self.Tree
        self.Tree = Tree
        return self
    
    def _predict_vec(self, x, Tree):
        for key in Tree.keys():
            if (x[key[0]] > key[1]) == key[2]:
                if type(Tree[key]) == dict:
                    y = self._predict_vec(x, Tree[key])
                else: y = Tree[key]
        return y
    
    def predict(self, X_test, Tree=None):
        y_pred = []
        if Tree == None:
            Tree = self.Tree
        for x in X_test:
            y_pred.append(self._predict_vec(x, Tree))
        return np.array(y_pred)