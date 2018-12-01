#!/usr/bin/env python
# coding: utf-8


import numpy as np
import metrics

class LClassifier():
    
    def __init__(self):
        self.theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))
    
    def _J(self, X_b, y, theta):
            p = self._sigmoid(X_b.dot(theta))
            #J = - np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y)
            try:
                return - np.sum(y*np.log(p) + (1-y)*np.log(1-p)) / len(y) #直接把一长串放在return里面还是这样子勒
            except:
                return float('inf')
    
    def _dJ(self, theta, X_b, y):
            p = self._sigmoid(X_b.dot(theta))
            return X_b.T.dot(p - y) / len(X_b)
    
    def fit(self, X_train, y_train, max_iter=1e3, epsilon=1e-4, eta=0.01): 
        X_b = np.hstack([X_train, np.ones((len(X_train),1))])
        initial_theta = np.zeros(X_b.shape[1])
        theta = initial_theta        
        cur_iter = 0
            
        while cur_iter < max_iter:                
            gradient = self._dJ(theta, X_b, y_train)
            lasttheta = theta
            theta = theta - eta * gradient
            #self.h.append(1)
            #cur_iter += 1
            if abs(self._J(X_b, y_train, theta) - self._J(X_b, y_train, lasttheta)) < epsilon:
                break
            cur_iter += 1        

        self.theta = theta
        return self
        
    def _predictp(self, X_pred):
        X_b = np.hstack([X_pred, np.ones((len(X_pred),1))])
        p = self._sigmoid(X_b.dot(self.theta))
        return p
    
    def predict(self, X_pred, rate=0.5):
        p = self._predictp(X_pred)
        return np.array(p >= rate, dtype='int')
    
    def score(self, X_test, y_test, scoring=metrics.acc_score):
        y_pred = self.predict(X_test)
        score = scoring(y_pred, y_test)
        return score
    
    def __repr__(self):
        return "LClassifier()"