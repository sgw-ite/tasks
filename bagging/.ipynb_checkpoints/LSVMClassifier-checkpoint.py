#!/usr/bin/env python
# coding: utf-8


import numpy as np
import metrics

class LSVMClassifier():
    def __init__(self):
        self._theta = None
        self._omiga = None
        self._b = 0

    def _J(self, theta, X_b, y_train, lamda):
        '''u = y_train * X_b.dot(self._theta)
        sum = 0
        for i in range(X_b.shape[0]):
            if u[i] < 1:
                sum += (1-u[i])
        cost = 0.5*lamda*np.sum(theta[1:]**2) + sum
        return cost'''
        sum = 0
        for i in range(X_b.shape[0]):
            u = y_train[i] * np.sum(X_b[i].dot(self._theta))
            if u<1:
                sum += 1-u
        cost = (1/2) * lamda * np.sum(theta[1:]**2) + sum
        return cost
    
    def _dJ(self, theta, X_b, y_train, lamda):
        '''u = y_train * X_b.dot(self._theta)
        sum = 0
        for i in range(X_b.shape[0]):
            if u[i] < 1:
                sum = sum - y_train.reshape(622,-1)[i] * X_b[i]
        gradient = sum + lamda*theta
        return gradient'''
        sum = 0
        for i in range(X_b.shape[0]):
            u = y_train[i] * np.sum(X_b[i].dot(self._theta))
            if u<1:
                sum -= y_train[i]*X_b[i]
        gradient = lamda * theta + sum
        return gradient

    def fit(self, X_train, y_train, max_iter=1e4, epsilon=1e-4, eta=0.1, lamda=1):
        ss = y_train.copy()#？？？？？？？？？？？？
        for i in range(len(ss)):
            if ss[i] == 0:
                ss[i] = -1
        X_b = np.hstack([X_train, np.ones((len(X_train),1))])
        theta = np.ones(X_b.shape[1])   
        self._theta = theta
        self._omiga = self._theta[:-1]
        cur_iter = 0
            
        for iter in range(int(max_iter)) :              
            gradient = self._dJ(theta, X_b, ss, lamda)
            #print(gradient,'/n')
            lasttheta = theta
            theta = theta - eta * gradient
            #print(theta,'/n',self._J(theta, X_b, ss, lamda))
            if (abs(self._J(theta, X_b, ss, lamda) - self._J(lasttheta, X_b, ss, lamda)) < epsilon):
                break
            cur_iter += 1 
            if ((cur_iter%20)==0):
                #print('delta:', abs(self._J(theta, X_b, ss, lamda) - self._J(lasttheta, X_b, ss, lamda)))
        
        #print(cur_iter,' ',abs(self._J(theta, X_b, self._y_train, lamda) - self._J(lasttheta, X_b, self._y_train, lamda)))
        self._theta = theta
        self._b = self._theta[-1]
        self._omiga = self._theta[:-1]
        #print('theta=',self._theta)
        return self
    
    def predict(self, X_test):
        X_b = np.hstack([X_test, np.ones((len(X_test),1))])
        y_pred = np.array((X_b.dot(self._theta)>0), dtype='int')
        return y_pred
    
    def score(self, X_test, y_test, scoring=metrics.acc_score):
        y_pred = self.predict(X_test)
        score = scoring(y_pred, y_test)
        return score
    
    def get_params(self)
        return self._omiga, self._b, self._theta
    
    def set_params(self, omiga=None, b=None, theta=None)
        if omiga = None:
            omiga = self._omiga
        if b = None:
            b = self._b
        if theta = None:
            theta = self.theta
        self._omiga = omiga
        self._b = b
        self._theta = theta
        return self