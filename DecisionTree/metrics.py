#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


def confusion_metrix(y_pred, y_test):
    TN = np.sum((y_pred == 1) & (y_test == 1))
    FN = np.sum((y_pred == 1) & (y_test == 0))
    TP = np.sum((y_pred == 0) & (y_test == 0))
    FP = np.sum((y_pred == 0) & (y_test == 1))
    return TN, FN, TP, FP

def acc_score(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_pred)

def precisionrate(y_pred, y_test):
    TN, FN, TP, FP = confusion_metrix(y_pred, y_test)
    precisionrate = TP / (TP + FP)
    return precisionrate

def recall(y_pred, y_test):
    TN, FN, TP, FP = confusion_metrix(y_pred, y_test)
    recall = TP / (TP + FN)
    return recall
                 
def F1_score(y_pred, y_test):
    TN, FN, TP, FP = confusion_metrix(y_pred, y_test)
    F1_score = (2 * TP) / (len(y_pred) + TP - TN)
    return F1_score

