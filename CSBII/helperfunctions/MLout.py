# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:42:45 2023

@author: nkp68
"""
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def printacc(ytest, ypred):
    score=accuracy_score(ytest, ypred)
    #print(score)
    
    return score

def printf1(ytest, ypred):
    score=f1_score(ytest, ypred)
    #print(score)
    
    return score


def printrecall(ytest, ypred):
    score=recall_score(ytest, ypred)
    #print(score)
    
    return score

def printprecision(ytest, ypred):
    score=precision_score(ytest, ypred)
    #print(score)
    
    return score

from sklearn.metrics import roc_auc_score

def printauc(ytest, ypred):
   score = roc_auc_score(ytest, ypred, average=None)
    
   return score
