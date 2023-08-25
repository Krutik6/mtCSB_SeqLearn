# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:36:07 2023

@author: nkp68
"""

import pandas as pd

def inData(dat, lim):
    df = pd.read_csv(dat)

    #save samples
    sampleNames = df.iloc[:, 0]
    df = df.iloc[:, 1:lim]
    
    return df, sampleNames
    
    


def reClass(df, col):
    df[col] = df[col].astype('category')
    
    encode_map = {
     'Yes': 1,
     'No': 0
}
    #replace Yes with 1 and No with 0
    df[col].replace(encode_map, inplace=True)
    
    return df


from sklearn.model_selection import train_test_split

def makeXy(df, col, splitsize=0.2, rs=0):
    X=df.drop(col, axis=1).values #independent as numpy
    y=df[col].values # dependent as numpy
    y = y.astype(float)
    
    
    X_train,X_test,y_train,y_test=train_test_split(X,
                                                   y,
                                                   test_size=splitsize,
                                                   random_state=rs)
    
    return X_train,X_test,y_train,y_test
    

def makeXyVal(X, y, splitsize=0.2, rs=0):
    X_train,X_test,y_train,y_test=train_test_split(X,
                                                   y,
                                                   test_size=splitsize,
                                                   random_state=rs)
    
    return X_train,X_test,y_train,y_test



def makeXyInd(df, col, splitsize=0.2, rs=0):
    X=df.drop('target', axis=1).values #independent as numpy
    y=df['target'].values # dependent as numpy
    y = y.astype(float)
    
    indices = range(len(df))
    
    X_train,X_test,y_train,y_test,indices_train,indices_test=train_test_split(X,
                                                   y,
                                                   indices,
                                                   test_size=splitsize,
                                                   random_state=rs)

    return X_train,X_test,y_train,y_test,indices_train,indices_test


def makeXyValInd(df, indices_train, col, splitsize=0.2, rs=0):
    df = df.iloc[indices_train]
    X=df.drop('target', axis=1).values #independent as numpy
    y=df['target'].values # dependent as numpy
    y = y.astype(float)
    
    indices = range(len(df))
    
    X_train,X_test,y_train,y_test,indices_train,indices_test=train_test_split(X,
                                                   y,
                                                   indices,
                                                   test_size=splitsize,
                                                   random_state=rs)

    return X_train,X_test,y_train,y_test,indices_train,indices_test


import torch

def makeTorch(X_train,X_test,y_train,y_test):
    X_train=torch.FloatTensor(X_train)
    X_test=torch.FloatTensor(X_test)
    y_train=torch.FloatTensor(y_train)
    y_test=torch.FloatTensor(y_test)
    
    return X_train,X_test,y_train,y_test


def makeTorch2(X, y):
    X_torch=torch.FloatTensor(X)
    y_torch=torch.FloatTensor(y)
    
    return X_torch,y_torch
    

from sklearn.preprocessing import StandardScaler   

def doScaler(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test



def doScalerVal(X_train, X_test, X_val):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    
    return X_train,X_test,X_val



import os

def savemodel(model, name, outDir):
    tsave = os.path.join(outDir, name)
    torch.save(model, tsave)
    

from imblearn.over_sampling import RandomOverSampler    
    
def applysmote(X_train, y_train):
  ROS = RandomOverSampler(random_state=42)
  x_train_oversampled, y_train_oversampled = ROS.fit_resample(X_train, y_train)
  return x_train_oversampled, y_train_oversampled