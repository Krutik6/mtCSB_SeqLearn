# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:02:11 2023

@author: nkp68
"""
#load helper module
import os
from helperfunctions import *
test.func()

#load essential libraries
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import random


#import nerual network libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torchsummary import summary
import torchvision.transforms as transforms
import torchvision.models as models

#import auxilary ML libraries
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import preprocessing
import itertools
from itertools import product
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import _safe_indexing, indexable
from itertools import chain
import json

#set global variables
device = torch.device('cpu')
WORKING_DIRECTORY = os.path.dirname(__file__)
PARENT_DIR = os.path.join(WORKING_DIRECTORY, '../')
DATA_DIR = os.path.join(PARENT_DIR, 'data/')
CSBI_DIR = os.path.join(PARENT_DIR, 'CSBI/')
CHURN_DIR = os.path.join(CSBI_DIR, 'churn/')

###############################################################################
#STEP 1 HP training
########LOAD DATA FROM EXPLORATION AND INITIAL SANITY CHECKS
# df, names = structure.inData(dat = os.path.join(DATA_DIR,'data_csbi.csv'),
#                               lim =12)

# #######CLEAN UP DATA
# # Encoding 'status' as label 1 & 0 , naming the field as target
# df['target'] = df['Mix']
# structure.reClass(df=df, col='target')
# df.drop('Mix',axis = 1, inplace=True)

# random.seed(None)

# #test_train split
# X_train,X_test,y_train,y_test = structure.makeXy(df=df, 
#                                                   col='target',
#                                                   splitsize=0.15,
#                                                   rs=42)
# #create validation set
# X_train,X_val,y_train,y_val = structure.makeXyVal(X_train, y_train,
#                                                 splitsize=0.15,
#                                                 rs=42)

# count_0 = list(y_train).count(0.0)
# count_1 = list(y_train).count(1.0)

# print("Number of 0.0s:", count_0)
# print("Number of 1.0s:", count_1)
# #scale training and testing X vals
# X_train,X_test,X_val = structure.doScalerVal(X_train=X_train,X_test=X_test,X_val=X_val)

# #oversample 
# X_train,y_train = structure.applysmote(X_train,y_train)

# #transform to tensors
# X_train,y_train= structure.makeTorch2(X_train,y_train)
# X_val,y_val= structure.makeTorch2(X_val,y_val)
# X_test,y_test= structure.makeTorch2(X_test,y_test)

# #define hyperparameters
# model = MLP.ChurnModel_CSBI(n_input_dim=X_train.shape[1])
# print(X_train.shape[1])
# loss_func = nn.BCELoss()

# #make lists
# bs = [15, 31, 62]
# e = [20, 40, 80, 160]
# lr = [0.1, 0.01, 0.001]

# hp = [dict(zip(('bs','lr','e'), (b,l,e))) for b,l,e in product(bs, lr, e)]

# savename = ['bs15lr01e20.txt','bs15lr01e40.txt','bs15lr01e80.txt','bs15lr01e160.txt',
#             'bs15lr001e20.txt','bs15lr001e40.txt','bs15lr001e80.txt','bs15lr001e160.txt',
#             'bs15lr0001e20.txt','bs15lr0001e40.txt','bs15lr0001e80.txt','bs15lr0001e160.txt',
#             'bs31lr01e20.txt','bs31lr01e40.txt','bs31lr01e80.txt','bs31lr01e160.txt',
#             'bs31lr001e20.txt','bs31lr001e40.txt','bs31lr001e80.txt','bs31lr001e160.txt',
#             'bs31lr0001e20.txt','bs31lr0001e40.txt','bs31lr0001e80.txt','bs31lr0001e160.txt',
#             'bs62lr01e20.txt','bs62lr01e40.txt','bs62lr01e80.txt','bs62lr01e160.txt',
#             'bs62lr001e20.txt','bs62lr001e40.txt','bs62lr001e80.txt','bs62lr001e160.txt',
#             'bs62lr0001e20.txt','bs62lr0001e40.txt','bs62lr0001e80.txt','bs62lr0001e160.txt']

# for k in range(len(hp)):
#     print(savename[k])
#     print(hp[k]['lr'])
#     print(hp[k]['e'])
#     print(hp[k]['bs'])
    
#     Train_Loss_list = []
#     Val_Loss_List = []
#     Val_Accucary_List = []
#     Val_F1_List = []
#     Val_Precision_List = []
#     Val_Recall_List = []
#     Val_ypred_List = []
#     Val_y_List = []
#     Test_Accucary_List = []
#     Test_F1_List = []
#     Test_Precision_List = []
#     Test_Recall_List = []
#     Test_ypred_List = []
#     Test_y_List = []
    
#     for i in range(0,25):
        
#         #create dataoaders
#         y_tensor = y_train.unsqueeze(1)
#         train_ds = TensorDataset(X_train, y_tensor)
#         train_dl = DataLoader(train_ds, batch_size=hp[k]['bs'], shuffle=True)

#         yval_tensor = y_val.unsqueeze(1)
#         val_ds = TensorDataset(X_val, yval_tensor)
#         val_dl = DataLoader(val_ds, batch_size=hp[k]['bs'])

#         ytest_tensor = y_test.unsqueeze(1)
#         test_ds = TensorDataset(X_test, ytest_tensor)
#         test_dl = DataLoader(test_ds, batch_size=hp[k]['bs'])
        
#         #define optimizer
#         optimizer = torch.optim.Adam(model.parameters(), lr=hp[k]['lr'])
#         train_step_fn = tensorfn.make_train_step_fun(model, loss_func, optimizer)
#         val_step_fn = tensorfn.make_val_step_fun(model, loss_func)

#         train_losses=[]
#         val_losses=[]
#         #train model
#         for epoch in range(hp[k]['e']):
#             loss = tensorfn.mini_batch(device, train_dl, train_step_fn)
#             train_losses.append(loss)

#             with torch.no_grad():
#                 val_loss = tensorfn.mini_batch(device, test_dl, val_step_fn)
#                 val_losses.append(val_loss)
                
#         train_loss_score = train_losses[hp[k]['e']-1]
#         val_loss_score = val_losses[hp[k]['e']-1]
        
#         #calculate validation accuracy
#         val_y, val_pred, val_loss = MLP.testModel(model, val_dl, y_val, loss_func)

#         #define model validity
#         acc_val = MLout.printacc(ytest=val_y, ypred=val_pred)
#         f1_val = MLout.printf1(ytest=val_y, ypred=val_pred)
#         precision_val = MLout.printprecision(ytest=val_y, ypred=val_pred)
#         recall_val = MLout.printrecall(ytest=val_y, ypred=val_pred)
        
#         #calculate testing accuracy
#         test_y, test_pred, test_loss = MLP.testModel(model, test_dl, y_test, loss_func)

#         #define model validity
#         test_acc = MLout.printacc(ytest=test_y, ypred=test_pred)
#         test_f1 = MLout.printf1(ytest=test_y, ypred=test_pred)
#         test_precision = MLout.printprecision(ytest=test_y, ypred=test_pred)
#         test_recall = MLout.printrecall(ytest=test_y, ypred=test_pred)


#         Train_Loss_list.append(train_loss_score)
#         Val_Loss_List.append(val_loss_score)
#         Val_Accucary_List.append(acc_val)
#         Val_F1_List.append(f1_val)
#         Val_Precision_List.append(precision_val)
#         Val_Recall_List.append(recall_val)
#         Test_Accucary_List.append(test_acc)
#         Test_F1_List.append(test_f1)
#         Test_Precision_List.append(test_precision)
#         Test_Recall_List.append(test_recall)

#     all_stats = list(zip(Train_Loss_list,Val_Loss_List,Val_Accucary_List,
#                           Val_F1_List,Val_Precision_List,Val_Recall_List,
#                           Test_Accucary_List,Test_F1_List,Test_Precision_List,
#                           Test_Recall_List))
#     all_stats_df = pd.DataFrame(list(all_stats))
    
#     all_stats_df.columns = ['Train_loss', 'Val_loss', 'Val_accuracy', 'Val_f1', 
#                             'Val_precision', 'Val_recall', 'Test_accuracy',
#                             'Test_f1', 'Test_precision', 'Test_recall']
    
#     save=os.path.join(CHURN_DIR, savename[k])
#     all_stats_df.to_csv(save, sep=' ')
###############################################################################
#STEP 2 test best model
#######LOAD DATA FROM EXPLORATION AND INITIAL SANITY CHECKS
# df, names = structure.inData(dat = os.path.join(DATA_DIR,'data_csbi.csv'),
#                               lim =12)

# #######CLEAN UP DATA
# # Encoding 'status' as label 1 & 0 , naming the field as target
# df['target'] = df['Mix']
# structure.reClass(df=df, col='target')
# df.drop('Mix',axis = 1, inplace=True)

# random.seed(None)

# #test_train split
# X_train,X_test,y_train,y_test = structure.makeXy(df=df, 
#                                                   col='target',
#                                                   splitsize=0.15,
#                                                   rs=42)
# #create validation set
# X_train,X_val,y_train,y_val = structure.makeXyVal(X_train, y_train,
#                                                 splitsize=0.18,
#                                                 rs=42)

# #scale training and testing X vals
# X_train,X_test,X_val = structure.doScalerVal(X_train=X_train,X_test=X_test,X_val=X_val)

# #transform to tensors
# X_train,y_train= structure.makeTorch2(X_train,y_train)
# X_val,y_val= structure.makeTorch2(X_val,y_val)
# X_test,y_test= structure.makeTorch2(X_test,y_test)

# #define batch size
# bs=15

# #combine both train and test data into dataloaders objects
# y_tensor = y_train.unsqueeze(1)
# train_ds = TensorDataset(X_train, y_tensor)
# train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

# yval_tensor = y_val.unsqueeze(1)
# val_ds = TensorDataset(X_val, yval_tensor)
# val_dl = DataLoader(val_ds, batch_size=bs)

# ytest_tensor = y_test.unsqueeze(1)
# test_ds = TensorDataset(X_test, ytest_tensor)
# test_dl = DataLoader(test_ds, batch_size=bs)

# #define hyperparameters
# print(X_train.shape[1])
# model = MLP.ChurnModel_CSBI(n_input_dim=X_train.shape[1])
# loss_func = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# epochs = 160

# train_step_fn = tensorfn.make_train_step_fun(model, loss_func, optimizer)
# val_step_fn = tensorfn.make_val_step_fun(model, loss_func)

# #######TRAIN MODEL
# train_losses=[]
# val_losses=[]

# for epoch in range(epochs):
#     loss = tensorfn.mini_batch(device, train_dl, train_step_fn)
#     train_losses.append(loss)

#     with torch.no_grad():
#         val_loss = tensorfn.mini_batch(device, test_dl, val_step_fn)
#         val_losses.append(val_loss)

# #plot model performance on training data
# # plots.plotlosses(epochs=epochs, train_losses=train_losses,  val_losses=val_losses, 
# #                     outDir=CSBI_DIR, plotname='lossplot_bs15e160lr01_3.png')

# print(train_losses[epochs-1])
# print(val_losses[epochs-1])

# #####GET RESULTS
# #calculate validation accuracy
# val_y, val_pred, val_loss = MLP.testModel(model, val_dl, y_val, loss_func)

# #create confusion matrix
# # plots.CM(y_test=val_y, y_pred=val_pred, outDir=CSBI_DIR, name='CM.val.png',
# #             CMmax=8, CMmin=0)

# #define model validity
# acc_val = MLout.printacc(ytest=val_y, ypred=val_pred)
# f1_val = MLout.printf1(ytest=val_y, ypred=val_pred)
# precision_val = MLout.printprecision(ytest=val_y, ypred=val_pred)
# recall_val = MLout.printrecall(ytest=val_y, ypred=val_pred)

# print(acc_val, f1_val, precision_val, recall_val)

# #calculate validation accuracy
# test_y, test_pred, test_loss = MLP.testModel(model, test_dl, y_test, loss_func)

# # create confusion matrix
# # plots.CM(y_test=test_y, y_pred=test_pred, outDir=CSBI_DIR, name='CM.png__bs15e160lr001_3.png'
# #           ,CMmax=8, CMmin=0)

# #define model validity
# test_acc = MLout.printacc(ytest=test_y, ypred=test_pred)
# test_f1 = MLout.printf1(ytest=test_y, ypred=test_pred)
# test_precision = MLout.printprecision(ytest=test_y, ypred=test_pred)
# test_recall = MLout.printrecall(ytest=test_y, ypred=test_pred)
# test_auc = MLout.printauc(ytest=test_y, ypred=test_pred)

# print(test_acc, test_f1, test_precision, test_recall, test_auc)

#save model
# structure.savemodel(model=model, name='MLP.churn.bs16e160lr001.pt', outDir=INFO_DIR)
#GOOD MODEL!!!
#############################################################################################################
#run the model 1000 iterations - gain interpretable results
df, names = structure.inData(dat = os.path.join(DATA_DIR,'data_csbi.csv'),
                              lim =12)

# #######CLEAN UP DATA
#Encoding 'status' as label 1 & 0 , naming the field as target
df['target'] = df['Mix']
structure.reClass(df=df, col='target')
df.drop('Mix',axis = 1, inplace=True)

#check dtypes
df['target'] =df['target'].astype(float)
print(df.dtypes)

#remove any missing values
tmp = df.isnull().sum().reset_index(name='missing_val')
tmp[tmp['missing_val']!= 0]

#test_train split
#####create indexes
X_train,X_test,y_train,y_test,indices_train,indices_test = structure.makeXyInd(df=df, 
                                                  col='target',
                                                  splitsize=0.15,
                                                  rs=42)
#create validation set
X_train,X_val,y_train,y_val,indices_train,indices_val = structure.makeXyValInd(df=df,
                                            indices_train= indices_train,
                                            col ='target',
                                            splitsize=0.18,
                                            rs=42)

X=df.drop('target', axis=1).values #independent as numpy
y=df['target'].values # dependent as numpy
y = y.astype(float)

indices = range(len(df))

#scale training and testing X vals
X_train,X_test,X_val = structure.doScalerVal(X_train=X_train,X_test=X_test,X_val=X_val)
#oversample
X_train,y_train = structure.applysmote(X_train,y_train)

#transform to tensors
X_train,y_train= structure.makeTorch2(X_train,y_train)
X_val,y_val= structure.makeTorch2(X_val,y_val)
X_test,y_test= structure.makeTorch2(X_test,y_test)

#define batch size
bs=62

#combine both train and test data into dataloaders objects
y_tensor = y_train.unsqueeze(1)
train_ds = TensorDataset(X_train, y_tensor)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

yval_tensor = y_val.unsqueeze(1)
val_ds = TensorDataset(X_val, yval_tensor)
val_dl = DataLoader(val_ds, batch_size=bs)

ytest_tensor = y_test.unsqueeze(1)
test_ds = TensorDataset(X_test, ytest_tensor)
test_dl = DataLoader(test_ds, batch_size=bs)


#define hyperparameters
model = MLP.ChurnModel_CSBI(n_input_dim=X_train.shape[1])
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
epochs = 40

train_step_fn = tensorfn.make_train_step_fun(model, loss_func, optimizer)
val_step_fn = tensorfn.make_val_step_fun(model, loss_func)

#######TRAIN MODEL
preds=[]
acc=[]
f1=[]
precision=[]
recall=[]
auc=[]
for i in range(0,100):
    train_losses=[]
    val_losses=[]
    
    for epoch in range(epochs):
        loss = tensorfn.mini_batch(device, train_dl, train_step_fn)
        train_losses.append(loss)
    
        with torch.no_grad():
            val_loss = tensorfn.mini_batch(device, test_dl, val_step_fn)
            val_losses.append(val_loss)
    
    #calculate validation accuracy
    test_y, test_pred, test_loss = MLP.testModel(model, test_dl, y_test, loss_func)
    
    #get train stats
    test_acc = MLout.printacc(ytest=test_y, ypred=test_pred)
    test_f1 = MLout.printf1(ytest=test_y, ypred=test_pred)
    test_precision = MLout.printprecision(ytest=test_y, ypred=test_pred)
    test_recall = MLout.printrecall(ytest=test_y, ypred=test_pred)
    test_auc = MLout.printauc(ytest=test_y, ypred=test_pred)
    
    preds.append(test_pred)
    acc.append(test_acc)
    f1.append(test_f1)
    precision.append(test_precision)
    recall.append(test_recall)
    auc.append(test_auc)

pred_df = pd.DataFrame(preds)
pred_df =  pred_df.transpose()
    
test_samples = names[indices_test].reset_index()
test_bin = pd.Series(test_y, name='Mixture')
pred_df = pd.concat([pred_df, test_bin, test_samples], axis=1)

save=os.path.join(CSBI_DIR, 'csbi_predictions.csv')
pred_df.to_csv(save, sep=',')

all_stats = list(zip(acc, f1, precision, recall, auc))
all_stats_df = pd.DataFrame(list(all_stats))
all_stats_df.columns = [ 'Test_accuracy', 'Test_f1',
                        'Test_precision', 'Test_recall', 'Test_auc']

save=os.path.join(CSBI_DIR, 'csbi_test_stats.csv')
all_stats_df.to_csv(save, sep=',')
###############################################################################
