# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 19:31:35 2023

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
import shap

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
TMP_DIR = os.path.join(PARENT_DIR, 'tmp/')
INFO_DIR = os.path.join(PARENT_DIR, 'info/')
PLOT_DIR = os.path.join(PARENT_DIR, 'plots/')
churn_DIR = os.path.join(INFO_DIR, 'churn/')
wide_DIR = os.path.join(INFO_DIR, 'wide/')
deep_DIR = os.path.join(INFO_DIR, 'deep/')
churn_filt_DIR = os.path.join(INFO_DIR, 'churn_filt/')
###############################################################################
#############STEP 1 USE MLP ANN WITH NO SEL####################################
#MLP ANN with advanced NN building + scaling
#This step is just to check if there are patterns which can be detected prior to more advance NN - this is to check we are asking an appropriate question based on our data
########LOAD DATA FROM EXPLORATION AND INITIAL SANITY CHECKS
# df, names = structure.inData(dat = os.path.join(DATA_DIR,'data_csbii.csv'),
#                               lim =22)

# #######CLEAN UP DATA
# #Encoding 'status' as label 1 & 0 , naming the field as target
# df['target'] = df['Mix']
# structure.reClass(df=df, col='target')
# df.drop('Mix',axis = 1, inplace=True)

# #check dtypes
# df['target'] =df['target'].astype(float)
# #print(df.dtypes)

# #remove any missing values
# tmp = df.isnull().sum().reset_index(name='missing_val')
# tmp[tmp['missing_val']!= 0]

# #test_train split
# X_train,X_test,y_train,y_test = structure.makeXy(df=df, 
#                                                   col='target',
#                                                   splitsize=0.2,
#                                                   rs=0)

# #scale training and testing X vals
# X_train,X_test = structure.doScaler(X_train,X_test)

# #oversample 
# X_train,y_train = structure.applysmote(X_train,y_train)

# #transform to tensors
# X_train,X_test,y_train,y_test = structure.makeTorch(X_train,X_test,
#                                                     y_train,y_test)

# #define batch size
# bs=5

# #combine both train and test data into dataloaders objects
# y_tensor = y_train.unsqueeze(1)
# train_ds = TensorDataset(X_train, y_tensor)
# train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

# ytest_tensor = y_test.unsqueeze(1)
# test_ds = TensorDataset(X_test, ytest_tensor)
# test_loader = DataLoader(test_ds, batch_size=bs)

# #create the MLP model
# model = MLP.ChurnModel(n_input_dim=X_train.shape[1])

# #define hyperparameters
# loss_func = nn.BCELoss()
# learning_rate = 0.1
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# epochs = 20

# #train model
# model, train_loss =MLP.trainModel(model=model, epochs=epochs, train_DL=train_dl, 
#                 loss_func=loss_func, optimizer=optimizer)

# #plot model performance on training data
# plots.plotTrain(epochs=epochs, listofloss=train_loss, 
#                 outDir=PLOT_DIR, name='MLP.trainPerformance.shuffle.png')

# #compare prediction of test set with real test set
# ytest, ytest_pred, test_loss = MLP.testModel(model=model, 
#                                               test_DL=test_loader,
#                                               y_test=y_test,
#                                               loss_func=loss_func)

# # print(test_loss.item())

# #create confusion matrix
# plots.CM(y_test=ytest, y_pred=ytest_pred, outDir=PLOT_DIR, name='MLP.CM.shuffle.png',
#             CMmax=10, CMmin=0)

# #accuracy score
# acc = MLout.printacc(ytest=ytest, ypred=ytest_pred)
# f1 = MLout.printf1(ytest=ytest, ypred=ytest_pred)
# precision = MLout.printprecision(ytest=ytest, ypred=ytest_pred)
# recall = MLout.printrecall(ytest=ytest, ypred=ytest_pred)

# save model
# structure.savemodel(model=model, name='MLP.model.shuffle.pt', outDir=INFO_DIR)
###############################################################################
#STEP 2 make major improvements to the model

########LOAD DATA FROM EXPLORATION AND INITIAL SANITY CHECKS
# df, names = structure.inData(dat = os.path.join(DATA_DIR,'data_csbii.csv'),
#                               lim =22)

# #######CLEAN UP DATA
# #Encoding 'status' as label 1 & 0 , naming the field as target
# df['target'] = df['Mix']
# structure.reClass(df=df, col='target')
# df.drop('Mix',axis = 1, inplace=True)

# #check dtypes
# df['target'] =df['target'].astype(float)
# #print(df.dtypes)

# #remove any missing values
# tmp = df.isnull().sum().reset_index(name='missing_val')
# tmp[tmp['missing_val']!= 0]

# random.seed(None)

# #test_train split
# X_train,X_test,y_train,y_test = structure.makeXy(df=df, 
#                                                   col='target',
#                                                   splitsize=0.15,
#                                                   rs=0)
# #create validation set
# X_train,X_val,y_train,y_val = structure.makeXyVal(X_train, y_train,
#                                                 splitsize=0.18,
#                                                 rs=0)

# #scale training and testing X vals
# X_train,X_test,X_val = structure.doScalerVal(X_train=X_train,X_test=X_test,X_val=X_val)

# #oversample 
# X_train,y_train = structure.applysmote(X_train,y_train)

# #transform to tensors
# X_train,y_train= structure.makeTorch2(X_train,y_train)
# X_val,y_val= structure.makeTorch2(X_val,y_val)
# X_test,y_test= structure.makeTorch2(X_test,y_test)

# #define batch size
# bs=7

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
# model = MLP.ChurnModel(n_input_dim=X_train.shape[1])
# # model = MLP.WideModel()
# # model = MLP.DeepModel()
# loss_func = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# epochs = 80

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
# plots.plotlosses(epochs=epochs, train_losses=train_losses,  val_losses=val_losses, 
#                   outDir=PLOT_DIR, plotname='lossplot.png')

# print(train_losses[epochs-1])
# print(val_losses[epochs-1])

# #####GET RESULTS
# #calculate validation accuracy
# val_y, val_pred, val_loss = MLP.testModel(model, val_dl, y_val, loss_func)


# #create confusion matrix
# plots.CM(y_test=val_y, y_pred=val_pred, outDir=PLOT_DIR, name='CM.val.png',
#             CMmax=8, CMmin=0)

# #define model validity
# acc_val = MLout.printacc(ytest=val_y, ypred=val_pred)
# f1_val = MLout.printf1(ytest=val_y, ypred=val_pred)
# precision_val = MLout.printprecision(ytest=val_y, ypred=val_pred)
# recall_val = MLout.printrecall(ytest=val_y, ypred=val_pred)

# print(acc_val, f1_val, precision_val, recall_val)

# #calculate validation accuracy
# test_y, test_pred, test_loss = MLP.testModel(model, test_dl, y_test, loss_func)

# #create confusion matrix
# plots.CM(y_test=test_y, y_pred=test_pred, outDir=PLOT_DIR, name='CM.test.png',
#             CMmax=8, CMmin=0)

# #define model validity
# test_acc = MLout.printacc(ytest=test_y, ypred=test_pred)
# test_f1 = MLout.printf1(ytest=test_y, ypred=test_pred)
# test_precision = MLout.printprecision(ytest=test_y, ypred=test_pred)
# test_recall = MLout.printrecall(ytest=test_y, ypred=test_pred)
# test_auc = MLout.printauc(ytest=test_y, ypred=test_pred)

# print(test_acc, test_f1, test_precision, test_recall, test_auc)

##save model
##structure.savemodel(model=model, name='MLP.churn.pt', outDir=INFO_DIR)

#GOOD MODEL!!!
###############################################################################
# #STEP 3 HP training
# #######LOAD DATA FROM EXPLORATION AND INITIAL SANITY CHECKS
# df, names = structure.inData(dat = os.path.join(DATA_DIR,'data_csbii.csv'),
#                               lim =22)

# #######CLEAN UP DATA
# #Encoding 'status' as label 1 & 0 , naming the field as target
# df['target'] = df['Mix']
# structure.reClass(df=df, col='target')
# df.drop('Mix',axis = 1, inplace=True)

# #check dtypes
# df['target'] =df['target'].astype(float)
# #print(df.dtypes)

# #remove any missing values
# tmp = df.isnull().sum().reset_index(name='missing_val')
# tmp[tmp['missing_val']!= 0]

# random.seed(None)

# #test_train split
# X_train,X_test,y_train,y_test = structure.makeXy(df=df, 
#                                                   col='target',
#                                                   splitsize=0.15,
#                                                   rs=0)
# #create validation set
# X_train,X_val,y_train,y_val = structure.makeXyVal(X_train, y_train,
#                                                 splitsize=0.18,
#                                                 rs=0)

# #scale training and testing X vals
# X_train,X_test,X_val = structure.doScalerVal(X_train=X_train,X_test=X_test,X_val=X_val)

# #oversample 
# X_train,y_train = structure.applysmote(X_train,y_train)

# #transform to tensors
# X_train,y_train= structure.makeTorch2(X_train,y_train)
# X_val,y_val= structure.makeTorch2(X_val,y_val)
# X_test,y_test= structure.makeTorch2(X_test,y_test)


# #define hyperparameters
# model = MLP.ChurnModel(n_input_dim=X_train.shape[1])
# # model = MLP.WideModel()
# # model = MLP.DeepModel()
# loss_func = nn.BCELoss()

# #make lists
# bs = [4, 7, 14]
# e = [20, 40, 80, 160]
# lr = [0.1, 0.01, 0.001]

# hp = [dict(zip(('bs','lr','e'), (b,l,e))) for b,l,e in product(bs, lr, e)]

# savename = ['bs4lr01e20.txt','bs4lr01e40.txt','bs4lr01e80.txt','bs4lr01e160.txt',
#             'bs4lr001e20.txt','bs4lr001e40.txt','bs4lr001e80.txt','bs4lr001e160.txt',
#             'bs4lr0001e20.txt','bs4lr0001e40.txt','bs4lr0001e80.txt','bs4lr0001e160.txt',
#             'bs7lr01e20.txt','bs7lr01e40.txt','bs7lr01e80.txt','bs7lr01e160.txt',
#             'bs7lr001e20.txt','bs7lr001e40.txt','bs7lr001e80.txt','bs7lr001e160.txt',
#             'bs7lr0001e20.txt','bs7lr0001e40.txt','bs7lr0001e80.txt','bs7lr0001e160.txt',
#             'bs14lr01e20.txt','bs14lr01e40.txt','bs14lr01e80.txt','bs14lr01e160.txt',
#             'bs14lr001e20.txt','bs14lr001e40.txt','bs14lr001e80.txt','bs14lr001e160.txt',
#             'bs14lr0001e20.txt','bs14lr0001e40.txt','bs14lr0001e80.txt','bs14lr0001e160.txt']

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
    
#     save=os.path.join(churn_filt_DIR, savename[k])
#     all_stats_df.to_csv(save, sep=' ')
###############################################################################
#STEP 4 - average stats
##https://www.kaggle.com/code/ceshine/feature-importance-from-a-pytorch-model
########LOAD DATA FROM EXPLORATION AND INITIAL SANITY CHECKS
# df, names = structure.inData(dat = os.path.join(DATA_DIR,'data_csbii.csv'),
#                               lim =22)

# #######CLEAN UP DATA
# #Encoding 'status' as label 1 & 0 , naming the field as target
# df['target'] = df['Mix']
# structure.reClass(df=df, col='target')
# df.drop('Mix',axis = 1, inplace=True)

# #check dtypes
# df['target'] =df['target'].astype(float)
# #print(df.dtypes)

# #remove any missing values
# tmp = df.isnull().sum().reset_index(name='missing_val')
# tmp[tmp['missing_val']!= 0]

# random.seed(None)

# #test_train split
# X_train,X_test,y_train,y_test,i_train,i_test = structure.makeXyInd(df=df, 
#                                                   col='target',
#                                                   splitsize=0.15,
#                                                   rs=0)
# #create validation set
# X_train,X_val,y_train,y_val = structure.makeXyVal(X_train, y_train,
#                                                 splitsize=0.18,
#                                                 rs=0)

# #scale training and testing X vals
# X_train,X_test,X_val = structure.doScalerVal(X_train=X_train,X_test=X_test,X_val=X_val)

# #oversample 
# X_train,y_train = structure.applysmote(X_train,y_train)

# #transform to tensors
# X_train,y_train= structure.makeTorch2(X_train,y_train)
# X_val,y_val= structure.makeTorch2(X_val,y_val)
# X_test,y_test= structure.makeTorch2(X_test,y_test)

# #define batch size
# bs=4

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
# model = MLP.ChurnModel(n_input_dim=X_train.shape[1])
# # model = MLP.WideModel()
# # model = MLP.DeepModel()
# loss_func = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# epochs = 40

# train_step_fn = tensorfn.make_train_step_fun(model, loss_func, optimizer)
# val_step_fn = tensorfn.make_val_step_fun(model, loss_func)

# auc_list = []
# for i in range(0,100):
#     train_losses=[]
#     val_losses=[]
    
#     for epoch in range(epochs):
#         loss = tensorfn.mini_batch(device, train_dl, train_step_fn)
#         train_losses.append(loss)
    
#         with torch.no_grad():
#             val_loss = tensorfn.mini_batch(device, test_dl, val_step_fn)
#             val_losses.append(val_loss)
            
#     test_y, test_pred, test_loss = MLP.testModel(model, test_dl, y_test, loss_func)
    
#     test_auc = MLout.printauc(ytest=test_y, ypred=test_pred)
    
#     auc_list.append(test_auc)
    

# save=os.path.join(INFO_DIR, 'auc.csv')
# pred_df.to_csv(save, sep=',')

# #######TRAIN MODEL
# preds=[]
# for i in range(0,100):
#     train_losses=[]
#     val_losses=[]
    
#     for epoch in range(epochs):
#         loss = tensorfn.mini_batch(device, train_dl, train_step_fn)
#         train_losses.append(loss)
    
#         with torch.no_grad():
#             val_loss = tensorfn.mini_batch(device, test_dl, val_step_fn)
#             val_losses.append(val_loss)
    
#     #calculate validation accuracy
#     test_y, test_pred, test_loss = MLP.testModel(model, test_dl, y_test, loss_func)

#     preds.append(test_pred)
#     pred_df = pd.DataFrame(preds)
#     pred_df =  pred_df.transpose()
    
#     test_samples = names[i_test].reset_index()
#     test_bin = pd.Series(test_y, name='Mixture')
#     pred_df = pd.concat([pred_df, test_bin, test_samples], axis=1)
    
#     save=os.path.join(INFO_DIR, 'predictions.csv')
#     pred_df.to_csv(save, sep=',')

###########################################################################################
#STEP 5 MODEL UNDERSTANDING
########LOAD DATA FROM EXPLORATION AND INITIAL SANITY CHECKS
# df, names = structure.inData(dat = os.path.join(DATA_DIR,'data_csbii.csv'),
#                               lim =22)

# #######CLEAN UP DATA
# #Encoding 'status' as label 1 & 0 , naming the field as target
# df['target'] = df['Mix']
# structure.reClass(df=df, col='target')
# df.drop('Mix',axis = 1, inplace=True)

# #check dtypes
# df['target'] =df['target'].astype(float)
# #print(df.dtypes)

# #test_train split
# X_train,X_test,y_train,y_test,i_train,i_test = structure.makeXyInd(df=df, 
#                                                   col='target',
#                                                   splitsize=0.15,
#                                                   rs=0)
# #create validation set
# X_train,X_val,y_train,y_val = structure.makeXyVal(X_train, y_train,
#                                                 splitsize=0.18,
#                                                 rs=0)

# #scale training and testing X vals
# X_train,X_test,X_val = structure.doScalerVal(X_train=X_train,X_test=X_test,X_val=X_val)

# #oversample 
# X_train,y_train = structure.applysmote(X_train,y_train)

# #transform to tensors
# X_train,y_train= structure.makeTorch2(X_train,y_train)
# X_val,y_val= structure.makeTorch2(X_val,y_val)
# X_test,y_test= structure.makeTorch2(X_test,y_test)

# #define batch size
# bs=4
# loss_func = nn.BCELoss()

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

# # #load model
# modpath=os.path.join(INFO_DIR, 'MLP.churn.pt')
# model = torch.load(modpath)

#calculate validation accuracy
# test_y, test_pred, test_loss = MLP.testModel(model, test_dl, y_test, loss_func)
# test_acc = MLout.printacc(ytest=test_y, ypred=test_pred)
# test_f1 = MLout.printf1(ytest=test_y, ypred=test_pred)
# test_precision = MLout.printprecision(ytest=test_y, ypred=test_pred)
# test_recall = MLout.printrecall(ytest=test_y, ypred=test_pred)
# test_auc = MLout.printauc(ytest=test_y, ypred=test_pred)
