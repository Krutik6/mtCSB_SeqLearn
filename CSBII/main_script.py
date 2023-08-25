# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:41:38 2023

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
import csv

#import nerual network libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchsummary import summary

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

#set global variables
device = torch.device('cpu')
WORKING_DIRECTORY = os.path.dirname(__file__)
PARENT_DIR = os.path.join(WORKING_DIRECTORY, '../')
DATA_DIR = os.path.join(PARENT_DIR, 'data/')
TMP_DIR = os.path.join(PARENT_DIR, 'tmp/')
INFO_DIR = os.path.join(PARENT_DIR, 'info/')
PLOT_DIR = os.path.join(PARENT_DIR, 'plots/')
###############################################################################
#############STEP 0 TEST USING SIMPLE ANN######################################
#Prepare data for an ANN
#load data
# df, names = structure.inData(dat = os.path.join(DATA_DIR,'OccuranceCSBII.csv'), lim =15)

#check for NANs
#preprocess.printNan(df)   

#check datatype in each column
#print(df.dtypes)

#plot classes to check abundance
#plots.plotClasses(df=df, col='Mix', outdir=PLOT_DIR)

#make pair plot to check variance between dependent and independent feaures
#plots.pairPlot(df=df, col='Mix', outdir=PLOT_DIR)

#remove weaker features
# df.drop(df.columns[[0, 3, 4, 5, 7]], axis=1, inplace=True)

#plots.pairPlot(df=df, col='Mix', outdir=PLOT_DIR, plotname='featureEngineer.pairplot.png')

#re-class yes:1 and no:0
# structure.reClass(df=df, col='Mix')

#create X and y data
# X_train,X_test,y_train,y_test = structure.makeXy(df=df, 
#                                                   col='Mix',
#                                                   splitsize=0.2,
#                                                   rs=0)

#turn the numpy objects into tensores
# X_train,X_test,y_train,y_test = structure.makeTorch(X_train,
                                                    # X_test,
                                                    # y_train,  
                                                    #  y_test)

#Apply a simple ANN
#initiate ANN
# torch.manual_seed(20)
# model=ANN.ANN_Model()
# print(model)   
#define backward propogation - loss function, optimizer
# loss_function=nn.CrossEntropyLoss()
# optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
# run model
# epochs=1000 
# final_losses=[]
# final_losses_item=[]
# for i in range(epochs):
    # i=i+1
    # y_pred=model.forward(X_train)
    # loss=loss_function(y_pred, y_train)
    # final_losses.append(loss)
    # final_losses_item.append(loss.item())
    
    #if i%10==1:
    #    print("Epoch number : {} and the loss : {}". format(i, loss.item()))
        
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
#plot model performance on training data
# plots.plotTrain(epochs=epochs, listofloss=final_losses_item, 
                # outDir=PLOT_DIR, name='ANN.trainPerformance.png')
#test model performance by creating y_pred
# y_pred = ANN.testANN(X_test=X_test, model=model)
#create confusion matrix
# plots.CM(y_test=y_test, y_pred=y_pred, outDir=PLOT_DIR, name='ANN.CM.png',
          # CMmax=8, CMmin=0)
#accuracy score
# score=accuracy_score(y_test, y_pred)
# print(score) #0.68
#save the model
# structure.savemodel(model=model, name='ANN.model.68P.pt', outDir=INFO_DIR)
###############################################################################
#############STEP 1 USE MLP ANN WITH NO SEL####################################
#MLP ANN with more advanced NN building + scaling
#load data
# df, names = structure.inData(dat = os.path.join(DATA_DIR,'OccuranceCSBII.csv'),
#                               lim =15)

# #plot classes to check abundance
# #plots.plotClasses(df=df, col='Mix', outdir=PLOT_DIR)

# #make pair plot to check variance between dependent and independent feaures
# #plots.pairPlot(df=df, col='Mix', outdir=PLOT_DIR)

# #Encoding 'status' as label 1 & 0 , naming the field as target
# df['target'] = df['Mix']
# structure.reClass(df=df, col='target')
# df.drop('Mix',axis = 1, inplace=True)

# #check dtypes
# df['target'] =df['target'].astype(float)
# print(df.dtypes)

# #remove any missing values
# tmp = df.isnull().sum().reset_index(name='missing_val')
# tmp[tmp['missing_val']!= 0]

# #separate catagorical features and continous features
# likely_cat, num_cols, cat_cols = MLP.sepCatsandConts(df=df, val=0.002)

# #create correlation plot for continous features
# plots.correlationPlot(df=df, OutDir=PLOT_DIR, name='Correlations.MLP.png',
#                       num_cols=num_cols)


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
# bs_train=73
# bs_test=19

# #combine both train and test data into dataloaders objects
# y_tensor = y_train.unsqueeze(1)
# train_ds = TensorDataset(X_train, y_tensor)
# train_dl = DataLoader(train_ds, batch_size=bs_train, shuffle=True)

# ytest_tensor = y_test.unsqueeze(1)
# test_ds = TensorDataset(X_test, ytest_tensor)
# test_loader = DataLoader(test_ds, batch_size=bs_test)

# #create the MLP model
# model = MLP.ChurnModel(n_input_dim=X_train.shape[1])

# #define hyperparameters
# loss_func = nn.BCELoss()
# learning_rate = 0.1
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# epochs = 500

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

# print(test_loss.item())

# #create confusion matrix
# plots.CM(y_test=ytest, y_pred=ytest_pred, outDir=PLOT_DIR, name='MLP.CM.shuffle.png',
#             CMmax=10, CMmin=0)

# #accuracy score
# acc = MLout.printacc(ytest=ytest, ypred=ytest_pred)
# f1 = MLout.printf1(ytest=ytest, ypred=ytest_pred)
# precision = MLout.printprecision(ytest=ytest, ypred=ytest_pred)
# recall = MLout.printrecall(ytest=ytest, ypred=ytest_pred)

# #save model
# structure.savemodel(model=model, name='MLP.model.shuffle.pt', outDir=INFO_DIR)
###############################################################################
#############STEP 2 ABLATION TO FIND FEATURES##################################
#keep all hyperparameters the same 
#perform ablation i.e use all features - then one by one

#import df
# df, names = structure.inData(dat = os.path.join(DATA_DIR,'OccuranceCSBII.csv'),
#                               lim =15)

# #drop features which only occur in a minority of samples
# #df.drop(df.columns[[0, 3, 4, 5, 6]], axis=1, inplace=True)
# # df.drop(df.columns[[0, 3, 4, 5]], axis=1, inplace=True)

# #Encoding 'status' as label 1 & 0 , naming the field as target
# df['target'] = df['Mix']
# structure.reClass(df=df, col='target')
# df.drop('Mix',axis = 1, inplace=True)

# # #save all column names except mix
# l = len(df.columns) -1
# colnames, a_list = ablation.storage(df=df, l=l)

# #create a new df for each ablation of 
# df_abl = ablation.lofdf(df=df, colnames=colnames)

# #create X_train, X_test, y_train and y_test for each df
# X_train_abl,X_test_abl,y_train_abl,y_test_abl = ablation.ablsplit(df_abl=df_abl, l=l)

# #scale training and testing X vals
# X_train_scaled,X_test_scaled = ablation.ablscale(X_train_abl=X_train_abl,
#                                                   X_test_abl=X_test_abl, l=l)

# #apply oversampling to train values
# X_train_oversampled, y_train_oversampled = ablation.abloversample(X_train_scaled,
#                                                                   y_train_abl,
#                                                                   l=l)

# #remove unneeded objects
# del(X_train_abl, X_test_abl, df_abl)

# #transform all data into tensors
# X_train_abl,X_test_abl,y_train_abl,y_test_abl = ablation.abltensor(X_train_np=X_train_oversampled,
#                                                                     X_test_np=X_test_scaled,
#                                                                     y_train_np=y_train_oversampled,
#                                                                     y_test_np=y_test_abl,
#                                                                     l=l)

# #define batch sizes
# bs_train=73
# bs_test=19

# #combine both train and test data into dataloaders objects
# train_DL,test_DL=ablation.abldataloader(X_train_t=X_train_abl,
#                             X_test_t=X_test_abl,
#                             y_train_t=y_train_abl,
#                             y_test_t=y_test_abl,
#                             l=l, bs_train=bs_train, bs_test=bs_test)

# #create the MLP model
# model = MLP.ChurnModel(n_input_dim=X_train_scaled[1].shape[1])

# #define hyperparameters
# loss_func = nn.BCELoss()
# learning_rate = 0.1
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# epochs = 500

# #train models
# model_list, train_loss_list=ablation.abltrain(model=model,
#                                               epochs=epochs,
#                                               train_DL=train_DL, 
#                                               loss_func=loss_func,
#                                               optimizer=optimizer,
#                                               l=l)

# #test models
# ytest_list,ypred_list = ablation.abltest(model_list=model_list,test_DL=test_DL, 
#                                           y_test_abl=y_test_abl,l=l,loss_func=loss_func)

# #retreive accuracy scores
# acc_list,f1_list,precision_list,recall_list=ablation.ablscore(ytest_list=ytest_list,
#                                                               ypred_list=ypred_list,
#                                                               l=l)
    
# abl = list(zip(colnames,acc_list,f1_list,precision_list,recall_list))
# abl = pd.DataFrame(list(abl))
# abl.columns = ['Ablated','Test_Acc', 'F1', 'Precision', 'Recall']
# save= os.path.join(INFO_DIR, 'ablation.1.score.txt')
# with open(save, "w") as output:
#     output.write(str(abl))

# tl = pd.DataFrame(train_loss_list)
# tl.index=colnames
# save= os.path.join(INFO_DIR, 'ablation.1.loss.txt')
# tl.to_csv(save, sep=' ')
###############################################################################
#############STEP 3 TEST NEW SET OF FEATURES###################################
# df, names = structure.inData(dat = os.path.join(DATA_DIR,'OccuranceCSBII.csv'),
#                               lim =15)
# #drop feautres with could be rare coding errors or technical glitches
# df.drop(df.columns[[0, 3, 4, 5, 6]], axis=1, inplace=True)

# #Encoding 'status' as label 1 & 0 , naming the field as target
# df['target'] = df['Mix']
# structure.reClass(df=df, col='target')
# df.drop('Mix',axis = 1, inplace=True)

# #check dtypes
# df['target'] =df['target'].astype(float)
# print(df.dtypes)

# #create correlation plot for continous features
# likely_cat, num_cols, cat_cols = MLP.sepCatsandConts(df=df, val=0.002)
# plots.correlationPlot(df=df, OutDir=PLOT_DIR, name='Correlations.selected.png',
#                       num_cols=num_cols)

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
# bs_train=73
# bs_test=19

# #combine both train and test data into dataloaders objects
# y_tensor = y_train.unsqueeze(1)
# train_ds = TensorDataset(X_train, y_tensor)
# train_dl = DataLoader(train_ds, batch_size=bs_train, shuffle=True)

# ytest_tensor = y_test.unsqueeze(1)
# test_ds = TensorDataset(X_test, ytest_tensor)
# test_loader = DataLoader(test_ds, batch_size=bs_test)

# #create the MLP model
# model = MLP.ChurnModel(n_input_dim=X_train.shape[1])

# #define hyperparameters
# loss_func = nn.BCELoss()
# learning_rate = 0.1
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# epochs = 500

# #train model
# model, train_loss =MLP.trainModel(model=model, epochs=epochs, train_DL=train_dl, 
#                 loss_func=loss_func, optimizer=optimizer)

# #plot model performance on training data
# plots.plotTrain(epochs=epochs, listofloss=train_loss, 
#                 outDir=PLOT_DIR, name='MLP.trainPerformance.selected.png')

# #compare prediction of test set with real test set
# ytest, ytest_pred, test_loss = MLP.testModel(model=model, test_DL=test_loader,
#                                              y_test=y_test, loss_func=loss_func)
# print(test_loss.item())

# #create confusion matrix
# plots.CM(y_test=ytest, y_pred=ytest_pred, outDir=PLOT_DIR, name='MLP.CM.selected.png',
#             CMmax=10, CMmin=0)

# #accuracy score
# acc = MLout.printacc(ytest=ytest, ypred=ytest_pred)
# f1 = MLout.printf1(ytest=ytest, ypred=ytest_pred)
# precision = MLout.printprecision(ytest=ytest, ypred=ytest_pred)
# recall = MLout.printrecall(ytest=ytest, ypred=ytest_pred)

# #save model
# structure.savemodel(model=model, name='MLP.model.selected.pt', outDir=INFO_DIR)
###############################################################################
##################STEP 4 Identify best hyperparams#############################
#Perform for-loop to identify ideal hyper-parameters
df, names = structure.inData(dat = os.path.join(DATA_DIR,'OccuranceCSBII.csv'),
                              lim =15)

#drop weak features
df.drop(df.columns[[0, 3, 4, 5, 6]], axis=1, inplace=True)

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

#separate catagorical features and continous features
likely_cat, num_cols, cat_cols = MLP.sepCatsandConts(df=df, val=0.002)

#create correlation plot for continous features
plots.correlationPlot(df=df, OutDir=PLOT_DIR, name='Correlations.MLP.png',
                      num_cols=num_cols)


#test_train split
X_train,X_test,y_train,y_test = structure.makeXy(df=df, 
                                                  col='target',
                                                  splitsize=0.2,
                                                  rs=0)

#scale training and testing X vals
X_train,X_test = structure.doScaler(X_train,X_test)

#oversample 
X_train,y_train = structure.applysmote(X_train,y_train)

#transform to tensors
X_train,X_test,y_train,y_test = structure.makeTorch(X_train,X_test,
                                                    y_train,y_test)
#define batch size
bs_train=73
bs_test=19

#combine both train and test data into dataloaders objects
y_tensor = y_train.unsqueeze(1)
train_ds = TensorDataset(X_train, y_tensor)
train_dl = DataLoader(train_ds, batch_size=bs_train, shuffle=True)

ytest_tensor = y_test.unsqueeze(1)
test_ds = TensorDataset(X_test, ytest_tensor)
test_loader = DataLoader(test_ds, batch_size=bs_test)

#create the MLP model
model = MLP.ChurnModel(n_input_dim=X_train.shape[1])

#create loop for different model hyperparameters
savenames=['lr0001_ep1000_opt_adam_BCEloss.txt']
lr = [0.001]
e = [1000]
hp = [dict(zip(('lr','e'), (i,j))) for i,j in product(lr,e)]

for i in range(len(hp)):
    print(savenames[i])
    print(hp[i]['lr'])
    print(hp[i]['e'])
    
for k in range(len(hp)):
    accuracy_list = []
    loss_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    
    for i in range(0,100):
        #define hyperparameters
        loss_func = nn.BCELoss()
        learning_rate = hp[k]['lr']
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        epochs = hp[k]['e']
    
        #train model
        model, train_loss =MLP.trainModel(model=model, epochs=epochs,
                                          train_DL=train_dl, loss_func=loss_func,
                                          optimizer=optimizer)
    
        loss_score = train_loss[epochs-1]
    
        #compare prediction of test set with real test set
        ytest, ytest_pred, test_loss = MLP.testModel(model=model, 
                                                      test_DL=test_loader, 
                                                      y_test=y_test,
                                                      loss_func=loss_func) 
        #accuracy score
        accuracy = MLout.printacc(ytest=ytest, ypred=ytest_pred)
        f1 = MLout.printf1(ytest=ytest, ypred=ytest_pred)
        precision = MLout.printprecision(ytest=ytest, ypred=ytest_pred)
        recall = MLout.printrecall(ytest=ytest, ypred=ytest_pred)
    
        loss_list.append(loss_score)
        accuracy_list.append(accuracy)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        
        
    all_stats = list(zip(loss_list,accuracy_list,f1_list,
                                            precision_list,recall_list))
    all_stats_df = pd.DataFrame(list(all_stats))
    
    all_stats_df.columns = ['Train_loss', 'Test_Acc', 'F1', 'Precision', 'Recall']
    
    save=os.path.join(INFO_DIR, savenames[k])
    all_stats_df.to_csv(save, sep=' ')
###############################################################################
###############STEP 5 CHECK FOR OVERFITTING####################################
# df, names = structure.inData(dat = os.path.join(DATA_DIR,'OccuranceCSBII.csv'),
#                               lim =15)
# #drop feautres with could be rare coding errors or technical glitches
# df.drop(df.columns[[0, 3, 4, 5, 6]], axis=1, inplace=True)

# #Encoding 'status' as label 1 & 0 , naming the field as target
# df['target'] = df['Mix']
# structure.reClass(df=df, col='target')
# df.drop('Mix',axis = 1, inplace=True)

# #check dtypes
# df['target'] =df['target'].astype(float)

# #test_train split
# X_train,X_test,y_train,y_test = structure.makeXy(df=df, 
#                                                   col='target',
#                                                   splitsize=0.2,
#                                                   rs=0)

# #scale training and testing X vals
# X_train,X_test = structure.doScaler(X_train,X_test)

# #transform to tensors
# X_train,X_test,y_train,y_test = structure.makeTorch(X_train,X_test,
#                                                     y_train,y_test)
# #define batch size
# bs_train=73
# bs_test=19

# #combine both train and test data into dataloaders objects
# y_tensor = y_train.unsqueeze(1)
# train_ds = TensorDataset(X_train, y_tensor)
# train_dl = DataLoader(train_ds, batch_size=bs_train, shuffle=True)

# ytest_tensor = y_test.unsqueeze(1)
# test_ds = TensorDataset(X_test, ytest_tensor)
# test_loader = DataLoader(test_ds, batch_size=bs_test)

# #create the MLP model
# model = MLP.ChurnModel(n_input_dim=X_train.shape[1])

# #define hyperparameters
# loss_func = nn.BCELoss()
# learning_rate = 0.01
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# epochs = 100

# #train model
# model, train_loss =MLP.trainModel(model=model, epochs=epochs, train_DL=train_dl, 
#                 loss_func=loss_func, optimizer=optimizer)

# #plot model performance on training data
# plots.plotTrain(epochs=epochs, listofloss=train_loss, 
#                 outDir=PLOT_DIR, name='MLP.trainPerformance.optimal.png')

# #compare prediction of test set with real test set
# ytest, ytest_pred, test_loss = MLP.testModel(model=model, test_DL=test_loader,
#                                               y_test=y_test, loss_func=loss_func)
# print(test_loss.item())#5.5
# #create confusion matrix
# plots.CM(y_test=ytest, y_pred=ytest_pred, outDir=PLOT_DIR, name='MLP.CM.optimal.png',
#             CMmax=10, CMmin=0)

# #accuracy score
# acc = MLout.printacc(ytest=ytest, ypred=ytest_pred)
# f1 = MLout.printf1(ytest=ytest, ypred=ytest_pred)
# precision = MLout.printprecision(ytest=ytest, ypred=ytest_pred)
# recall = MLout.printrecall(ytest=ytest, ypred=ytest_pred)

# #save model
# structure.savemodel(model=model, name='MLP.model.optimal.pt', outDir=INFO_DIR)

###############################################################################
#THINGS TO DO
#to do  - new script - plots and analysis
        #- check if the best hyperparamers (fastest + best accuracy are leading to overfitting)