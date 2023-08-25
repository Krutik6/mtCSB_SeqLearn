# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:55:06 2023

@author: nkp68
"""

def storage(df, l):
    #save all column names except mix
    colnames = df.columns.values[0:l]
    #add prefix to each one
    prefix_ADD = 'Ablation_of_'
    ablation_names = [prefix_ADD + x for x in colnames if isinstance(x, str)]
    
    return colnames, ablation_names


def lofdf(df, colnames):
    #store ablated dfs in list
    df_ablation = []
    for i in colnames:
        df2 = df.drop([i], axis=1)
        df_ablation.append(df2)
        
    return df_ablation


from helperfunctions import structure

def ablsplit(df_abl, l):
    X_train_abl = []
    X_test_abl = []
    y_train_abl = []
    y_test_abl = []
    
    for i in range(0, l):
        X_train,X_test,y_train,y_test = structure.makeXy(df=df_abl[i], 
                                                      col='target',
                                                      splitsize=0.2,
                                                      rs=0)
        X_train_abl.append(X_train)
        X_test_abl.append(X_test)
        y_train_abl.append(y_train)
        y_test_abl.append(y_test)
        
    return X_train_abl, X_test_abl, y_train_abl, y_test_abl


def ablscale(X_train_abl, X_test_abl, l):
    X_train_scaled = []
    X_test_scaled = []
    
    for i in range(0,l):
        X_train_s,X_test_s = structure.doScaler(X_train_abl[i],X_test_abl[i])
    
        X_train_scaled.append(X_train_s)
        X_test_scaled.append(X_test_s)
    
    return X_train_scaled, X_test_scaled



def abltensor(X_train_np, X_test_np, y_train_np, y_test_np, l):
    X_train_tensor_list=[]
    X_test_tensor_list=[]
    y_train_tensor_list=[]
    y_test_tensor_list=[]
    
    for i in range(0,l):
        X_train,X_test,y_train,y_test = structure.makeTorch(X_train_np[i],
                                                            X_test_np[i],
                                                            y_train_np[i],
                                                            y_test_np[i])
        X_train_tensor_list.append(X_train)
        X_test_tensor_list.append(X_test)
        y_train_tensor_list.append(y_train)
        y_test_tensor_list.append(y_test)
        
    return X_train_tensor_list,X_test_tensor_list,y_train_tensor_list,y_test_tensor_list


from torch.utils.data import DataLoader, TensorDataset

def abldataloader(X_train_t,X_test_t,y_train_t,y_test_t, l, bs_train, bs_test):
    train_DL_list = []
    
    for i in range(0,l):
        ytrain_tensor = y_train_t[i].unsqueeze(1)
        train_ds = TensorDataset(X_train_t[i], ytrain_tensor)
        train_loader = DataLoader(train_ds, batch_size=bs_train)
        
        train_DL_list.append(train_loader)
    
    test_DL_list = []
    
    for i in range(0,l):
        ytest_tensor = y_test_t[i].unsqueeze(1)
        test_ds = TensorDataset(X_test_t[i], ytest_tensor)
        test_loader = DataLoader(test_ds, batch_size=bs_test)
    
        test_DL_list.append(test_loader)
    
    return train_DL_list, test_DL_list


from helperfunctions import MLP

def abltrain(model, epochs, train_DL, loss_func, optimizer, l):
    model_list = []
    train_lost_list = []

    for i in range(0,l):
        model, train_loss = MLP.trainModel(model=model, 
                                           epochs=epochs, 
                                           train_DL=train_DL[i], 
                                           loss_func=loss_func, 
                                           optimizer=optimizer)
        model_list.append(model)
        train_lost_list.append(train_loss)
        
    return model_list, train_lost_list


def abltest(model_list, test_DL, y_test_abl, l, loss_func):
    ytest_list = []
    ypred_list =[]

    for i in range(0, l):
        ytest, ytest_pred, test_loss = MLP.testModel(model=model_list[i], 
                                          test_DL=test_DL[i],
                                          y_test=y_test_abl[i],
                                          loss_func=loss_func)
        
        ytest_list.append(ytest)
        ypred_list.append(ytest_pred)
        
    return ytest_list,ypred_list

from helperfunctions import MLout

def ablscore(ytest_list, ypred_list, l):
    acc_list = []

    for i in range(0, l):
        score = MLout.printacc(ytest=ytest_list[i], ypred=ypred_list[i])
        acc_list.append(score)
        
    f1_list = []
    
    for i in range(0, l):
        score = MLout.printf1(ytest=ytest_list[i], ypred=ypred_list[i])
        f1_list.append(score)
        
    precision_list = []
    
    for i in range(0, l):
        score = MLout.printprecision(ytest=ytest_list[i], ypred=ypred_list[i])
        precision_list.append(score)
        
    recall_list = []
    
    for i in range(0, l):
        score = MLout.printrecall(ytest=ytest_list[i], ypred=ypred_list[i])
        recall_list.append(score)
        
    return acc_list,f1_list,precision_list,recall_list


def abloversample(X_train_abl, y_train_abl, l):
    X_train_oversampled = []
    y_train_oversampled = []
    
    for i in range(0,l):
       X_train_os,y_train_os = structure.applysmote(X_train_abl[i], y_train_abl[i])
    
       X_train_oversampled.append(X_train_os)
       y_train_oversampled.append(y_train_os)
    
    return X_train_oversampled, y_train_oversampled