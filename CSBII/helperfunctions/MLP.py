# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:09:01 2023

@author: nkp68
"""

def sepCatsandConts(df, val=0.002):
    df_data = df.iloc[:, :-1]
    likely_cat = {}
    for var in df_data.iloc[:,1:].columns:
        likely_cat[var] = 1.*df_data[var].nunique()/df_data[var].count() < 0.002 

    num_cols = []
    cat_cols = []
    for col in likely_cat.keys():
        if (likely_cat[col] == False):
            num_cols.append(col)
        else:
            cat_cols.append(col)
            
    return likely_cat, num_cols, cat_cols



import torch.nn as nn


class ChurnModel(nn.Module):
    def __init__(self, n_input_dim, n_hidden1=14, n_hidden2=7, n_output=1):
        super(ChurnModel, self).__init__()
        self.layer_1 = nn.Linear(n_input_dim, n_hidden1) 
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer_out = nn.Linear(n_hidden2, n_output) 
        
        
        self.relu = nn.ReLU()
        self.sigmoid =  nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden2)
        
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        
        return x
    
    
    
class WideModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(14, 42)
        self.relu = nn.ReLU()
        self.output = nn.Linear(42, 1)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.sigmoid(self.output(x))
        
        return x
    
    
class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(14, 14)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(14, 14)
        self.relu2 = nn.ReLU()
        self.hidden3 = nn.Linear(14, 14)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(14, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        x = self.relu1(self.hidden1(x))
        x = self.relu2(self.hidden2(x))
        x = self.relu3(self.hidden3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.output(x))
        
        return x
        
    
def trainModel(model, epochs, train_DL, loss_func, optimizer):
    model.train()
    train_loss = []
    for epoch in range(epochs):
        #Within each epoch run the subsets of data = batch sizes.
        for xb, yb in train_DL:
            y_pred = model(xb)            # Forward Propagation
            loss = loss_func(y_pred, yb)  # Loss Computation
            optimizer.zero_grad()         # Clearing all previous gradients, setting to zero 
            loss.backward()               # Back Propagation
            optimizer.step()              # Updating the parameters 
        #print("Loss in iteration :"+str(epoch)+" is: "+str(loss.item()))
        train_loss.append(loss.item())
    #print('Last iteration loss value: '+str(loss.item()))
    
    return model, train_loss


import torch
import itertools

def testModel(model, test_DL, y_test, loss_func):
    y_pred_list = []
    model.eval()
    loss = 0
    
    with torch.no_grad():
        for xb_test,yb_test in test_DL:
            y_test_pred = model(xb_test)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.detach().numpy())
            
            loss += loss_func(y_test_pred, yb_test)
              
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    
    ytest_pred = list(itertools.chain.from_iterable(y_pred_list))
    ytest = y_test.tolist()
    
    test_loss = loss / len(test_DL)
    
    return ytest, ytest_pred, test_loss


class MyExtractor:
    def __init__(self, extractor, features = 20):
        self.extractor = extractor
        self.projection = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LazyLinear(features))

    def forward(self, x):
        return self.projection(self.extractor(x))
    
    
    
import torch.nn as nn


class ChurnModel_CSBI(nn.Module):
    def __init__(self, n_input_dim, n_hidden1=10, n_hidden2=5, n_output=1):
        super(ChurnModel_CSBI, self).__init__()
        self.layer_1 = nn.Linear(n_input_dim, n_hidden1) 
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer_out = nn.Linear(n_hidden2, n_output) 
        
        
        self.relu = nn.ReLU()
        self.sigmoid =  nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden2)
        
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        
        return x
    
    
    
    
import torch.nn as nn


class ChurnModel_CSBIII(nn.Module):
    def __init__(self, n_input_dim, n_hidden1=11, n_hidden2=5, n_output=1):
        super(ChurnModel_CSBIII, self).__init__()
        self.layer_1 = nn.Linear(n_input_dim, n_hidden1) 
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer_out = nn.Linear(n_hidden2, n_output) 
        
        
        self.relu = nn.ReLU()
        self.sigmoid =  nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden2)
        
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        
        return x