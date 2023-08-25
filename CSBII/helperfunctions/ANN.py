# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:39:08 2023

@author: nkp68
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN_Model(nn.Module):
    def __init__(self, input_feature=8, hidden1=20, hidden2=20, output_features=2):
        super().__init__()
        self.f_connected1=nn.Linear(input_feature, hidden1)
        self.f_connected2=nn.Linear(hidden1, hidden2)
        self.out=nn.Linear(hidden2, output_features)
    def forward(self, x):
        x=F.relu(self.f_connected1(x))
        x=F.relu(self.f_connected2(x))
        x=self.out(x)
        return x
    
    
    
def testANN(X_test, model):
    predictions=[]
    with torch.no_grad():
        for i, data in enumerate(X_test):
            y_pred = model(data)
            predictions.append(y_pred.argmax().item())
            #print(y_pred.argmax().item())
            
    return predictions