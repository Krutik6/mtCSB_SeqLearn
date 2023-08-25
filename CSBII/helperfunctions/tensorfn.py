# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 08:32:11 2023

@author: nkp68
"""

def make_train_step_fun(model, loss_fn, optimizer):
    
    def perform_trainstep_fn(x, y):
        #set model to train mode
        model.train()
        #forward pass
        yhat = model(x)
        #compute loss
        loss = loss_fn(yhat, y)
        #compute gradients
        loss.backward()
        #update params
        optimizer.step()
        optimizer.zero_grad()
        
        #return loss
        return loss.item()
    
    return perform_trainstep_fn

def make_val_step_fun(model, loss_fn):
    
    def perform_val_step_fn(x, y):
        
        #set model to eval mode
        model.eval()
        #forward pass
        yhat = model(x)
        #compute loss
        loss = loss_fn(yhat, y)
        #return loss
        return loss.item()
    
    return perform_val_step_fn

import numpy as np

def mini_batch(device, data_loader, step_fn):
    
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
        
    loss = np.mean(mini_batch_losses)
    return loss

        

    
