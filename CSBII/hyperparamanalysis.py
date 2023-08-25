# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:52:36 2023

@author: nkp68

Script is to assess which set of hyperparameters lead to the best trained model
"""

#load helper module
import os
from glob import glob
from helperfunctions import *

#load essential libraries
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from matplotlib.colors import ListedColormap


#set global variables
WORKING_DIRECTORY = os.path.dirname(__file__)
PARENT_DIR = os.path.join(WORKING_DIRECTORY, '../')
DATA_DIR = os.path.join(PARENT_DIR, 'data/')
TMP_DIR = os.path.join(PARENT_DIR, 'tmp/')
INFO_DIR = os.path.join(PARENT_DIR, 'info/')
PLOT_DIR = os.path.join(PARENT_DIR, 'plots/')
churn_norm_DIR = os.path.join(INFO_DIR, 'churn_norm/')
wide_norm_DIR = os.path.join(INFO_DIR, 'wide_norm/')
deep_norm_DIR = os.path.join(INFO_DIR, 'deep_norm/')
CSBI_DIR = os.path.join(PARENT_DIR, 'CSBI/')
CSBI_CHURN_DIR = os.path.join(CSBI_DIR, 'churn/')
CSBIII_DIR = os.path.join(PARENT_DIR, 'CSBIII/')
CSBIII_CHURN_DIR = os.path.join(CSBIII_DIR, 'churn/')

###################################################################################
#HP TRAIN 1 NORM. WITH LOSSES
# DIR = CSBIII_CHURN_DIR

# files = os.listdir(DIR)
# file = files[0]

# txt = []  
# for i in files:
#     i = os.path.join(DIR, i)
#     file=pd.read_table(i, sep=' ', header=0, on_bad_lines='skip')
#     txt.append(file)

# loss_columns = [df.iloc[:,1:3] for df in txt]
# files = [filename.replace('.txt', '') for filename in files]

# #find mean and STDV for each loss
# column_means = [df.mean() for df in loss_columns]
# column_STDV = [df.std() for df in loss_columns]

# means = pd.concat(column_means, axis=1)
# norm_means = means.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
# norm_means.columns = files

# #add params as new rows
# bs = [re.search(r'bs(\d+)lr', string).group(1) for string in files]
# lr = [re.search(r'lr(\d+)e', string).group(1) for string in files]
# e = [re.search(r'e(\d+)', string).group(1) for string in files]

# lr = [s[:1] + '.' + s[1:] for s in lr]

# bs = list(map(int, bs))
# e = list(map(int, e))
# lr = list(map(float, lr))

# norm_means.loc['batch_size'] = bs
# norm_means.loc['epochs'] = e
# norm_means.loc['learning_rate'] = lr

# #alter column order
# column_order = [9,10,11,8,5,6,7,4,1,2,3,0]
# column_order2 = [x+12 for x in column_order]
# column_order3 = [x+24 for x in column_order]
# column_order = column_order + column_order2 + column_order3

# means = norm_means.iloc[:, column_order]
# df = means.transpose()

# plots.plotHPloss(df, CSBIII_DIR, 'HP_losses_churn.png')
###############################################################################
#HP TRAIN 3 NORM. WITH VAL
# DIR=CSBIII_CHURN_DIR

# files = os.listdir(DIR)
# file = files[0]

# txt = []  
# for i in files:
#     i = os.path.join(DIR, i)
#     file=pd.read_table(i, sep=' ', header=0, on_bad_lines='skip')
#     txt.append(file)

# val_columns = [df.iloc[:,3:7] for df in txt]
# files = [filename.replace('.txt', '') for filename in files]

# #find mean and STDV for each loss
# column_means = [df.mean() for df in val_columns]
# column_STDV = [df.std() for df in val_columns]

# means = pd.concat(column_means, axis=1)
# stdv = pd.concat(column_STDV, axis=1)
# #convert to SEM
# stdv = stdv/25

# #add params as new rows
# bs = [re.search(r'bs(\d+)lr', string).group(1) for string in files]
# lr = [re.search(r'lr(\d+)e', string).group(1) for string in files]
# e = [re.search(r'e(\d+)', string).group(1) for string in files]
# lr = [s[:1] + '.' + s[1:] for s in lr]

# bs = list(map(int, bs))
# e = list(map(int, e))
# lr = list(map(float, lr))

# means.loc['batch_size'] = bs
# means.loc['epochs'] = e
# means.loc['learning_rate'] = lr

# means.columns = files
# stdv.columns = files

# #alter column order
# column_order = [9,10,11,8,5,6,7,4,1,2,3,0,21,22,23,20,17,18,19,16,13,14,15,12,33,34,35,32,29,30,31,28,25,26,27,24]
# means = means.iloc[:, column_order]
# df = means.transpose()
# stdv = stdv.iloc[:, column_order]
# df_stdv = stdv.transpose()
# df_stdv = df_stdv.add_suffix('_stdv')
# df = pd.concat([df, df_stdv], axis=1)
# df.insert(0, 'Index', df.index)

# #convert parameters into colours
# def map_to_color1(value):
#     if value == 15:
#         return 'lightpurple'
#     elif value == 31:
#         return 'lightorange'
#     elif value == 62:
#         return 'lightgreen'
#     else:
#         return 'white' 
    
# def map_to_color2(value):
#     if value == 20:
#         return 'black'
#     elif value == 40:
#         return 'magenta'
#     elif value == 80:
#         return 'yellow'
#     else:
#         return 'white' 
    
# def map_to_color3(value):
#     if value == 0.1:
#         return '//'
#     elif value == 0.01:
#         return 'x'
#     elif value == 0.001:
#         return 'o'
#     else:
#         return 'white' 
    
# df['batch_size'] = df['batch_size'].map(map_to_color1)
# df['epochs'] = df['epochs'].map(map_to_color2)
# df['learning_rate'] = df['learning_rate'].map(map_to_color3)

# num_rows = len(df['Val_accuracy'])

# combined_values = []
# for row in range(num_rows):
#     for col in ['Val_accuracy', 'Val_f1', 'Val_precision', 'Val_recall']:
#         combined_values.append(df[col][row])
        
# combined_err = []
# for row in range(num_rows):
#     for col in ['Val_accuracy_stdv', 'Val_f1_stdv', 'Val_precision_stdv', 'Val_recall_stdv']:
#         combined_err.append(df[col][row])  
        
# index_array = df['Index'].values
# index_array = np.tile(index_array, 4)

# bs_array = df['batch_size'].values
# bs_array = np.tile(bs_array, 4)

# e_array = df['epochs'].values
# e_array = np.tile(e_array, 4)

# lr_array = df['learning_rate'].values
# lr_array = np.tile(lr_array, 4)

# col_array = np.tile(['red', 'darkblue', 'orange', 'purple'], 36)

# df = pd.DataFrame([index_array, combined_values,combined_err, bs_array, e_array, lr_array, col_array])
# df = df.transpose()
# df.columns=['Index', 'Score', 'STDV', 'batch_size', 'epochs', 'learning_rate', 'TYPE']

# plots.plotHPval(df, CSBIII_DIR, "churn_val_scores.png", 0.3)
#plots.plotHPval(df, PLOT_DIR, "deep_norm_val_scores.png")
#plots.plotHPval(df, PLOT_DIR, "wide_norm_val_scores.png")

######################################################################################################
#HP3 print which is best
DIR = CSBI_CHURN_DIR

files = os.listdir(DIR)
file = files[0]

txt = []  
for i in files:
    i = os.path.join(DIR, i)
    file=pd.read_table(i, sep=' ', header=0, on_bad_lines='skip')
    txt.append(file)

# columns = [df.iloc[:,1:7] for df in txt]
files = [filename.replace('.txt', '') for filename in files]

#find mean and STDV for each loss
column_means = [df.mean() for df in txt]
column_STDV = [df.std() for df in txt]


means = pd.concat(column_means, axis=1)
means = means.drop(means.index[0])
means.columns = files

#filt1
#remove any columns where Val loss and Train loss are >0.5
columns_to_remove = means.columns[(means.loc['Val_loss'] > 1) | (means.loc['Train_loss'] > 1)]
filt_means = means.drop(columns=columns_to_remove)
#filt2
#only keep columns where train loss was lower than val loss
columns_to_keep = filt_means.columns[filt_means.loc['Val_loss'] > filt_means.loc['Train_loss']]
filt_means = filt_means[columns_to_keep]

#now select HP with highest precision
row_index = 'Val_precision'
highest_column = filt_means.loc[row_index].idxmax()

print("Column name with the highest value in row", row_index, ":", highest_column)

