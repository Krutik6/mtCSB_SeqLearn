# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:15:41 2023

@author: nkp68
"""
#load helper module
import os
from glob import glob
from helperfunctions import *

#load essential libraries
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import figure
import seaborn as sns
import numpy as np
import math


#set global variables
WORKING_DIRECTORY = os.path.dirname(__file__)
PARENT_DIR = os.path.join(WORKING_DIRECTORY, '../')
DATA_DIR = os.path.join(PARENT_DIR, 'data/')
TMP_DIR = os.path.join(PARENT_DIR, 'tmp/')
INFO_DIR = os.path.join(PARENT_DIR, 'info/')
PLOT_DIR = os.path.join(PARENT_DIR, 'plots/')
STAT_DIR = os.path.join(INFO_DIR, 'stats/')
CSBI_DIR = os.path.join(PARENT_DIR, 'CSBI/')
CSBIII_DIR = os.path.join(PARENT_DIR, 'CSBIII/')
###############################################################################
#load predicitons file
DIR=STAT_DIR

pred_file = os.path.join(DIR, 'predictions.csv')

df = pd.read_csv(pred_file, index_col=0)

#First plot show the number of times each test sample was correctly predicted
df_sort = df.sort_values(by='index', ascending=True)
df_names = df_sort['Sample'].values 
df_mixture = df_sort['Mixture'].values 
df = df_sort.iloc[:, :-3]
df = df.reset_index(drop=True)
df.index = df_names

#find percentages
total_columns = len(df.columns)
df['Percentage_1'] = (df.eq(1).sum(axis=1) / total_columns) * 100
df['Percentage_2'] = (df.eq(0).sum(axis=1) / total_columns) * 100
#save=os.path.join(DIR, 'predictions_percentages.csv')
# df.to_csv(save, sep=',')
# p1 = df['Percentage_1']
# p2 = df['Percentage_2']

# df = df.iloc[:, :-2]

# custom_palette = ["orange", "skyblue"]
# sns.set_palette(custom_palette)

# # Create a grid with 1 row and 2 columns
# fig, axs = plt.subplots(1, 2, figsize=(12, 8), 
#                         gridspec_kw={'width_ratios': [0.8, 0.2]})

# # Plot the heatmap on the left-hand side
# sns.heatmap(df, cmap=custom_palette, cbar=False, 
#             annot=False, 
#             ax=axs[0])

# # Add ylabel, xlabel, and title to the heatmap
# axs[0].set_ylabel('Test Samples', fontsize=14, fontweight='bold')
# axs[0].set_xlabel('Predictions', fontsize=14, fontweight='bold')
# axs[0].set_title('Predictions of Test Samples for 1000 models', fontsize=16, 
#                   fontweight='bold')
# axs[0].tick_params(axis='both', which='both', length=0)  # Remove tick marks
# axs[0].set_xticks([])  # Hide y-axis ticks
# axs[0].tick_params(axis='y', labelsize=14)

# # Plot the percentages on the right-hand side
# bar_plot_1 = axs[1].barh(p1.index, p1, 
#                           color='SkyBlue', 
#                           label='Percentage of 1s')
# bar_plot_0 = axs[1].barh(df.index, p2,
#                           color='Orange',
#                           left=p1, label='Percentage of 0s')

# # Customize the percentages plot
# axs[1].set_xlabel("")
# axs[1].set_title('Percentages', fontsize=16, fontweight='bold')
# axs[1].set_yticks([])  # Hide y-axis ticks
# axs[1].tick_params(axis='x', labelsize=14)

# # Move the legend outside the plot area to the right and center vertically
# axs[1].legend(handles=[bar_plot_1[0], bar_plot_0[0]], 
#               labels=['Predicted Mixture', 'Predicted No Mixture'],
#               bbox_to_anchor=(1.05, 0.5), loc='center left',
#               fontsize=14)

# # Remove white space from top and bottom of the second plot
# plt.margins(y=0)
# axs[1].invert_yaxis()
# plt.tight_layout()  # Improve spacing between plot elements

# savename = os.path.join(DIR, "Predictions.png")
# plt.savefig(savename, dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()
###############################################################################
#Plot test results
# DIR=STAT_DIR

# stat_file = os.path.join(DIR, 'test_stats.csv')
# df = pd.read_csv(stat_file, index_col=0)

# df_mean = df.mean()
# df_std = df.std(axis=0, ddof=1)
# df_mean.index = ['Accuracy', 'F1', 'Precision', 'Recall', "AUC"]

# plt.bar(df_mean.index, df_mean, yerr=df_std, capsize=10, alpha=1)
# plt.tick_params(axis='both', which='both', length=0)

# plt.tick_params(axis='both', labelsize=14)
# plt.xlabel('Stats', size=14, fontweight="bold")
# plt.ylabel('Score', size=14, fontweight="bold")
# plt.title('Mean Test Scores from 100 models', size=16, 
#           fontweight="bold")

# # Show the plot
# savename = os.path.join(DIR, "test_stats.png")
# plt.savefig(savename, dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()
###############################################################################
#Plot heatplot
# df, names = structure.inData(dat = os.path.join(DATA_DIR,'data_csbi.csv'),
#                               lim =12)
# #Encoding 'status' as label 1 & 0 , naming the field as target
# df['target'] = df['Mix']
# structure.reClass(df=df, col='target')
# df.drop('Mix',axis = 1, inplace=True)
# mix = df['target']
# df.drop('target',axis = 1, inplace=True)


# underline_indices_1 = [0, 1]
# underline_indices_2 = [2, 3]

# #convert by percentage
# df_percentage = df.div(df.sum(axis=1), axis=0) * 100
# df_percentage = df_percentage.transpose()
#heatplot
# plt.figure(figsize=(12, 8))
# cm = sns.color_palette("light:b", as_cmap=True)
# heatmap  = sns.heatmap(df_percentage, cmap=cm, cbar=True, 
#             annot=False)
# plt.axvline(x=42, color='red', linestyle='--')
# plt.tick_params(axis='y', size = 14, )
# plt.xlabel('Samples', size=14, fontweight="bold")
# plt.ylabel('Sequences at M:215-245 (%)', size=14, fontweight="bold")
# plt.title('Spread of sequences per sample', size=16, 
#           fontweight="bold")
# plt.xticks([])

# # Add title to the colorbar
# cbar = heatmap.collections[0].colorbar  # Get the colorbar object
# cbar.ax.set_title('%', fontsize=14, fontweight="bold")  # Set the colorbar title and font size

# savename = os.path.join(CSBI_DIR, "heatmap_percentage.png")
# plt.savefig(savename, dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()


