# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:59:10 2022

@author: nkp68

This script explores the CSBII region of the mitochondrial DNA samples.
We create a text file for use in a ML model here.
"""

import os
from glob import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
import seaborn as sns
import matplotlib.ticker as ticker
import csv
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch


#set global variables
WORKING_DIRECTORY = os.path.dirname(__file__)
RAW_DATA = os.path.join(WORKING_DIRECTORY, "raw")
FILE_DIR = os.path.join(RAW_DATA, "C5TC5")
TMP = os.path.join(WORKING_DIRECTORY, "tmp")
barplot_DIR = os.path.join(WORKING_DIRECTORY, "barplots")
dotplot_DIR = os.path.join(WORKING_DIRECTORY, "dotplots")

#start off with processed data
#add names to columns
ids=os.listdir(FILE_DIR)
ids=[ x for x in ids if "CTC" in x ]
rid=[14, 28, 57, 58, 84]
ids = [ids[i] for i in range(len(ids)) if i not in rid]

# #save object to save time
def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True

# sv = os.path.join(TMP, "data_seq_qual_name.pkl")
# #load
# with open(sv, 'rb') as f:
#     data_seq = pickle.load(f)
    
# array_length = len(data_seq) 

    
#only keep sequences from ranges 303-315 (CSBI region)
#first we will keep between 300-318 and narrow down
# keep = ['Chromosome', 'Position', 'Quality', 'Sequence', 'Sample', 'Posend',
#         212,213,214,215,216,217,218,219,220,
#         221,222,223,224,225,226,227,228,229,230,
#         231,232,233,234,235,236,237,238]

# X = data_seq[ids[1]]
# x0 = X[0]
# x0 = x0.loc[:, [col for col in x0.columns if col in keep]]

# df = pd.concat(X, ignore_index=True)
# #CSBI_data = [df[keep] for df in X]
# CSBI_data = [df.loc[:, [col for col in df.columns if col in keep]] for df in X]
# df = pd.concat(CSBI_data, ignore_index=True)

# CSBI_data_list = []
# for i in range(array_length):
#     X = data_seq[ids[i]]
#     CSBI_data = [df.loc[:, [col for col in df.columns if col in keep]] for df in X]
#     df = pd.concat(CSBI_data, ignore_index=True)
#     CSBI_data_list.append(df)

# #attach names
# data_name_deqpos = dict(zip(ids, CSBI_data_list))

sv = os.path.join(TMP, "data_csbi_raw.pkl")
# with open(sv, 'wb') as f:
#     pickle.dump(data_name_deqpos, f)
    
# load new file    
with open(sv, 'rb') as f:
    data_csbI = pickle.load(f)
array_length = len(data_csbI) 

#count the number of times each nt occurs in each column
data_csbI_seq = []
for i in range(array_length):
    df = data_csbI[ids[i]]
    df = df.iloc[:, 6:]
    data_csbI_seq.append(df)
    
df = pd.concat(data_csbI_seq)

counts_df = pd.DataFrame()
for column in df.columns:
    counts = df[column].value_counts()
    counts_df[column] = counts


#Get the row and column labels
row_labels = counts_df.index
column_labels = counts_df.columns

# Create a plot
fig, ax = plt.subplots()

for i, row_label in enumerate(counts_df.index):
    values = counts_df.loc[row_label].values
    
    if row_label == 'A':
        color = 'blue'
        marker = 'x'
    elif row_label == 'C':
        color = 'orange'
        marker = 'x'
    elif row_label == 'T':
        color = 'green'
        marker = 'x'
    elif row_label == 'G':
        color = 'pink'
        marker = 'x'
    else:
        color = 'black'
        marker = 'x'
        
    ax.scatter(y=values, x=counts_df.columns, label=row_label,
               color=color, marker=marker)

# Add vertical lines between 300 and 301, and between 319 and 320
ax.axvline(x=214.5, color='black')
ax.axvline(x=235.5, color='black')

#Set labels and title
ax.set_xlabel('Nucleotide Position')
ax.set_ylabel('Number of Occurances')
ax.set_title('Nucleotides seen around the CSBI region')

# Set legend
ax.legend(title='', loc='upper center',
          bbox_to_anchor=(0.5, -0.15), ncol=len(df.index))

save=os.path.join(dotplot_DIR, 'csbi_scatter.png')
plt.savefig(save, dpi=600)
plt.show()

#remove columns 300-302 + 320-323
data_csbI = list(data_csbI.values())
data_csbI_edit = []
for i in range(array_length):
    df = data_csbI[i]
    df = df.drop(columns=[212, 213, 214, 236, 237, 238])
    data_csbI_edit.append(df)

#join sequences in region of interest into single column
#only keep samples and sequence of interest
csbi_named = []
for i in range(array_length):
    df = data_csbI_edit[i]
    df['CSBI'] = df[df.columns[6:]].apply(
        lambda x: ''.join(x.dropna().astype(str)),
        axis=1
        )
    df = df.iloc[:, [4,27]]
    csbi_named.append(df)
    
combined_df = pd.concat(csbi_named)


#count number of times each sequence was counted
value_counts = combined_df['CSBI'].value_counts().reset_index()
count = len(value_counts.loc[value_counts['CSBI'] < 2])
#remove any sequences seen fewer than 100 times
value_counts100 = value_counts[value_counts['CSBI'] > 100] 
print(value_counts100['CSBI'].sum())
print(561440/614908 * 100) 
#215-235 108 sequences 91.30471550215643%

#alter to matrix
tokeep = value_counts100['index'].tolist()
tokeep = [seq for seq in tokeep if len(seq) >= 20]
combined_df = combined_df[combined_df['CSBI'].isin(tokeep)]
combined_stretched = combined_df.groupby(['Sample', 'CSBI']).size().reset_index(name='Count')
df_wide = combined_stretched.pivot(index='Sample', columns='CSBI', values='Count')
df_wide.index = df_wide.index.str.replace('.CTC.txt', '')

#NORMALISE DATA BY RPM
length = os.path.join(RAW_DATA, 'egg_tcounts.txt')
lfile = pd.read_table(length, header=None)
lfile[['Sample', 'TotalReads']] = lfile[0].apply(lambda x: pd.Series(str(x).split(": ")))
lfile['Sample'] = lfile['Sample'].str.replace('_dedup', '')
lfile['TotalReads'] = lfile['TotalReads'].astype(int)
lfile['Divisor'] = lfile['TotalReads']/100000
lfile.index = lfile['Sample']

#Keep indexes the same and remove 0s
lfile = lfile.loc[df_wide.index]
df_wide = df_wide.fillna(0)

#divide each number of reads by divisor via RPM*10
df_wide = df_wide.div(lfile['Divisor'], axis='index')

#remove columns which do not have a norm number of reads above 10
df_wide = df_wide.loc[:, (df_wide > 25).any()]
print(df_wide.values.sum())
#50 - 5 sequences
#25 - 10 sequences

# #remove columns which have 3As 
# #Remove columns if the column name contains 'AAA' - as we suspect these may be due to a small deletion upsteam of the CSBII region
# df_filtered = df_wide[[col for col in df_wide.columns if 'AAA' not in col]]
# df_filtered = df_filtered[[col for col in df_filtered.columns if 'CCCCCCCCCCCC' not in col]]
# #AAA + C12 14 sequences

#assess what percentage of usable sequences remained
df_filtered = df_wide.mul(lfile['Divisor'], axis='index')
print(df_filtered.values.sum())
print(253579/614908 * 100) 
#40.66380661822582% 5 sequences -f kept to 50
#41.23852673895932 10 sequences -f kept to 25 - stick to 25

sv = os.path.join(TMP, "data_csbi.pkl")
with open(sv, 'wb') as f:
    pickle.dump(df_filtered, f)
#Save the DataFrame to CSV
csv_file =  os.path.join(TMP, "data_csbi.csv")
df_wide.to_csv(csv_file, index=True)

#create plots
# load new file    
with open(sv, 'rb') as f:
    data_csbI = pickle.load(f)
    
#create tile plot
df = data_csbI.transpose()
df_norm = df.div(df.sum()) * 100
df_norm = df_norm.round(2)

# plt.figure(figsize=(8, 6), dpi=300)
# ax = sns.heatmap(df_norm, annot=False, cmap='Blues')
# ax.set_xticklabels([])
# ax.set_xticks([])
# plt.xlabel('Samples')
# plt.ylabel('Sequences')
# plt.title('Normalised abundance of sequences per samples (%)')

#plot number of reads per sample
# r1 = 43
# r2 = 44
# dep = ['Mixture'] * r1 + ['No Mixture'] * r2
# colors = ['orange' if val == 'Mixture' else 'blue' for val in dep]
# x = np.arange(len(dep))

# row_sums = data_csbII.sum(axis=1)

# plt.figure(figsize=(8, 6), dpi=300)
# plt.bar(data_csbII.index, row_sums, 
#         color=colors, width=1)
# plt.legend(['Mixture', 'No Mixture'])
# plt.xlabel('Samples')
# plt.ylabel('Number of Reads')
# plt.title('Total number of reads per sample')
# plt.setp(plt.gca(), xticks=[], xticklabels=[])
# plt.show()

#create dendogram
#Calculate the distance matrix
distance_matrix = sch.distance.pdist(data_csbI)

# Perform hierarchical clustering
linkage_matrix = sch.linkage(distance_matrix, method='single')

# Create the dendrogram with indexes as branches
dendrogram = sch.dendrogram(linkage_matrix, 
                            labels=data_csbI.index,
                            orientation='right')

fig = plt.gcf()
fig.set_size_inches(6, 18)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.xlabel('Distance', fontsize=14)
plt.ylabel('Samples', fontsize=14)
plt.title('Distance between Samples', fontsize=16)
plt.grid(False)

save=os.path.join(barplot_DIR, 'dendogram_CSBI.png')
plt.savefig(save, dpi=600)
plt.show()

