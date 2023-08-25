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
MLdata_DIR = os.path.join(WORKING_DIRECTORY, "mldata")

#add names to columns
ids=os.listdir(FILE_DIR)
ids=[ x for x in ids if "CTC" in x ]

# make object which comprises of all files from a sequence
# files = glob(os.path.join(FILE_DIR, "*txt"))

# txt = []  
# #list files
# for i in files:
#     file=pd.read_table(i, sep='\t', header=None, on_bad_lines='skip')
#     txt.append(file)
    
# # object for loops    
# array_length = len(txt)  

# #only keep chromosome, position, quality, sequence and ID
# tokeep = [2, 3, 4, 9]
# newList = [[l[i] for i in tokeep] for l in txt]

# del tokeep, files, txt, file


# named_list=[]
# for i in range(array_length):
#     x = newList[i]
#     file = pd.concat([x[0],x[1], x[2], x[3]], axis=1)
#     file.columns = ['Chromosome', 'Position', 'Quality', 'Sequence']
#     file['Sample'] = ids[i]
#     named_list.append(file)
#     del x
#     del file
    
# #plot to show NUMPS
# df = pd.concat(named_list)
# listch = ['chrM', 'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
#           'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
#           'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']
# ch = df[df['Chromosome'].isin(listch)]
# chnot = df[~df['Chromosome'].isin(listch)]
# ch = ch['Chromosome']
# ch_count = ch.value_counts()
# ch_count = ch_count.append(pd.Series([529]))
# ch_count = ch_count.rename(index={0: 'Other'})
# ch_count = ch_count.drop('chrM')
# ch_count = ch_count.drop('Other')
# column_sum = ch_count.sum()

# plt.figure(figsize=(20, 20), dpi=300)
# ch_count.plot.pie(autopct='%1.1f%%', 
#                   pctdistance=0.9, textprops={'fontsize':16},
#                   labeldistance=1.2)
# plt.title('Non-mitochondrial reads aligned to the Mitochondrial Reference Genome',
#           fontsize=30)
# plt.axis('off')
# plt.show()

#only keep samples which are between positions 100 and 300 of the mt genome
#use T/F to do this
# honed=[]
# for i in range(array_length):
#     x = named_list[i]
#     file = x.loc[x['Chromosome'] == "chrM"]
#     file = file[file['Quality'] > 59]
#     file = file[file['Position'].between(100, 300)]
#     honed.append(file)
#     del x
#     del file
    
#plot positions
# df = pd.concat(honed)
# p = df['Position']
# p_count = p.value_counts()
# pos = p_count.index
# x = np.arange(len(pos))

# ps = p_count.sort_index()

# plt.figure(figsize=(8, 6), dpi=300)
# plt.bar(x=ps.index, height=ps.values)
# plt.yscale('log')
# plt.xlabel('Read Start Sites')
# plt.ylabel('Number of reads (log)')
# plt.title('Number of reads which begin between M:100-300')

# row_sum = sum(len(df) for df in honed)
# print(row_sum)
#make showing spread of quality samples for each samples
# df = pd.concat(honed)
# df = df.iloc[:, [2, 4]].reset_index()
# df = df.iloc[:, [1, 2]]
# counts = df.groupby('Sample')['Quality'].value_counts()

# del named_list, newList, txt, tokeep

#remove IDs: 14, 28, 57, 58, 84
rid=[14, 28, 57, 58, 84]

# for i in rid:
#     print(f'Remove {ids[i]}')
    
ids = [ids[i] for i in range(len(ids)) if i not in rid]
array_length = len(ids)
# honed = [honed[i] for i in range(len(honed)) if i not in rid]

###############################################################################
#REMOVES TOO MUCH
#Add new T/F columns if CxTCx is found in Sequence column
# vnames = ["C5TC7", "C5TC8", "C6TC7", "C6TC8", "C6TC9"]    
# vseqs = ["CCCCCTCCCCCCC", "CCCCCTCCCCCCCC", "CCCCCCTCCCCCCC",
#           "CCCCCCTCCCCCCCC", "CCCCCCTCCCCCCCCC"]

# boolann = []
# for i in range(array_length):
#     x = honed[i]
#     for j in range(5):
#         x[vnames[j]] = np.where(x['Sequence'].str.contains(vseqs[j]), 
#                                 'TRUE', 'FALSE')
#     boolann.append(x)
    

# # del x,i,j

# #'#Remove if all vnames are false
# boolann_true = []
# for i in range(array_length):
#     x = boolann[i]
#     x.drop(x[x['C5TC7'] == "FALSE"].index, inplace=True)
#     boolann_true.append(x)
    
#del i,boolann,honed,x

#compare bias to all
# cc = pd.concat(boolann_true)
# print(len(cc)/len(df) * 100)
# un = cc['Sample'].unique()

# #Count number of times each variant is found per read
# vnames_count = ["G5AG7_count", "G5AG8_count", "G6AG7_count", "G6AG8_count", "G6AG9_count"]

# boolann_count =[]
# for i in range(array_length):
#       x = boolann_true[i]
#       for j in range(5):
#           x[vnames_count[j]]=x.Sequence.str.count(vseqs[j])
#       boolann_count.append(x)
     

# del x,i,j,boolann_true
###############################################################################
# #save object to save time
def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True

# sv = os.path.join(TMP, "Qual.pkl")
# with open(sv, 'wb') as f:
#     pickle.dump(honed, f)
    
#load new file    
# with open(sv, 'rb') as f:
#     qual_count = pickle.load(f)
    
#create array object for loops

# array_length = len(qual_count) 

# del f,sv

# data_end = []
# for i in range(1):
#     x = qual_count[i]
#     x['Posend'] = x.Position + 250
#     x = x.reset_index()
#     x = x.drop('index', axis=1)
    
#     y = pd.DataFrame(x.Sequence.apply(list).tolist())
#     z = pd.concat([x,y], axis=1)
    
#     list_seq = []
    
#     for index, row in z.iterrows():
#         new_df = pd.DataFrame(row).transpose()
#         list_seq.append(new_df)
    
#     data_end.append(list_seq)

# data_end_named = []
# for lst in range(1):
#     x = data_end[lst]
#     X_list = []
#     for i in range(len(x)):
#         y = x[i]
#         y['Position'] = pd.to_numeric(y['Position'])
#         y['Posend'] = pd.to_numeric(y['Posend'])
#         p1 = y['Position'][i]
#         p2 = y['Posend'][i] + 1
#         le = list(range(p1, p2))
#         le = list(['Chromosome', 'Position', 'Quality', 'Sequence', 'Sample', 'Posend'] + le)
#         y.columns = le
        
#         X_list.append(y)
        
#     data_end_named.append(X_list)
    
#attach names
# data_name_deqpos = dict(zip(ids, data_end))

#save
# sv = os.path.join(TMP, "data_seq_qual_name.pkl")
# # with open(sv, 'wb') as f:
# #     pickle.dump(data_name_deqpos, f)
# # #load
# with open(sv, 'rb') as f:
#     data_seq = pickle.load(f)
    
# array_length = len(data_seq) 
    
# # only keep sequences from ranges 303-315 (CSBII region)
# #first we will keep between 300-318 and narrow down
# keep = ['Chromosome', 'Position', 'Quality', 'Sequence', 'Sample', 'Posend',
#         300,301,302,
#         303, 304, 305, 306, 307, 308, 309,
#         310, 311, 312, 313, 314, 315, 316,
#         317, 318, 319, 320, 
#         321, 322, 323]

# CSBII_data_list = []
# for i in range(array_length):
#     X = data_seq[ids[i]]
#     CSBII_data = [df[keep] for df in X]
#     df = pd.concat(CSBII_data, ignore_index=True)
#     CSBII_data_list.append(df)

# # #attach names
# data_name_deqpos = dict(zip(ids, CSBII_data_list))

sv = os.path.join(TMP, "data_csbii_raw.pkl")
# with open(sv, 'wb') as f:
#     pickle.dump(data_name_deqpos, f)
    
# load new file    
with open(sv, 'rb') as f:
    data_csbII = pickle.load(f)

#count the number of times each nt occurs in each column
data_csbII_seq = []
for i in range(array_length):
    df = data_csbII[ids[i]]
    df = df.iloc[:, 6:]
    data_csbII_seq.append(df)
    
df = pd.concat(data_csbII_seq)

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

# Add vertical lines between 300 and 301, and between 304 and 305
ax.axvline(x=302.5, color='black')
ax.axvline(x=319.5, color='black')

# Set labels and title
ax.set_xlabel('Nucleotide Position')
ax.set_ylabel('Number of Occurances')
ax.set_title('Nucleotides seen around the CSBII region')

# Set legend
ax.legend(title='', loc='upper center',
          bbox_to_anchor=(0.5, -0.15), ncol=len(df.index))

save=os.path.join(dotplot_DIR, 'csbii_scatter.png')
plt.savefig(save, dpi=600)
plt.show()

#remove columns 300-302 + 320-323
# data_csbII = list(data_csbII.values())
# data_csbII_edit = []
# for i in range(array_length):
#     df = data_csbII[i]
#     df = df.drop(columns=[300, 301, 302, 320, 321, 322, 323])
#     data_csbII_edit.append(df)

# #join sequences in region of interest into single column
# #only keep samples and sequence of interest
# csbii_named = []
# for i in range(array_length):
#     df = data_csbII_edit[i]
#     df['CSBII'] = df[df.columns[6:]].apply(
#         lambda x: ''.join(x.dropna().astype(str)),
#         axis=1
#         )
#     df = df.iloc[:, [4,23]]
#     csbii_named.append(df)
    
# combined_df = pd.concat(csbii_named)

# #count number of times each sequence was counted
# value_counts = combined_df['CSBII'].value_counts().reset_index()
# count = len(value_counts.loc[value_counts['CSBII'] < 2])
# #remove any sequences seen fewer than 100 times
# value_counts100 = value_counts[value_counts['CSBII'] > 100] 
# # print(value_counts100['CSBII'].sum())
# # print(547623/614908 * 100) 
# #300-318 206 sequences
# #303-318 174 sequences
# #303-316 143 sequences 91.28585089151548%
# #303-319 189 sequences 89.05771269848498%

# #alter to matrix
# tokeep = value_counts100['index'].tolist()
# combined_df = combined_df[combined_df['CSBII'].isin(tokeep)]
# combined_stretched = combined_df.groupby(['Sample', 'CSBII']).size().reset_index(name='Count')
# df_wide = combined_stretched.pivot(index='Sample', columns='CSBII', values='Count')
# df_wide.index = df_wide.index.str.replace('.CTC.txt', '')

# #NORMALISE DATA BY RPM
# length = os.path.join(RAW_DATA, 'egg_tcounts.txt')
# lfile = pd.read_table(length, header=None)
# lfile[['Sample', 'TotalReads']] = lfile[0].apply(lambda x: pd.Series(str(x).split(": ")))
# lfile['Sample'] = lfile['Sample'].str.replace('_dedup', '')
# lfile['TotalReads'] = lfile['TotalReads'].astype(int)
# lfile['Divisor'] = lfile['TotalReads']/100000
# lfile.index = lfile['Sample']

# #Keep indexes the same and remove 0s
# lfile = lfile.loc[df_wide.index]
# df_wide = df_wide.fillna(0)

# #divide each number of reads by divisor via RPM*10
# df_wide = df_wide.div(lfile['Divisor'], axis='index')

# #remove columns which do not have a norm number of reads above 10
# df_wide = df_wide.loc[:, (df_wide > 50).any()]
# # print(df_wide.values.sum())
# #50 - 17 sequences

# #remove columns which have 3As 
# #Remove columns if the column name contains 'AAA' - as we suspect these may be due to a small deletion upsteam of the CSBII region
# df_filtered = df_wide[[col for col in df_wide.columns if 'AAA' not in col]]
# df_filtered = df_filtered[[col for col in df_filtered.columns if 'CCCCCCCCCCCC' not in col]]
# #AAA + C12 14 sequences

# #assess what percentage of usable sequences remained
# df_wide = df_filtered.mul(lfile['Divisor'], axis='index')
# # print(df_wide.values.sum())
# # print(471870/614908 * 100) 
# #76.73% 14 sequences

# sv = os.path.join(TMP, "data_csbii.pkl")
# with open(sv, 'wb') as f:
#     pickle.dump(df_filtered, f)
# #Save the DataFrame to CSV
# csv_file =  os.path.join(TMP, "data_csbii.csv")
# df_filtered.to_csv(csv_file, index=True)

#create plots
# load new file    
# with open(sv, 'rb') as f:
#     data_csbII = pickle.load(f)
    
#create tile plot
# df = data_csbII.transpose()
# df_norm = df.div(df.sum()) * 100
# df_norm = df_norm.round(2)

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
# Calculate the distance matrix
# distance_matrix = sch.distance.pdist(data_csbII)

# # Perform hierarchical clustering
# linkage_matrix = sch.linkage(distance_matrix, method='single')

# # Create the dendrogram with indexes as branches
# dendrogram = sch.dendrogram(linkage_matrix, 
#                             labels=data_csbII.index,
#                             orientation='right')

# fig = plt.gcf()
# fig.set_size_inches(6, 18)
# plt.tick_params(axis='both', which='major', labelsize=12)

# plt.xlabel('Distance', fontsize=14)
# plt.ylabel('Samples', fontsize=14)
# plt.title('Distance between Samples', fontsize=16)
# plt.grid(False)

# save=os.path.join(barplot_DIR, 'dendogram.png')
# plt.savefig(save, dpi=600)
# plt.show()

