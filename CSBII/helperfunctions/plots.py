# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:07:14 2023

@author: nkp68
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt

def plotlosses(epochs, train_losses, val_losses, outDir, plotname='lossplot.png'):
    plt.plot(range(epochs), train_losses, 'orange', label= 'Training loss')
    plt.plot(range(epochs), val_losses, 'blue', label= 'Validation loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    savename = os.path.join(outDir, plotname)
    plt.savefig(savename) 
    plt.close()


def plotClasses(df, col, outdir):
    sns.countplot(x = col, data=df)
    savename = 'Samples_count.png'
    plt.savefig(os.path.join(outdir, savename),
                bbox_inches='tight', dip=300)
    plt.close()
    

def pairPlot(df, col, outdir, plotname='pairplot.png'):
    pair_plot = sns.pairplot(df, hue=col)
    fig = pair_plot.fig
    pairplotname = os.path.join(outdir, plotname)
    fig.savefig(pairplotname) 
    plt.close()
    
    
    
def plotTrain(epochs, listofloss, outDir, name):
    plt.plot(range(epochs), listofloss)
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    savename = os.path.join(outDir, name)
    plt.savefig(savename)
    plt.close()
    

from sklearn.metrics import confusion_matrix
    
def CM(y_test, y_pred, outDir, name, CMmax, CMmin):
    cm=confusion_matrix(y_test, y_pred)  
    plt.figure(figsize=(10,6))
    sns.heatmap(cm, annot=True, cmap="crest", vmax=CMmax, vmin=CMmin)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    savename = os.path.join(outDir, name)
    plt.savefig(savename)
    plt.close()
    

import numpy as np
    
def correlationPlot(df, OutDir, name, num_cols):
    df_data = df.iloc[:, :-1]
    corr = df_data[num_cols].corr()
    
    plt.figure(figsize=(12,12),dpi=80)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='magma', robust=True, center=0,
                square=True, linewidths=.5)
    plt.title('Correlation of Numerical(Continous) Features', fontsize=15,font="Serif")
    name=os.path.join(OutDir, name)
    plt.savefig(name)
    plt.close()
    

from sklearn import metrics
    
def plotROC(model, X_test, y_test, OutDir, name):  
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 17
    
    plotname="AUC - "+name
    name.replace(" ", "_")
    savename=name+".png"
    plt.figure(figsize=(20,20))
    metrics.plot_roc_curve(model, X_test, y_test, pos_label="Yes") 
    plt.title(plotname)
    plt.xlabel('False Positive Rate (Pos = Yes to Mixture)')
    plt.ylabel('True Positive Rate (Pos = Yes to Mixture)')
    plt.rc('font', size=MEDIUM_SIZE)         
    plt.rc('axes', titlesize=MEDIUM_SIZE)    
    plt.rc('axes', labelsize=MEDIUM_SIZE)   
    plt.rc('xtick', labelsize=SMALL_SIZE)   
    plt.rc('ytick', labelsize=SMALL_SIZE)  
    plt.rc('legend', fontsize=MEDIUM_SIZE)   
    plt.rc('figure', titlesize=BIGGER_SIZE)  
    plt.plot([0, 1], [0, 1],'r--', alpha=0.7)
    plt.savefig(os.path.join(OutDir, name), dpi=300) 
    plt.close()


def plotHPloss(df, outDir, name='HP_losses.png'):
    #begin creating image
    df['X'] = range(1, len(df) + 1)

    #shape mapping
    shape_mapping = {0.1:'o',
                      0.01:'s'
                      ,0.001:'^'}
    # Set the background color based on the group column
    background_data = [[0.1, 0.3, 0.5],
                       [0.1, 0.3, 0.5],
                       [0.1, 0.3, 0.5],
                       [0.1, 0.3, 0.5],
                       [0.1, 0.3, 0.5]] 
    #set outline colours
    color_mapping = {20:'black',
                      40:'magenta',
                      80:'yellow',
                      160:'white'}

    x_labels = df.index

    # Plotting the scatter plot with two Y-axis lines
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Scatter plot for Y1 with shapes and outline colors
    for (shape, color), shape_df in df.groupby(['learning_rate', 'epochs']):
        marker = shape_mapping.get(shape, 'o')
        outline_color = color_mapping.get(color, 'black')
        ax2.scatter(shape_df['X'], shape_df['Train_loss'], 
                    marker=marker, edgecolors=outline_color, 
                    facecolors='red',
                    label=f'Y2 - {shape} ({color})')

    #Scatter plot for Y2 with shapes and outline colors
    for (shape, color), shape_df in df.groupby(['learning_rate', 'epochs']):
       marker = shape_mapping.get(shape, 'o')
       outline_color = color_mapping.get(color, 'black')
       ax2.scatter(shape_df['X'], shape_df['Val_loss'],
                   marker=marker, edgecolors=outline_color, 
                   facecolors='blue',
                   label=f'Y2 - {shape} ({color})')

    # Plot the background using pcolor
    extent = [-0.5, len(x_labels) - 0.5, -0.5, 1.5]
    ax2.imshow(background_data, cmap='brg',
               extent=extent, aspect='auto', alpha=0.2)
    ax1.set_ylim(-0.1,1.1)
    ax2.set_ylim(-0.1,1.1)


    #Adding labels and title
    ax1.set_xlabel('Hyperparameters')
    ax1.set_ylabel('Scaled Mean Training MSE', color='red')
    ax2.set_ylabel('Scaled Mean Validation MSE', color='blue')
    plt.title('Hyperparameter testing on training and validation sets')
    plt.xticks([])

    #saving the plot
    savename=os.path.join(outDir, name)
    plt.savefig(savename, dpi=600)
    

def plotHPval(df, outDir, name='HP_val.png', low=0.5):
    df.set_index('Index', inplace=True)
    x_labels = df.index

    # Set the background color based on the group column
    background_data = [[0.1, 0.3, 0.5],
                        [0.1, 0.3, 0.5],
                        [0.1, 0.3, 0.5],
                        [0.1, 0.3, 0.5],
                        [0.1, 0.3, 0.5]] 

    # Create a bar graph for the selected columns
    # Create a bar graph for the selected columns with error bars
    ax = df['Score'].plot(kind='bar',
                        yerr=df['STDV'],
                        capsize=3,
                        figsize=(17, 8),
                        width=0.5,
                        align='center', 
                        alpha=0.7,
                        ecolor='black',
                        color=df['TYPE'],
                        edgecolor=df['epochs'],
                        linewidth=5,
                        hatch=df['learning_rate'])

    # Plot the background using pcolor
    extent = [-0.5, len(x_labels) - 0.5, -0.5, 1.5]
    ax.imshow(background_data, cmap='brg',
                extent=extent, aspect='auto', alpha=0.2)
    ax.set_ylim(low,1.05)


    # Add labels and title
    plt.xlabel('Hyper Paramters', size=20)
    plt.ylabel('Validation Scores', size=20)
    plt.yticks(fontsize=18)
    # Remove x-axis ticks
    plt.xticks([])
    #adjust white space
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    #saving the plot
    savename=os.path.join(outDir, name)
    plt.savefig(savename, dpi=600)
    plt.close()
    

def plotpie(y, outDir, name, theset):
    tset = theset
    total = len(y)
    percentage_1 = (np.count_nonzero(y == 1.0) / total) * 100
    percentage_0 = (np.count_nonzero(y == 0.0) / total) * 100

    # Plot the pie chart
    labels = ['No Mixture Anticipated', 'Mixture Anticipated']
    sizes = [percentage_0, percentage_1]
    colors = ['lightcoral', 'lightskyblue']
    explode = (0, 0.1)  # Explode the second slice (1.0) for emphasis
    plt.pie(sizes, explode=explode, labels=labels, 
            colors=colors, shadow=False, startangle=140,
            textprops={'fontsize': 12})


    # Add the percentage and number of 0.0 and 1.0 to the pie chart
    plt.text(-0.5, -0.1, f"{np.count_nonzero(y == 0.0)} ({percentage_0:.1f}%)", 
             ha='center', va='center', color='black', size=15)
    plt.text(0.4, 0.5, f"{np.count_nonzero(y == 1.0)} ({percentage_1:.1f}%)",
             ha='center', va='center', color='black', size=15)

    # Add a title
    plt.title(f"Spread of classes in {tset} set", fontsize=18)
    
    #save
    savename=os.path.join(outDir, name)
    plt.savefig(savename, dpi=600)
    plt.close()
