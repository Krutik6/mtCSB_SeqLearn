# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 09:25:39 2023

@author: nkp68
"""

def printNan(df):
    df_null = df.isnull().sum()
    for i in df_null:
        if i>0:
            print('NAN values present')
        if i ==0:
            print('No NAN values present')