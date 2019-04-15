#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:04:55 2019

@author: abraxas
"""

# Data manipulation
import pandas as pd
import numpy as np
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


def show_corr(df,th,method='pearson'):
    "Compute pairwise correlation of columns (excluding NA/null values) of a dataframe (df) using a method, and provide only those that are higher than a threshold (th)"
    corre=np.triu(df.corr(method=method).values,1) # takes upper triangle of correlation matrix, ignoring the diagonal
    ixth=np.argwhere(corre>=th)
    larray=[]
    for i in ixth: 
        larray.append([df.columns[i[0]],df.columns[i[1]],corre[tuple(i)]])
    return larray

def plot_value_counts(df, col):
    "Plot value counts of a column (col) from a dataframe (df)"
    plt.figure(figsize = (8, 6))
    df[col].value_counts().sort_index().plot.bar(color = 'blue', edgecolor = 'k', linewidth = 2)
    plt.xlabel(f'{col}'); plt.title(f'{col} Value Counts'); plt.ylabel('Count')
    plt.show();
    
def group_categorical_features(df,cols,conorder=True):
    "Create a new categorical feature (newcol) grouping other categorical features (cols) from a dataframe(df)"
    "If conorder==False order of columns is not important (e.g, [1,0]=[0,1])"
    newcol=df[cols].values.tolist()
    if conorder==False: newcol=np.sort(newcol).tolist()   
    return newcol

def count_categorical(df,cols,colsval=[],prefix_val='value_',prefix_n=False):
    "Create a dataframe with features that count the number of times that categorical values appear in the columns (cols) of a dataframe (df)"
    "The output dataframe has contain the value of each category if they are introduced in colsval (list qith the same length of cols) "
    "Names of columns contain the categories and optionally a prefix (see prefix_n and prefix_val optional inputs)"
    coladd=pd.concat([df[icol] for icol in cols])
    coladd_dum=pd.get_dummies(coladd)
    if colsval!=[]:
        colvaladd=pd.concat([df[icol] for icol in colsval])
        colvaladd_dum=coladd_dum.mul(colvaladd,axis='rows')
        if prefix_val: colvaladd_dum.columns = prefix_val + colvaladd_dum.columns.astype(str)
    if prefix_n: coladd_dum.columns = prefix_n + coladd_dum.columns.astype(str)
    if colsval!=[]: coladd_dum=pd.concat([coladd_dum,colvaladd_dum],axis='columns')
    dfcount=coladd_dum.groupby(level=0).sum() 
    return dfcount