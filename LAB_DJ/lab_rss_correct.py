# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 10:07:43 2018

@author: ASUS
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis


def plot_1drss(X):
    fig,ax = plt.subplots()
    
    plt.plot(X, marker='o')
    for i in range(X.shape[0]):
        ax.annotate(str(i), tuple([i,X[i]]))
    
    return None
    
    
    
    
def plot_2drss(X):
    '''
    input: X: X[n_physical_node, n_AP]
    '''
    skew_X2 = skew(X.T[1])
    kurt_X2 = kurtosis(X.T[1])
    fig,ax = plt.subplots()
    
    ax.scatter(X[:,0], X[:,1])
    plt.axis('equal')
    for i in range(X.shape[0]):
        ax.annotate(str(i), tuple(X[i,:]))
    print('skew: {:.2f}\nkurt: {:.2f}'.format(skew_X2, kurt_X2))
        
        
    return None

def plot_2ddis(X):
    X_mat = X.reshape((4,3))
    
    sns.heatmap(X_mat, cmap='Greys')
    