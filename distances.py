# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 17:08:30 2018

@author: ASUS
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

def euclidean_distances_scaled(B):
    if len(B.shape)==1: B=B.reshape(-1,1)
    
    for i in range(B.shape[1]):
        B[:,i] /= np.max(B[:,i])
        
    return euclidean_distances(B)
def manhattan_distances_scaled(B):
    if len(B.shape)==1: B=B.reshape(-1,1)
    
    for i in range(B.shape[1]):
        B[:,i] /= np.max(B[:,i])
    
    return manhattan_distances(B)

def ln_distances(B):
    if len(B.shape)==1: B=B.reshape(-1,1)
    
    Ln_dist=np.zeros((B.shape[0],B.shape[0]))
    for i in range(B.shape[1]):
        Ln_dist += np.log2(manhattan_distances(B[:,i].reshape((-1,1)))+1)
    
    return Ln_dist

def ln_distances_scaled(B):
    if len(B.shape)==1: B=B.reshape(-1,1)
        
    for i in range(B.shape[1]):
        B[:,i] /= np.max(B[:,i])
        
    Ln_dist=np.zeros((B.shape[0],B.shape[0]))
    for i in range(B.shape[1]):
        Ln_dist += np.log2(manhattan_distances(B[:,i].reshape((-1,1)))+1)
    
    return Ln_dist

def E_dist(APcounts, APSet):
    Array=APcounts.loc[APSet][APcounts.columns[:len(APcounts.columns)-1]].values
    Array=np.array(Array).T
    
    return euclidean_distances(Array)

def M_dist(APcounts, APSet):
    Array=APcounts.loc[APSet][APcounts.columns[:len(APcounts.columns)-1]].values
    Array=np.array(Array).T
    
    return manhattan_distances(Array)

def ES_dist(APcounts, APSet):
    Array=APcounts.loc[APSet][APcounts.columns[:len(APcounts.columns)-1]].values
    Array=np.array(Array).T
    
    return euclidean_distances_scaled(Array)

def MS_dist(APcounts, APSet):
    Array=APcounts.loc[APSet][APcounts.columns[:len(APcounts.columns)-1]].values
    Array=np.array(Array).T
    
    return manhattan_distances_scaled(Array)

def Ln_dist(APcounts, APSet):
    Array=APcounts.loc[APSet][APcounts.columns[:len(APcounts.columns)-1]].values
    Array=np.array(Array).T
    
    return ln_distances(Array)

def LnS_dist(APcounts, APSet):
    Array=APcounts.loc[APSet][APcounts.columns[:len(APcounts.columns)-1]].values
    Array=np.array(Array).T
    
    return ln_distances_scaled(Array)
    

def X_dist(APcounts, APSet, method='E'):

    if method == 'E':
        return E_dist(APcounts, APSet)
    elif method == 'ES':
        return ES_dist(APcounts, APSet)
    elif method == 'M':
        return M_dist(APcounts, APSet)
    elif method == 'MS':
        return MS_dist(APcounts, APSet)
    elif method == 'Ln':
        return Ln_dist(APcounts, APSet)
    elif method == 'LnS':
        return LnS_dist(APcounts, APSet)
    else:
        print('error')
        return None