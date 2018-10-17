# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:31:28 2018

@author: HuQiang
"""

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np

import os

from sklearn.cluster import SpectralClustering, KMeans, AffinityPropagation, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.decomposition import PCA

###Parameters
colors_=['#949483','#f47b7b','#9f1f5c','#ef9020','#00af3e','#85b7e2','#29245c',
 '#ffd616','#e5352b','#e990ab','#0081b4','#96cbb3','#91be3e','#39a6dd', 
 '#eb0973','#dde2e0','#333c41']


###Data Loading
def getFilePath(path):
    
    filenameList = os.listdir(path)
    filenameList = [path + '/'+filename for filename in filenameList]
    return filenameList
'''
def getRecord(filepath):

    names = ['time','RPid','ordrep','x','y','none1','BSSID','SSID','RSSI','timestamp','band']
    drop_list = ['RPid','none1','SSID','timestamp','band']
    recordsInSingleRP = pd.read_table(filepath, names=names).drop(drop_list,axis=1)
    recordsInSingleRP = recordsInSingleRP.drop_duplicates()
    
    return recordsInSingleRP

def getRecords(filepaths):

    records = getRecord(filepaths[0])
    for i in range(len(filepaths)-1):
        records = records.append(getRecord(filepaths[i+1]), ignore_index=True)
    
    return records
'''

def getRecords(folderPath):
    
    rpList = os.listdir(folderPath)
    #rpList = ['Rss(120.0,1010.0).txt','Rss(120.0,20.0).txt']
    rdsList = []
    numrp = len(rpList)
    count = 0
    
    print('Loading...({0} items)'.format(numrp))
    for rp in rpList:
        count = count + 1
        names = ['time','RPid','ordrep','x','y','none1','BSSID','SSID','RSSI','timestamp','band']
        drop_list = ['RPid','ordrep','none1','SSID','timestamp','band']
        rs = pd.read_table(folderPath+'\\'+rp, names=names).drop(drop_list,axis=1)
        #
        #
        #
        rdsList.extend([rs])
        print(str(count), end=',')

    print('\nDone!')
    
    print('Concatenation...')
    rds_ori = pd.concat(rdsList,axis=0)
    rds_ori['coord'] = rds_ori.apply(lambda df: (df['x'],df['y']),axis=1)
    rds_ori.drop(columns=['x','y'],inplace=True)
    rds = rds_ori.pivot_table(values='RSSI',index=['time','coord'],columns='BSSID')
    rds['coord'] = rds.index.get_level_values(1)
    rds.index = rds.index.droplevel(1)
    print('Done!')
    return rds

###Data selection
def sel_ap1(rds):
    rds_fpt = rds[rds.columns[:-1]].copy()
    crd = rds['coord']
    ser = rds_fpt.count(axis=0)
    res_col = ser[ser>0.9*4761].index
    rdsfpt_ap1 = rds_fpt[res_col]
    rdsfpt_ap1.fillna(rdsfpt_ap1.min(axis=0),axis=0,inplace=True)
    fpt_ap1 = rdsfpt_ap1.values
    return fpt_ap1,crd

###Visualization
def plot_1dscatter(vec):
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(vec, np.ones(len(vec)), c='black', alpha=0.3)
    
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_title('number of hearable samples')
    
def plot_result(y,crd):
    df = DataFrame({'coord':crd,'label':y})
    num_label = len(set(df['label'].values))
    '''
    colors = ['#DE5B7B','#1B3764','#0E0220','#2E99B0',
              '#7A4579','#A2792F','#FF2E4C','#4E9525',
              '#35D0BA','#7971EA','#00818A','#FF5E3A',
              '#00D0BA','#7900EA','#FF818A','#FF5E00']
    '''
              
    fig, ax = plt.subplots()
    for i in range(num_label):
        coordList = df[df['label']==i]['coord'].tolist()
        coordList = [list(x) for x in coordList]
        coordArray = np.array(coordList)
        ax.scatter(coordArray[:,0], coordArray[:,1], c=colors_[i])
        ax.text(coordArray[:,0].mean(), coordArray[:,1].mean(), str(i))
    plt.axis('equal')
    
    return
    
###Clustering
def spe(X):
    spectral = SpectralClustering(n_clusters=12)
    return spectral.fit_predict(X)

def kme(X, n_clusters, No_):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, max_iter=1000)
    #labels:kmeans.fit_predict(X)
    #cluster_centers:kmeans.cluster_centers_
    labels = kmeans.fit_predict(X)
    cluster_list = []
    for i in np.unique(labels).tolist():
        cluster_list.append(No_[np.where(labels == i)[0]])
        
    return cluster_list

def aff(X):
    affinity = AffinityPropagation(preference=-113000)
    return affinity.fit_predict(X)

def agg(X):
    agglomerative = AgglomerativeClustering(n_clusters=3, linkage='complete')
    return agglomerative.fit_predict(X)

def dbs(X):
    dbscan = DBSCAN(eps=42)
    return dbscan.fit_predict(X)

def mea(X):
    meanshift = MeanShift()
    return meanshift.fit_predict(X)


###Dimensionality reduction
def mds(X):
    m = MDS()
    return m.fit_transform(X)

def pca(X):
    p = PCA(n_components=2)
    return p.fit_transform(X)

def tsne(X):
    t = TSNE()
    return t.fit_transform(X)

def isomap(X):
    i = Isomap()
    return i.fit_transform(X)


###Actual clustering process(split)
def _split(X, cluster_list, n_clusters, No_):
    
    cl = []
    for subX_index in cluster_list:
        _cl = kme(X[subX_index,:], n_clusters, No_[subX_index])
        cl.extend(_cl)
    
    return cl

def split(X, orders):
    
    cl = [np.arange(X.shape[0])]
    No_ = np.arange(X.shape[0])
    labels = np.ones(X.shape[0],dtype='int32')*-1
    order_list = [int(num) for num in list(orders)]
    for order in order_list:
        cl = _split(X, cl, order, No_)
        
    for i,index in enumerate(cl):
        labels[index] = i
        
    return cl



###generate cluster fingerprint
def getcf(X, cl):
    
    cf = []
    for sub_index in cl:
        cf.append(X[sub_index,:].mean(axis=0)) #mean by row
    
    return np.array(cf)


def plot_mds(m,crd):
    crd_array = np.array([x for x in crd.values.tolist()])
    fig,ax = plt.subplots()
    sc = plt.scatter(m[:,0], m[:,1], c=crd_array[:,1], cmap='Dark2')
    plt.axis('equal')
    plt.colorbar(sc)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    