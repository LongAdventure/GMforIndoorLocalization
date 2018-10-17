# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 19:41:07 2018

@author: ASUS
"""

import os

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

###Parameters
colors_=['#949483','#f47b7b','#9f1f5c','#ef9020','#00af3e','#85b7e2','#29245c',
 '#ffd616','#e5352b','#e990ab','#0081b4','#96cbb3','#91be3e','#39a6dd', 
 '#eb0973','#dde2e0','#333c41']



class Preprocess(object):
    
    ###Data load
    def getRecords(self, folderPath):
    
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
            print('\r'+'{:.2f}'.format(count/numrp), end='', flush=True)
    
        print('\nDone!')
        
        print('Concatenation...')
        rds_ori = pd.concat(rdsList,axis=0)
        print('concat...done')
        rds_ori['coord'] = rds_ori.apply(lambda df: (df['x'],df['y']),axis=1)
        print('crd_merge...done')
        rds_ori.drop(columns=['x','y'],inplace=True)
        print('redundance_drop...done')
        rds = rds_ori.pivot_table(values='RSSI',index=['time','coord'],columns='BSSID')
        print('rds_pivot...done')
        rds['coord'] = rds.index.get_level_values(1)
        rds.index = rds.index.droplevel(1)
        print('Done!')
        return rds
    
    ###Data selection
    #rds: records
    #fpt: fingerprint
    #sps: samples
    #crd: coordinate
    def filter_ap1(self, rds, filter_coefficient):
        rds_fpt = rds[rds.columns[:-1]].copy()
        crd = rds['coord']
        ser = rds_fpt.count(axis=0)
        threshold = filter_coefficient*len(rds)
        print('threshold:{:.0f}'.format(threshold))
        res_col = ser[ser>filter_coefficient*len(rds)].index
        print('reserved aps:{:d}'.format(len(res_col)))
        rdsfpt_ap1 = rds_fpt[res_col].copy()
        rdsfpt_ap1.fillna(rdsfpt_ap1.min(axis=0),axis=0,inplace=True)
        sps = rdsfpt_ap1.values
        return sps, crd
    
    
    
###Split
###Actual clustering process(split)
def kme(X, n_clusters, No_):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, max_iter=1000)
    #labels:kmeans.fit_predict(X)
    #cluster_centers:kmeans.cluster_centers_
    labels = kmeans.fit_predict(X)
    cluster_list = []
    for i in np.unique(labels).tolist():
        cluster_list.append(No_[np.where(labels == i)[0]])
        
    return cluster_list

def _split(X, cluster_list, n_clusters, No_):
    
    cl = []
    for subX_index in cluster_list:
        _cl = kme(X[subX_index,:], n_clusters, No_[subX_index])
        cl.extend(_cl)
    
    return cl

def split(sps, orders):
    
    n_sample = sps.shape[0]
    cluster_list = [np.arange(n_sample)]
    No_ = np.arange(n_sample)
    labels = np.ones(n_sample, dtype='int32')*-1
    order_list = [int(num) for num in list(orders)]
    for order in order_list:
        cluster_list = _split(sps, cluster_list, order, No_)
    
    for i,index in enumerate(cluster_list):
        labels[index] = i
        
    return cluster_list, labels

def plot_result(y,crd,name):
    df = DataFrame({'coord':crd,'label':y})
    num_label = len(set(df['label'].values))
    '''
    colors = ['#DE5B7B','#1B3764','#0E0220','#2E99B0',
              '#7A4579','#A2792F','#FF2E4C','#4E9525',
              '#35D0BA','#7971EA','#00818A','#FF5E3A',
              '#00D0BA','#7900EA','#FF818A','#FF5E00']
    '''
              
    #fig, ax = plt.subplots()
    coordList = df['coord'].tolist()
    coordList = [list(x) for x in coordList]
    coordList = np.array(coordList)
    x_max = np.max(coordList[:,0])
    x_min = np.min(coordList[:,0])
    y_max = np.max(coordList[:,1])
    y_min = np.min(coordList[:,1])
    
    figsize = ((x_max-x_min)/100, (y_max-y_min)/100)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    
    for i in range(num_label):
        coordList = df[df['label']==i]['coord'].tolist()
        coordList = [list(x) for x in coordList]
        coordArray = np.array(coordList)
        ax.scatter(coordArray[:,0], coordArray[:,1], c=colors_[i])
        ax.text(coordArray[:,0].mean(), coordArray[:,1].mean(), str(i), fontsize=24)
    plt.axis('equal')
    
    return


###Matching
#preprocessing
def getClusterFingerprint(X, cl):
    
    cf = [] #cluster fingerprint
    for sub_index in cl:
        cf.append(X[sub_index,:].mean(axis=0)) #mean by row
    
    return np.array(cf)

def getPhysicalCoordinate(area_size, numrow, numcol):
    (x_length, y_length) = area_size
    x_stride = x_length /(2*numcol)
    y_stride = y_length / (2*numrow)
    x = [x_stride+i*2*x_stride for i in range(numcol)]
    y = [y_stride+i*2*y_stride for i in range(numrow)]
    
    pcoord = []
    for j in y:
        for i in x:
            pcoord.append([i,j])
    
    return np.array(pcoord)

'''
def tmp(n):
    labels = split(sps_site12,str(n))
    plot_result(labels,crd_site12,'tmp')
    
def tmp1(filter_coefficient):
    sps,crd = p.filter_ap1(rds_site12, filter_coefficient)
    labels = split(sps,'2222')
    plot_result(labels,crd_site12,'coverage_99%')
'''

def tmpa(rds):
    
    p = Preprocess()
    sps, crd = p.filter_ap1(rds,0.9)
    cluster_list, labels = split(sps,'2222')
    cfp = getClusterFingerprint(sps, cluster_list)
    pcs = getPhysicalCoordinate((776,1050),4,4)
    plot_result(labels,crd,'tmp')
    A = euclidean_distances(pcs)
    B = euclidean_distances(cfp)
    R = DSPFP(A,B,1.5)
    
    return R
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    