# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:26:42 2018

@author: HuQiang
"""
import pandas as pd
import numpy as np
import os
import pickle
import scipy.io as sio
from sklearn.metrics.pairwise import euclidean_distances,manhattan_distances
from scipy.spatial.distance import pdist

import scipy.io as sio
dataPath = "C:/Users/HuQiang/Documents/Python Scripts/Data/"
crd_names = ['crd_07', 'crd_08', 'crd_09', 'crd_10', 'crd_11', 'crd_12', 'crd_13']


#文件存取
def save(objname):
    file = open(objname+'.pkl', 'wb')
    pickle.dump(eval(objname), file)
    file.close
    
def load(filename):
    file = open(filename, 'rb')  
    obj = pickle.load(file)  
    file.close()
    return obj

def savemat(objectname): 
    sio.savemat(objectname+'.mat',{objectname:eval(objectname)})



#records(df)读取
def getFilePath(path):
    
    filenameList = os.listdir(path)
    filenameList = [path + '/'+filename for filename in filenameList]
    return filenameList

def getRecord(filepath):
    '''
    input: single filename(single RP)
        ex: 'D:/Python file/2018TermOne/TX1/Rss(150.0,20.0).txt'
    output:
        DataFrame, {index; time, BSSID, RSSI, x, y}
    '''
    names = ['time','order','none1','none2','BSSID','SSID','RSSI','none3','band','x','y']
    drop_list = ['order','none1','none2','SSID','none3','band']
    recordsInSingleRP = pd.read_table(filepath, names=names).drop(drop_list,axis=1)
    recordsInSingleRP = recordsInSingleRP.drop_duplicates()
    
    return recordsInSingleRP

def getRecords(filepaths):
    '''
    input:  filenames, list of filename
    output:  
        DataFrame of all filename in filenames, {index; time, BSSID, RSSI, x, y}
    '''
    records = getRecord(filepaths[0])
    for i in range(len(filepaths)-1):
        records = records.append(getRecord(filepaths[i+1]), ignore_index=True)
    
    return records




#模拟聚类函数，数据集建立函数
def simClusterColumn(df,xlist,ylist):
    subdf = df.query('x in '+str(xlist)+'& y in '+str(ylist))
    subdf_pivot = pd.pivot_table(subdf,index='BSSID',values='RSSI')
    pivot_dict = subdf_pivot.to_dict()
    return pivot_dict['RSSI']

def simClusterColumns(df):
    
    #12个点实验
    xlists = [[50+i*200,150+i*200,250+i*200] for i in range(4)]
    ylists = [[20+i*200,120+i*200,220+i*200] for i in range(3)]
    
    #63个点实验
    #xlists = [[50+i*100,150+i*100] for i in range(9)]
    #ylists = [[20+i*100,120+i*100] for i in range(7)]
    S_LOD = []
    for ylist in ylists:
        for xlist in xlists:
            S_LOD.extend([simClusterColumn(df,xlist,ylist)])
    
    return S_LOD

def intersection(S_LOD):
    '''
    input:
        S_LOD: list of dict, each dict denotes a RP
    output:
        APSet: intersection of AP for all RP in S
    '''
    APSetList = [set(x.keys()) for x in S_LOD]
    APSet = APSetList[0]
    for apset in APSetList:
        APSet = APSet&apset
    return APSet

def setToArray(S_LOD,APSet):
    #build AP Matrix according to Set from dict_S
    Array=[]
    for i in range(len(S_LOD)):
        Array.append([S_LOD[i][x] for x in APSet])
    
    Array=np.array(Array)
    
    return Array

#得到数据集
def dfToArray(df):
    S_LOD = simClusterColumns(df)
    APSet = intersection(S_LOD)
    Array = setToArray(S_LOD,APSet)
    
    return Array,APSet








def UDboundaryVector(r0, r1):
    r0_S_LOD = simClusterColumns(r0)
    r1_S_LOD = simClusterColumns(r1)
    del r0_S_LOD[4:8]
    del r1_S_LOD[4:8]
    r0_S_LOD.extend(r1_S_LOD)
    
    apiSet = intersection(r0_S_LOD)
    Array = setToArray(r0_S_LOD,apiSet)
    
    dist = np.ones(8)*float('inf')
    dist[0] = np.sum(np.sqrt(np.sum(np.square(Array[0:4]-Array[8:12]),axis=1)))
    dist[1] = np.sum(np.sqrt(np.sum(np.square(Array[0:4]-uptodown(Array[8:12])),axis=1)))

    dist[2] = np.sum(np.sqrt(np.sum(np.square(Array[0:4]-Array[12:16]),axis=1)))
    dist[3] = np.sum(np.sqrt(np.sum(np.square(Array[0:4]-uptodown(Array[12:16])),axis=1)))    

    dist[4] = np.sum(np.sqrt(np.sum(np.square(Array[4:8]-Array[8:12]),axis=1)))
    dist[5] = np.sum(np.sqrt(np.sum(np.square(Array[4:8]-uptodown(Array[8:12])),axis=1)))    
    
    dist[6] = np.sum(np.sqrt(np.sum(np.square(Array[4:8]-Array[12:16]),axis=1)))
    dist[7] = np.sum(np.sqrt(np.sum(np.square(Array[4:8]-uptodown(Array[12:16])),axis=1)))
    return dist

def LRboundaryVector(r0, r1):
    r0_S_LOD = simClusterColumns(r0)
    r1_S_LOD = simClusterColumns(r1)
    r0 = []
    for x in [1,5,9,4,8,12]:
        r0.extend([r0_S_LOD[x-1]])
    r1 = []
    for x in [1,5,9,4,8,12]:
        r1.extend([r1_S_LOD[x-1]])

    r0.extend(r1)
    
    apiSet = intersection(r0)
    Array = setToArray(r0,apiSet)
    
    dist = np.ones(8)*float('inf')
    dist[0] = np.sum(np.sqrt(np.sum(np.square(Array[0:3]-Array[6:9]),axis=1)))
    dist[1] = np.sum(np.sqrt(np.sum(np.square(Array[0:3]-uptodown(Array[6:9])),axis=1)))

    dist[2] = np.sum(np.sqrt(np.sum(np.square(Array[0:3]-Array[9:12]),axis=1)))
    dist[3] = np.sum(np.sqrt(np.sum(np.square(Array[0:3]-uptodown(Array[9:12])),axis=1)))    

    dist[4] = np.sum(np.sqrt(np.sum(np.square(Array[3:6]-Array[6:9]),axis=1)))
    dist[5] = np.sum(np.sqrt(np.sum(np.square(Array[3:6]-uptodown(Array[6:9])),axis=1)))    
    
    dist[6] = np.sum(np.sqrt(np.sum(np.square(Array[3:6]-Array[9:12]),axis=1)))
    dist[7] = np.sum(np.sqrt(np.sum(np.square(Array[3:6]-uptodown(Array[9:12])),axis=1)))
    return dist

def uptodown(OriginalArray):
    Array = OriginalArray.copy()
    numrow = Array.shape[0]
    for i in range(numrow//2):
        Array[[0+i,numrow-1-i],:] = Array[[numrow-1-i,0+i],:]
    return Array
        
    
    
    






















































def error(A, B):
    A_scaled=A/np.max(A)
    B_scaled=B/np.max(B)    
    return np.sqrt(np.sum(np.abs(A_scaled-B_scaled)**2))





def expo1(a,alpha):
    #minus_min = -np.min(a,axis=0)
    a1 = a-np.min(a,axis=0)
    positive_max = np.max(a1,axis=0)
    e = np.exp(a1/alpha)/np.exp(positive_max/alpha)
    return e

def expo2(a,alpha):
    #minus_min = -np.min(a)
    a2 = a-np.min(a)
    positive_max = np.max(a2)
    e = np.exp(a2/alpha)/np.exp(positive_max/alpha)
    return e

def et(a,B):
    
    e1 = []
    e2 = []
    length = 1000
    for i in range(length):
        a1 = expo1(a,i+100000)
        a2 = expo2(a,i+100000)
        A1 = manhattan_distances(a1)
        A2 = manhattan_distances(a2)
        e1.extend([error(A1,B)])
        e2.extend([error(A2,B)])
    
    return e1, e2

def powe1(a,beta):
    #minus_min = -np.min(a,axis=0)
    a1 = a-np.min(a,axis=0)
    positive_max = np.max(a1,axis=0)
    e = a1**beta/positive_max**beta
    return e

def powe2(a,beta):
    #minus_min = -np.min(a)
    a2 = a-np.min(a)
    positive_max = np.max(a2)
    e = a2**beta/positive_max**beta
    return e

def pt(a,B):
    
    e1 = []
    e2 = []
    length = 100
    for i in range(length):
        a1 = powe1(a,0.01*i)
        a2 = powe2(a,0.01*i)
        A1 = manhattan_distances(a1)
        A2 = manhattan_distances(a2)
        e1.extend([error(A1,B)])
        e2.extend([error(A2,B)])
    
    return e1, e2
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    