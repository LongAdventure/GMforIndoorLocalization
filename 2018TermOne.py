# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:46:02 2018

@author: Latalio
"""
"""
CATELOG
20180411  结果性能度量  
    symmetryDirection()
    matchingAccurary()
20180412  DSFPF算法,AP逐一匹配
    DSPFP()
    matchPerAP()
    
         平面图
    getCoordinateFromFolder()
    coordinateToPlan()
    labelRPOnPlan()
20180506  最大值法
    getRecord()
    getRecords()
20180507
20180508
    getRPDict()
    getAPiSet()
    getRSSArray()
    

"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from functools import reduce

import basicFunc as bf
import time
import os
import re

from CONSTANT import *

##20180411
def symmetryDirection(sVec, shape):
    '''
    input: 
        sVec: np.1darray, matching results in signal space
            ex:  P0 P1 P2,...,Pn
                [s0,s1,s2,...,sn]
        shape:tuple(num(row), num(column))
        
    output: 
        symList:list, each elements is a np.1darray,
                all symmetry direction, 8 for square and 4 for rectangle
    '''
    (m,n) = shape
    sMat = sVec.reshape(shape)
    
    symList = []
    symList.append(sMat.copy().reshape((1,-1))) #original
    
    i=0
    while(i < n-1-i):
        sMat[:, [i, n-1-i]] = sMat[:, [n-1-i, i]]
        i+=1
    symList.append(sMat.copy().reshape((1,-1))) #left-right swap
   
    i=0
    while(i < m-1-i):
        sMat[[i, m-1-i], :] = sMat[[m-1-i, i], :]
        i+=1
    symList.append(sMat.copy().reshape((1,-1))) #up-down swap
    
    i=0
    while(i < n-1-i):
        sMat[:, [i, n-1-i]] = sMat[:, [n-1-i, i]]
        i+=1
    symList.append(sMat.copy().reshape((1,-1))) #left-right swap
    
    if(m==n):
        sMat = sMat.T
        symList.append(sMat.copy().reshape((1,-1))) #transposition
        
        i=0
        while(i < n-1-i):
            sMat[:, [i, n-1-i]] = sMat[:, [n-1-i, i]]
            i+=1
        symList.append(sMat.copy().reshape((1,-1))) #left-right swap
        
        i=0
        while(i < m-1-i):
            sMat[[i, m-1-i], :] = sMat[[m-1-i, i], :]
            i+=1
        symList.append(sMat.copy().reshape((1,-1))) #up-down swap
        
        i=0
        while(i < n-1-i):
            sMat[:, [i, n-1-i]] = sMat[:, [n-1-i, i]]
            i+=1
        symList.append(sMat.copy().reshape((1,-1))) #left-right swap
    
    return symList

def matchingAccurary(real, current, shape):
    '''
    input:
        real: np.1darray, real match
        current: np.1darray, current matching results
        shape: tuple(num(row), num(column)), floor plan shape
    output:
        str, accurary rate
        ex: 7/9 means 7 right matched for all 9 vertexes
    '''
    symList = symmetryDirection(real, shape)
    
    max_matched = 0
    for sym in symList:
        matched = np.sum(current==sym)
        if(matched>max_matched): max_matched = matched
    
    return str(max_matched)+'/'+str(len(real))

def mA_PerAP(real, match_results, shape):
    '''
    input:
        real: np.1darray, real match
        match_results: list of np.1darray
                        every element is a matching result of one specific AP
        shape: tuple(num(row), num(column)), floor plan shape
    output:
        mr_withma:list of tuple
                tuple[0] denote matching result
                tuple[1] denote matching accurary
                ex:
                    (array([2, 1, 6, 0, 3, 8, 7, 4, 5], dtype=int64), '2/9')
    '''
    mr_withma = []
    for result in match_results:
        mr_withma.append((result, matchingAccurary(real, result, shape)))
    return mr_withma
        


##20180412
def DSPFP(A,B):
    '''
    input:
        A,B: np.2darray, adjacent matrix
    output:
        [0]: np.1darray, matching results
        [1]: float, runtime
    '''
    
    start = time.clock()  #timestamp, since 
    n = A.shape[0]
    
    #Initialize　Ｘ，Ｙ
    ONES = np.ones(n)
    ONESS = ONES.dot(ONES.T)
    I = np.identity(n)
    X = np.ones((n,n))/n**2
    Y = np.zeros((n,n))
    
    #Iteratuin
    countX = 0
    alpha = 0.5
    t1 = 0.00000000000000001
    t2 = 0.00000000000000001
    I1 = 100000
    I2 = 100000
    while(True):
        countX +=1
        countY = 0
        Y = np.dot(np.dot(A,X),B)
        while(True):
            countY +=1
            YY = Y 
            + np.dot(I/n + np.dot(np.dot(np.dot(ONES.T,Y),ONES),I)/n**2 - Y/n,ONESS) 
            - np.dot(ONESS,Y)/n
            YY = (YY + np.abs(YY))/2
            if(np.max(np.abs(YY - Y))<=t2 or countY>I2): break
            Y = YY
        XX = (1-alpha)*X + alpha*YY
        XX = XX/np.max(XX)
        if(np.max(np.abs(XX - X))<=t1 or countX>I1): break
        X = XX
    P = np.zeros((n,n))
    X = XX.copy()
    for count in range(n):
        mindex = np.argmax(XX)
        i = int(np.floor(mindex/n))
        j = int(mindex - i*n)
        P[i,j] = 1
        XX[i,:] = -1
        XX[:,j] = -1
        
    end = time.clock()
    
    return np.where(P==1),X,YY,end-start

def matchPerAP(RP,S):
    '''
    input:
        RP: must be column vector
        S: num(AP)×num(RP) np.matrix, 
            must be matrix to guarentee S[:,i] is a cloumn vector
    output:
        match_result: list of np.1darray
                        every element is a matching result of one specific AP
    '''
    match_results = []
    for i in range(S.shape[1]):
        results = DSPFP(euclidean_distances(RP), euclidean_distances(S[:,i]))
        print(results[1])
        match_results.append(results[0][1])
    return match_results

def getCoordinateFromFolder(path):
    '''
    input:
        path: path of RSS.txt files folder
    output:
        np.2darray [[coord_x, coord_y],...]
        ex:array([['150.0', '120.0'],
                   ['150.0', '20.0'],
                   ['150.0', '220.0'],
                   ..., 
                   ['950.0', '520.0'],
                   ['950.0', '620.0'],
                   ['950.0', '720.0']], 
                  dtype='<U5') 
    '''
    coordList = []
    filenameList = os.listdir(path)
    for filename in filenameList:
        filenameSplit = re.split(r'[(),]', filename)
        coordList.append([filenameSplit[1], filenameSplit[2]])
    
    return np.array(coordList)

def coordinateToPlan(coordArray):
    '''
    input:
        output of getCoordinateFromFolder
        np.2darray [[coord_x, coord_y],...]
    output:
        plan of all RP
    '''
    
    fig,ax = plt.subplots()
    ax.scatter(coordArray[:,0], coordArray[:,1])
    plt.xticks(np.arange(100,1000,100))
    plt.yticks(np.arange(70,770,100))
    plt.show()
    
def labelRPOnPlan(coordArrayOverall, coordArrayLocal):
    '''
    input:
        coordArrayOverall: all RP
        coordArrayLocal: RP tseted
    output:
        plan of tested RP in all RP
    '''
    fig, ax = plt.subplots()
    ax.scatter(coordArrayOverall[:,0], coordArrayOverall[:,1])
    ax.scatter(coordArrayLocal[:,0], coordArrayLocal[:,1], s=150, c='red', marker='^')
    plt.xticks(np.arange(100,1000,100))
    plt.yticks(np.arange(70,770,100))
    plt.show()
    
    
    
##20180506
def getRecord(filename):
    '''
    input: single filename(single RP)
        ex: 'D:/Python file/2018TermOne/TX1/Rss(150.0,20.0).txt'
    output:
        DataFrame, {index; time, BSSID, RSSI, x, y}
    '''
    names = ['time','order','none1','none2','BSSID','SSID','RSSI','none3','band','x','y']
    drop_list = ['order','none1','none2','SSID','none3','band']
    recordsInSingleRP = pd.read_table(filename, names=names).drop(drop_list,axis=1)
    recordsInSingleRP = recordsInSingleRP.drop_duplicates()
    
    return recordsInSingleRP

def getRecords(filenames):
    '''
    input:  filenames, list of filename
    output:  
        DataFrame of all filename in filenames, {index; time, BSSID, RSSI, x, y}
    '''
    records = getRecord(filenames[0])
    for i in range(len(filenames)-1):
        records = records.append(getRecord(filenames[i+1]), ignore_index=True)
    
    return records



##20180507
def numberAP(query):
    '''
    input: 
        query: str or int
            if str, return index of the AP
            if int, return BSSID of the AP
    output: 
        see above
    '''
    global AP_NUMBER
    if type(query[0]) == type('a'):
        return [AP_NUMBER.index(q) for q in query]
    else:
        return [AP_NUMBER[q] for q in query]

def numberRP(query):
    '''
    input: 
        query: str or int
            if str, return index of the RP
            if int, return coord of the RP
    output: 
        see above
    '''
    global RP_NUMBER
    if type(query[0]) == type('a'):
        return [RP_NUMBER.index(q) for q in query]
    else:
        return [RP_NUMBER[q] for q in query]


def getRPDict(numRP):
    '''
    input:
        numRP:list, NO. of RP
    output:
        rd:dict_outer of dict_iner
        ex:
            dict_ouer={(x,y):dict_iner}
            dict_iner={AP:RSSI}
    '''
    global RECORDS
    coord = numberRP(numRP)
    xlist = [x for (x,y) in coord]
    ylist = [y for (x,y) in coord]
    condition = 'x in '+str(xlist)+' and '+'y in '+str(ylist)
    rs_sub = RECORDS.query(condition)
    rs_sub_grouped = rs_sub.groupby(['x', 'y', 'BSSID']).mean()
    
    idx = pd.IndexSlice
    rd = {}
    for x in xlist:
        for y in ylist:
            df = rs_sub_grouped.loc[idx[x,y,:],:]
            rv = dict(zip(df.index.get_level_values('BSSID'), [float(v) for v in df.values]))
            rd.update({(x,y):rv})
    
    return rd

def getAPiSet(RPDict):
    '''
    input:
        RPDict: output of getRPDict()
    output:
        intersection of AP
    '''
    
    return list(reduce(lambda x,y: set(x)&set(y), RPDict.values()))

def getRSSArray(RPDict, APiSet):
    '''
    input:
        see name
    output:
        np.array
        coordinate of RP
        AP set
    '''
    arr = []
    for key in RPDict.keys():
        arr_sub = []
        for ap in APiSet:
            arr_sub.extend([RPDict[key][ap]])
        arr.append(arr_sub)
    return np.array(arr),list(RPDict.keys()),APiSet

def ntmp(r,c):
    ind_max = r.argmax()
    return [(r[ind_max]-r[i])/np.log10(euclidean_distances(c[ind_max].reshape(1,-1),c[i].reshape(1,-1)))/10 for i in range(len(r))]


    
    
def nCalculate(numRP):
    
    coord = numberRP(numRP)
    condition = 'x in '+str(xlist)+' and '+'y in '+str(ylist)
    
    
def twod(s1,s2):
    fig,ax = plt.subplots()
    ax.plot(s1,s2,'o')
    plt.show()


def simClusterColumn(df,xlist,ylist):
    subdf = df.query('x in '+str(xlist)+'& y in '+str(ylist))
    subdf_pivot = pd.pivot_table(subdf,index='BSSID',values='RSSI')
    pivot_dict = subdf_pivot.to_dict()
    return pivot_dict['RSSI']

def simClusterColumns(df):
    xlists = [[50,150,250],[250,350,450],[450,550,650],[650,750,850]]
    ylists = [[520,620,720],[320,420,520],[120,220,320]]
    S_LOD = []
    for ylist in ylists:
        for xlist in xlists:
            S_LOD.extend([simClusterColumn(df,xlist,ylist)])
    
    return S_LOD
    

        
    
    
    
    
    