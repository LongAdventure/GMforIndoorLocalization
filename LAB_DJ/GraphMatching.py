# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:36:03 2018

@author: HuQiang
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import time

def DSPFP(A_ori,B_ori,k):
    '''
    input:
        A,B: np.2darray, adjacent matrix
    output:
        [0]: np.1darray, matching results
        [1]: float, runtime
    '''
    A = A_ori/np.max(A_ori)*k
    B = B_ori/np.max(B_ori)*k
    
    start = time.clock()  #timestamp, since 
    n = A.shape[0]
    E = np.ones((n,1))
    I = np.identity(n) 
    
    #Initialize　Ｘ，Ｙ    
    X = np.dot(E,E.T)/n**2
    Y = np.zeros((n,n))
    XX = np.ones((n,n))*float('inf')
    YY = np.ones((n,n))*float('inf')
    
    #Iteratuin
    alpha = 0.5
    I1 = 1000000
    I2 = 1000000
    t1 = 0.0000000001
    t2 = 0.0000000001


    countX = 0
    while((np.max(np.abs(X - XX)) > t1) and (countX < I1)):
        countX += 1
        XX = X.copy()
        
        Y = np.dot(np.dot(A,X),B)
        countY = 0
        while((np.max(np.abs(Y - YY)) > t2) and (countY < I2)):
            countY += 1
            YY = Y.copy()
            Y = Y + np.dot(I/n + np.dot(np.dot(E.T,Y),E)*I/n**2 - Y/n,np.dot(E,E.T)) - np.dot(np.dot(E,E.T),Y)/n
            Y = (Y + np.abs(Y))/2
            
        X = (1-alpha)*X + alpha*Y
    
    P = X
    XX = X.copy()

    M = np.zeros(n)
    for count in range(n):
        mindex = np.argmax(XX)
        i = int(np.floor(mindex/n))
        j = int(mindex - i*n)
        M[i] = j
        XX[i,:] = -1
        XX[:,j] = -1
    
    end = time.clock()
    
    return P, M, end-start

def PRG(real_order, numrow, numcol):
    
    L = numrow * numcol
    PR = np.zeros((4, L))
    
    tmpMat = np.array(real_order).reshape((numrow, numcol))
    PR[0,:] = tmpMat.ravel()
    
    #交换上下
    for i in range(numrow//2):
        tmpMat[[i,numrow-1-i],:] = tmpMat[[numrow-1-i,i],:]
    PR[1,:] = tmpMat.ravel()
    
    #交换左右
    for i in range(numcol//2):
        tmpMat[:,[i,numcol-1-i]] = tmpMat[:,[numcol-1-i,i]]
    PR[2,:] = tmpMat.ravel()
    
    #交换上下
    for i in range(numrow//2):
        tmpMat[[i,numrow-1-i],:] = tmpMat[[numrow-1-i,i],:]
    PR[3,:] = tmpMat.ravel()
    
    return PR
    
        

def matchingAccuracy(M, real_order):
    
    Acc = np.zeros(4)
    print("Have you updated the numrow and numcol?")
    PR = PRG(real_order, 4, 4)
    for i in range(4):
        Acc[i] = np.sum(M==PR[i,:])
    
    
    return np.max(Acc)
    



def simClusterColumn(df,xlist,ylist):
    subdf = df.query('x in '+str(xlist)+'& y in '+str(ylist))
    subdf_pivot = pd.pivot_table(subdf,index='BSSID',values='RSSI')
    pivot_dict = subdf_pivot.to_dict()
    return pivot_dict['RSSI']

def simClusterColumns(df):
    '''
    12点实验
    xlists = [[50,150,250],[250,350,450],[450,550,650],[650,750,850]]
    ylists = [[520,620,720],[320,420,520],[120,220,320]]
    '''
    xlists = [[50+i*100,150+i*100] for i in range(9)]
    ylists = [[20+i*100,120+i*100] for i in range(7)]
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


def rowToMatrix(row):
    M = np.zeros((8,10))
    
    for i in range(10):
        M[:,9-i] = row[8*i:8*i+8]
    
    M[[6,7],:] = M[[7,6],:]
    
    for i in range(4):
        M[:,[4-i,3-i]] = M[:,[3-i,4-i]]
    
    return M


###generate physical coordinate
def PCG(area_size, numrow, numcol):
    (x_length, y_length) = area_size
    x_stride = x_length /(2*numcol)
    y_stride = y_length / (2*numrow)
    x = [x_stride+i*2*x_stride for i in range(numcol)]
    y = [y_stride+i*2*y_stride for i in range(numrow)]
    
    pcoord = []
    for i in x:
        for j in y:
            pcoord.append([i,j])
    
    return np.array(pcoord)
    
    