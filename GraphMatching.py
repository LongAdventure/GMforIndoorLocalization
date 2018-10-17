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

def PRG(numrow, numcol):
    
    L = numrow * numcol
    PR = np.zeros((4, L))
    
    tmpMat = np.arange(L).reshape((numrow, numcol))
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
    
        

def matchingAccuracy(M):
    
    Acc = np.zeros(4)
    print("Have you updated the numrow and numcol?")
    PR = PRG(12,2)
    for i in range(4):
        Acc[i] = np.sum(M==PR[i,:])
    
    return np.max(Acc)

def MA(A_ori,B_ori,k):
    r = DSPFP(A_ori,B_ori,k)
    
    return matchingAccuracy(r[1]), r[2]
    






def rowToMatrix(row):
    M = np.zeros((8,10))
    
    for i in range(10):
        M[:,9-i] = row[8*i:8*i+8]
    
    M[[6,7],:] = M[[7,6],:]
    
    for i in range(4):
        M[:,[4-i,3-i]] = M[:,[3-i,4-i]]
    
    return M
    
    