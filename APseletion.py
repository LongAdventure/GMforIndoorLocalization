# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 14:58:41 2017

@author: ASUS
"""

import time
import re
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import distances as dist

def error(A, B):
    A_scaled=A/np.max(A)
    B_scaled=B/np.max(B)    
    return np.sqrt(np.sum(np.abs(A_scaled-B_scaled)**2))

def load_APs(filename):
    '''
    input: single filename(single RP)
        ex: 'D:/Python file/2018TermOne/TX1/Rss(150.0,20.0).txt'
    output:
        list of dict, {BSSID:RSSI}
    '''
    names = ['time','order','none1','none2','BSSID','SSID','RSSI','none3','band','x','y']
    drop_list = ['order','none1','none2','SSID','none3','band']
    R_origi_temp = pd.read_table(filename, names=names).drop(drop_list,axis=1)
    
    try:
        AP_df = R_origi_temp.drop_duplicates().pivot(index='time',columns='BSSID',values='RSSI')
        AP_array = np.array(AP_df.values)  #row means times; columns means BSSID; nan denotes missing values
        AP_name = list(AP_df.columns)
    except ValueError:
        print('Error in ' + filename)
        return R_origi_temp
    
    #求第j列过滤掉nan之后的和
    def columnsMean(j):
        count=0
        mean=0
        for x in AP_array[:,j]:
            if not np.isnan(x): 
                mean+=x
                count+=1
        mean/=count
    
        return mean
    
    AP_vector = [columnsMean(j) for j in range(AP_array.shape[1])]
    
    AP_meanRSS=dict(zip(AP_name,AP_vector))
        
    return AP_meanRSS


def load_APs_all(filenames):
    '''
    input:
        list of filename
    output:
        S_coord: list of coordinates, S_coord.index(x,y) return index of (x,y) in S
        S: list of RSS dict(all RP)
    '''
    S_coord = [[float(x) for x in re.split(r'[(),]', filename)[1:3]] for filename in filenames]
    S = [load_APs(filename) for filename in filenames]
    
    return S_coord,S



def dictToSet(S):
    return [set(S[i].keys()) for i in range(len(S))]

def setToArray(S_LOD,APSet):
    #build AP Matrix according to Set from dict_S
    Array=[]
    for i in range(len(S_LOD)):
        Array.append([S_LOD[i][x] for x in APSet])
    
    Array=np.array(Array)
    
    return Array

    
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
        
        
        
def APcounts(Set,Array):
    df = DataFrame(Array.T,index=Set)
    df.columns.names=['RP']
    
    df_var = DataFrame(df.var(axis=1),columns=['var'])
    df_var.columns.names=['statistics']
    df=pd.concat([df,df_var],axis=1)
    #name去哪了？？？
    
    return df


    
    
    
def errorTestSub(A,n):
    errorList=[]
    for i in range(100000):
        B=np.random.rand(9,n)
        B=euclidean_distances(B)
        errorList.extend([error(A,B)])
    return sum(errorList)/len(errorList)

def errorTest(A):
    errorList=[]
    for i in range(20):
        errorList.extend([errorTestSub(A,i+1)])
    
    return errorList

def ierror(iAPcounts,A):
    errorList=[]
    for i in range(20):
        df=iAPcounts.nlargest(i+1,columns='var')
        Edist_inlar=dist.E_dist(iAPcounts,df.index)
        errorList.extend([error(A,Edist_inlar)])
        
    return errorList

def eListToExcel(errorList,filename):
    df=DataFrame(errorList,index=[str(i+1) for i in range(len(errorList))])
    writer=pd.ExcelWriter(filename)
    df.to_excel(writer)
    writer.save()

def CnMinErrorSub(iAPcounts,A,n):
    CnList = list(combinations(iAPcounts.index, n))
    minError = 100
    AP=set()
    for x in CnList:
        Edist = dist.E_dist(iAPcounts,set(x))
        Error=error(A,Edist)
        if Error<minError:
            minError=Error
            AP=set(x)
    
    return minError,AP

def CnMinError(iAPcounts,A):
    start=time.clock()
    CnErrorList=[]
    CnAPList=[]
    for i in range(20):
        minError,AP=CnMinErrorSub(iAPcounts,A,i+1)
        CnErrorList.extend([minError])
        CnAPList.extend([AP])
    
    end=time.clock()
    return CnErrorList,CnAPList,end-start

def getIndex(iAPSet,CnAPList):
    APList=sorted(list(iAPSet))
    indexList=[]
    for i in range(len(CnAPList)):
        List=[APList.index(x) for x in CnAPList[i]]
        indexList.append(List)
    
    return indexList

def APPlot(iAPSet,CnAPList):
    #getIndex
    APList=sorted(list(iAPSet))
    indexList=[]
    for i in range(len(CnAPList)):
        List=[APList.index(x) for x in CnAPList[i]]
        indexList.append(List)
        
    Matrix=np.zeros((20,20))
    for i in range(len(indexList)):
        for j in indexList[i]:
            Matrix[j][i]=1
    
    df=DataFrame(Matrix,index=APList,columns=[str(i+1) for i in range(len(APList))])
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    sns.heatmap(df,linewidths=0.05,ax=ax,cbar=False)
    ax.set_xlabel('n')
    ax.set_ylabel('AP')
    
    return df

def matchPlot(Matrix):
    #x:n y:RP
    df=DataFrame(Matrix,index=[str(i+1) for i in range(9)],columns=[str(i+1) for i in range(20)])
    fig=plt.figure()
    ax = fig.add_subplot(111)
    
    sns.heatmap(df,linewidths=0.05,ax=ax,cbar=False)
    ax.set_xlabel('n')
    ax.set_ylabel('RP')
    
    
    
    
    

def accuracy(GM_result):
    Matrix=[]
    for Tuple in GM_result:
        column = [Tuple[0][0][i]==Tuple[0][1][i] for i in range(len(Tuple[0][0]))]
        Matrix.append(column)
    
    #x:n y:RP
    return np.array(Matrix).T


    
def APScatter(Array,APList_orig):
    #x:RP  y:AP
    xmax=np.max(Array)
    xmin=np.min(Array)
    
    APList=APList_orig.copy()
    
    for i in range(len(APList)):
        iAPSplitList=APList[i].split(':')
        APList[i]=''
        for x in iAPSplitList:
            APList[i]=APList[i]+x
    
    def APScatterSub(RPVector,i):
        fig=plt.figure()
        fig.set_size_inches(10,0.3)
        ax=fig.add_subplot(111)
        
        ax.scatter(RPVector,np.zeros(len(RPVector)))
        ax.set_xlabel('RSSI')
        ax.set_title(APList_orig[i])
        
        plt.yticks([])
        plt.xlim(xmax=xmax,xmin=xmin)
        plt.savefig('scatter_'+str(i+1)+'_'+APList[i]+'png')
    
    for i in range(Array.shape[0]):
        APScatterSub(Array[i],i)
        

def CnError_M(iAPcounts,CnAPList,A):
    errorList_M=[]
    for x in CnAPList:
        Mdist=dist.M_dist(iAPcounts,x)
        Error=error(A,Mdist)
        errorList_M.extend([Error])
        
    return errorList_M

def mulDistancesErrorSub(A, APcounts, APSetList, Dist):
    errorList = []
    for APSet in APSetList:
        B = dist.X_dist(APcounts, APSet, method=Dist)
        errorList.extend([error(A,B)])
            
    return errorList

def mulDistancesError(A, APcounts,APSetList):
    
    
            
        
        
    distList = ['E','ES','M','MS','Ln','LnS']
    MulErrorList=[]
    for Dist in distList:
        

        return
    
    
    

def CnMinErrorSub_LnS(iAPcounts,A,n):
    CnList = list(combinations(iAPcounts.index, n))
    minError = 100
    AP=set()
    for x in CnList:
        LnSdist = dist.LnS_dist(iAPcounts,set(x))
        Error=error(A,LnSdist)
        if Error<minError:
            minError=Error
            AP=set(x)
    
    return minError,AP

def CnMinError_LnS(iAPcounts,A):
    start=time.clock()
    CnErrorList=[]
    CnAPList=[]
    for i in range(1,20):
        minError,AP=CnMinErrorSub(iAPcounts,A,i+1)
        CnErrorList.extend([minError])
        CnAPList.extend([AP])
    
    end=time.clock()
    return CnErrorList,CnAPList,end-start

def serach(AP_df):
    for i in range(len(AP_df)):
        for j in range(i+1,len(AP_df)):
            if (AP_df.loc[i,['time']]== AP_df.loc[j,['time']]).values and (AP_df.loc[i,['BSSID']] == AP_df.loc[j,['BSSID']]).values:
                print(str(i)+" and "+str(j))




    

                
                


        
    


    
        
            


        
        
    
    
        
    

    
            
        
        
        
        
        
        
        
        
