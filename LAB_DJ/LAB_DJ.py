# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 12:51:45 2018

@author: HuQiang
"""

'''
>>> plt.style.available
>>> 
['bmh',
 'classic',
 'dark_background',
 'fast',
 'fivethirtyeight',
 'ggplot',
 'grayscale',
 'seaborn-bright',
 'seaborn-colorblind',
 'seaborn-dark-palette',
 'seaborn-dark',
 'seaborn-darkgrid',
 'seaborn-deep',
 'seaborn-muted',
 'seaborn-notebook',
 'seaborn-paper',
 'seaborn-pastel',
 'seaborn-poster',
 'seaborn-talk',
 'seaborn-ticks',
 'seaborn-white',
 'seaborn-whitegrid',
 'seaborn',
 'Solarize_Light2',
 'tableau-colorblind10',
 '_classic_test']
'''

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from pandas import DataFrame
import numpy as np
from GraphMatching import DSPFP, matchingAccuracy
import seaborn as sns

'''
def plot_r1():
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    plt.style.use('seaborn-whitegrid')
    ax1.scatter(r1_stv['range'],np.ones(len(r1_stv['range'])),color='black',alpha=0.2)
    ax1.set_title('range')
    
    ax2.scatter(r1_stv['var'],np.ones(len(r1_stv['range'])),color='black',alpha=0.2)
    ax2.set_title('var')
    
    ax3.scatter(r1_stv['skew'],np.ones(len(r1_stv['range'])),color='black',alpha=0.2)
    ax3.set_title('skew')
    
    ax4.scatter(r1_stv['kurt'],np.ones(len(r1_stv['range'])),color='black',alpha=0.2)
    ax4.set_title('kurt')
    fig.suptitle('r1')


'''
def plot_a08(a08_stv,r1_stv):
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    plt.style.use('seaborn-whitegrid')
    
    a08 = ax1.scatter(a08_stv['range'],np.ones(len(a08_stv['range'])),color='y',alpha=0.2)
    r1  = ax1.scatter(r1_stv['range'],np.ones(len(r1_stv['range']))*2,color='g',alpha=0.2)
    ax1.axis(ymin=0,ymax=4)
    ax1.set_title('range')
    ax1.legend([r1,a08],['r1','08'])
    
    a08 = ax2.scatter(a08_stv['var'],np.ones(len(a08_stv['range'])),color='y',alpha=0.2)
    r1  = ax2.scatter(r1_stv['var'],np.ones(len(r1_stv['range']))*2,color='g',alpha=0.2)
    ax2.axis(ymin=0,ymax=4)
    ax2.set_title('var')
    ax2.legend([r1,a08],['r1','08'])
    
    a08 = ax3.scatter(a08_stv['skew'],np.ones(len(a08_stv['range'])),color='y',alpha=0.2)
    r1  = ax3.scatter(r1_stv['skew'],np.ones(len(r1_stv['range']))*2,color='g',alpha=0.2)
    ax3.axis(ymin=0,ymax=4)
    ax3.set_title('skew')
    ax3.legend([r1,a08],['r1','08'])
    
    a08 = ax4.scatter(a08_stv['kurt'],np.ones(len(a08_stv['range'])),color='y',alpha=0.2)
    r1  = ax4.scatter(r1_stv['kurt'],np.ones(len(r1_stv['range']))*2,color='g',alpha=0.2)
    ax4.axis(ymin=0,ymax=4)
    ax4.set_title('kurt')
    ax3.legend([r1,a08],['r1','08'])

    fig.suptitle('statistical feature comparison')
'''
    
    

def plot():
    plt.style.use('seaborn-whitegrid')
    
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    
    plt.style.use('seaborn-whitegrid')
    
    ax1.scatter(r1_stv['range'],np.ones(len(r1_stv['range'])),color='black',alpha=0.2)
    ax1.set_title('range')
    
    ax2.scatter(r1_stv['var'],np.ones(len(r1_stv['range'])),color='black',alpha=0.2)
    ax2.set_title('var')
    
    ax3.scatter(r1_stv['skew'],np.ones(len(r1_stv['range'])),color='black',alpha=0.2)
    ax3.set_title('skew')
    
    ax4.scatter(r1_stv['kurt'],np.ones(len(r1_stv['range'])),color='black',alpha=0.2)
    ax4.set_title('kurt')
    fig.suptitle('r1')
'''
def error(A, B):
    A_scaled=A/np.max(A)
    B_scaled=B/np.max(B)    
    return np.sqrt(np.sum(np.abs(A_scaled-B_scaled)**2))

def testAccs(stv_target,sta,stv,B):
    
    ##range
    #lolist = [0, 2, 2.5, 3.85, 4.25, 5.25]
    #uplist = [float('inf'), 23.5, 12, 10.85, 10.5, 8]
    
    ##var
    #lolist = [0, 0.25, 0.5, 1.25, 1.5, 2.5]
    #uplist = [float('inf'), 48, 20, 11.5, 10, 6.25]
    
    ##kurt
    #lolist = [-float('inf'),-2.012966, -1.637579, -1.334792, -1.294013, -1.018891 ]
    #uplist = [float('inf'), 6.342324, 2.353731, 1.404222, 0.865874, -0.110179]
    
    ##cv
    #lolist = [-float('inf'), -0.086311,-0.067795,-0.049409,-0.046999,-0.041501]
    #uplist = [float('inf'),-0.007752,-0.009917,-0.014647,-0.015539,-0.020498]
    
    ##skew
    
    lolists, uplists = getlouplist(stv_target)
    Accs = {}
    
    index = ['range', 'var', 'skew', 'kurt', 'cv']
    
    for ind in index:
        lolist = lolists[ind]
        uplist = uplists[ind]
        Acc = []
        for i in range(len(lolist)):
            df = sta[(stv[ind] >= lolist[i]) & (stv[ind] <= uplist[i])]
            a = np.array(df.values).T
            A = euclidean_distances(a)
            #sio.savemat('LAB5_A'+str(i)+'.mat',{'A'+str(i):A})
            P,M,deltaT = DSPFP(A,B,1.5)
            Acc.extend([matchingAccuracy(M)])
        Accs[ind] = Acc
    
    return Accs

def getAcc(sta,stv,B):
    
    ##range
    #lolist = [0, 2, 2.5, 3.85, 4.25, 5.25]
    #uplist = [float('inf'), 23.5, 12, 10.85, 10.5, 8]
    
    ##var
    #lolist = [0, 0.25, 0.5, 1.25, 1.5, 2.5]
    #uplist = [float('inf'), 48, 20, 11.5, 10, 6.25]
    
    ##kurt
    #lolist = [-float('inf'),-2.012966, -1.637579, -1.334792, -1.294013, -1.018891 ]
    #uplist = [float('inf'), 6.342324, 2.353731, 1.404222, 0.865874, -0.110179]
    
    ##cv
    #lolist = [-float('inf'), -0.086311,-0.067795,-0.049409,-0.046999,-0.041501]
    #uplist = [float('inf'),-0.007752,-0.009917,-0.014647,-0.015539,-0.020498]
    
    ##skew
    
    Acc = []
    
    df = sta[((stv['skew'] >= -0.5) & (stv['skew'] <= 1)) | ((stv['kurt'] >= -1.5) & (stv['kurt'] <= 1.5))]
    a = np.array(df.values).T
    A = euclidean_distances(a)
    #sio.savemat('LAB5_A'+str(i)+'.mat',{'A'+str(i):A})
    P,M,deltaT = DSPFP(A,B,1.5)
    Acc.extend([matchingAccuracy(M)])
    
    return Acc

def getstastv(a, apiset):
    index = ['RP'+str(i+1) for i in range(a.shape[0]) ]
    sta = DataFrame(data=a, index=index, columns=apiset).transpose()
    stv = DataFrame(index=sta.index)
    
    stv['range'] = sta.max(axis=1) - sta.min(axis=1)
    stv['var'] = sta.var(axis=1)
    stv['skew'] = sta.skew(axis=1)
    stv['kurt'] = sta.kurt(axis=1)
    stv['cv'] = sta.std(axis=1)/sta.mean(axis=1)
    
    return sta, stv
    
    

'''        
def plot_distribution(df):
    fig, ax = plt.subplots()
    columnsList = ['RP1','RP2','RP3','RP4','RP5','RP6','RP7','RP8','RP9','RP10','RP11','RP12']
    
    for i in range(50):
        target_series = np.array(df.loc[[i],columnsList].values)
        tmp = target_series-np.max(target_series)
        x = (tmp)/np.abs(np.min(tmp))
        
        ax.scatter(x, np.ones(len(columnsList))*(i+1),c='black',alpha=0.3)

def ps(i):
    fig, ax = plt.subplots()
    ax.scatter(a08_s.loc[[i],columnsList], np.ones(len(columnsList)),c='black',alpha=0.3)
    

def select(q,array):
    interval = np.max(array) - np.min(array)
    array = np.sort(array)
    halfrange = q*interval
    aset = []
    amin = array[0]-halfrange
    amax = array[0]+halfrange
    for i in range(len(array)):
        if (array[i] > amin) and (array[i] < amax):
            aset.extend([array[i]])
            amax = array[i]+halfrange
    
    return len(aset)==len(array)
'''    
def getlouplist(stv):
    index = ['range', 'var', 'skew', 'kurt', 'cv']
    qualist = [[0,1], [0.05,0.95], [0.10,0.90], [0.125,0.875], [0.20,0.80], [0.25,0.75], [0.30, 0.70]]
    lolists = {}
    uplists = {}
    for ind in index:
        lolist = []
        uplist = []
        for q in qualist:
            [lo, up] = stv[ind].quantile(q=q).tolist()
            lolist.extend([lo])
            uplist.extend([up])
        lolists[ind] = lolist
        uplists[ind] = uplist
    return lolists, uplists


def plot_dis_heat(row):
    fig = plt.figure()
    Mat = row.reshape((4,3))
    sns.heatmap(Mat,annot=True)

def plot_dis_scatter(row):
    
    fig = plt.figure(figsize=(6,2))
    plt.scatter(row*2,np.ones(len(row)),c='black',alpha=0.3)

def accsTodf(Accsname):
    df = DataFrame(data=eval(Accsname),index = ['100%','90%','80%','75%','60%','50%','40%']).transpose()
    df.to_excel(Accsname+'.xlsx')
    
def plot_final(Accs):
    df = DataFrame(data=Accs,index = ['100%','90%','80%','75%','60%','50%','40%']).transpose()
    fig = plt.figure()
    plt.plot(np.arange(1,8),df.loc[['range'],:].T,label='range')
    plt.plot(np.arange(1,8),df.loc[['var'],:].T,label='var')
    plt.plot(np.arange(1,8),df.loc[['skew'],:].T,label='skew')
    plt.plot(np.arange(1,8),df.loc[['kurt'],:].T,label='kurt')
    plt.plot(np.arange(1,8),df.loc[['cv'],:].T,label='cv')
    plt.legend()
    plt.title('09')

"""
def tmp_p():
    fig = plt.figure(figsize=(24,24))
    ax = plt.axes()

    plt.scatter(m[:,0],m[:,1],c=sps_crd[:,1],cmap='Dark2')
    plt.axis('equal')
    plt.colorbar()
    plt.savefig('mds_result.png')
"""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    