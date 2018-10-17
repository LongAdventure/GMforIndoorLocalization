# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:56:59 2018

@author: HuQiang
"""
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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
        coordList.append([float(filenameSplit[1]), float(filenameSplit[2])])
    
    return np.array(coordList)

'''
def getGridCentroid(coordArray, n):
    coordArray.reshape()
    (h,w) = coordArray.shape
    G = np.zeros((h-n+1, w-n+1))
    
    for j in range(h-n+1):
        for i in range(w-n+1):
            G[j,i] = np.mean(coordArray[j:j+n,i:i+n])
    return G
'''    

def coordinateToPlan(coordArray):
    '''
    input:
        output of getCoordinateFromFolder
        np.2darray [[coord_x, coord_y],...]
    output:
        plan of all RP
    '''
    
    x = coordArray[:,0]
    y = coordArray[:,1]
    
    x_num = len(set(x))
    y_num = len(set(y))
    
    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)
    
    xticks = np.linspace(x_min, x_max, x_num)
    yticks = np.linspace(y_min, y_max, y_num)
    
    figsize = ((x_max-x_min)/100, (y_max-y_min)/100)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    
    plt.xticks(xticks)
    plt.yticks(yticks)
    ax.scatter(x, y, c='c', marker='.')
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.axis('equal')

def coordinateToPlan_s(coordArray):
    '''
    input:
        output of getCoordinateFromFolder
        np.2darray [[coord_x, coord_y],...]
    output:
        plan of all RP
    '''
    
    x = coordArray[:,0]
    y = coordArray[:,1]
    
    x_num = len(set(x))
    y_num = len(set(y))
    
    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)
    
    #xticks = np.linspace(x_min, x_max, x_num)
    #yticks = np.linspace(y_min, y_max, y_num)
    
    figsize = ((x_max-x_min)/100, (y_max-y_min)/100)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    
    plt.xticks(xticks)
    plt.yticks(yticks)
    ax.scatter(x, y, c='c', marker='.')
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.axis('equal')                                                                                                                                                   
    
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
    

def tmpfunc(coordArray,name):
    '''
    input:
        output of getCoordinateFromFolder
        np.2darray [[coord_x, coord_y],...]
    output:
        plan of all RP
    '''
    
    x = coordArray[:,0]
    y = coordArray[:,1]
    
    x_num = len(set(x))
    y_num = len(set(y))
    
    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)
    
    xticks = np.linspace(x_min, x_max, x_num)
    yticks = np.linspace(y_min, y_max, y_num)
    
    figsize = ((x_max-x_min)/100, (y_max-y_min)/100)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    
    plt.xticks(xticks)
    plt.yticks(yticks)
    ax.scatter(x, y, c='c', marker='.')
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.axis('equal')                                                                                                                                                 
    plt.savefig(name+'.png')

fig = plt.figure(figsize=(11,11))
plt.scatter(crd_D110[:,0],crd_D110[:,1],c='#326756',label='siteD110')
plt.scatter(crd_S110[:,0],crd_S110[:,1],c='#27253D',marker='x',label='siteS110')

plt.scatter(crd_D210[:,0],crd_D210[:,1],c='#7DA87B',label='siteD210')
#plt.scatter(crd_D310[:,0],crd_D310[:,1],c='#F85959',marker='+',label='siteD310')
plt.axis('equal')
#plt.title('siteD310')
plt.legend(frameon=True)
plt.title('siteS110 + siteD110 + siteD210')

fig.savefig('siteS110 + siteD110 + siteD210.png')
    
    
