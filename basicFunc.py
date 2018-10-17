# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 20:54:23 2018

@author: ASUS
"""
import pickle

def save(obj, filename):
    file = open(filename, 'wb')
    pickle.dump(obj, file)
    file.close
    
def load(filename):
    file = open(filename, 'rb')  
    obj = pickle.load(file)  
    file.close()
    return obj