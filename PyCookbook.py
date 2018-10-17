# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:19:46 2018

@author: HuQiang
"""

class Pair:
    
    def __init__ (self, x, y):
        self.x = x
        self.y = y
        
    def __repr__(self):
        return 'Pair({0.x!r}, {0.y!r})'.format(self)
    
    def __str__(self):
        return '({0.x!s}, {0.y!s})'.format(self)


_formats = {
    'ymd' : '{d.year}-{d.month}-{d.day}',
    'mdy' : '{d.month}/{d.day}/{d.year}',
    'dmy' : '{d.day}/{d.month}/{d.year}'
    }
class Date:
    
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    
    def __format__(self, code):
        if code == '':
            code = 'ymd'
        fmt = _formats[code]
        return fmt.format(d=self)
    
    
        