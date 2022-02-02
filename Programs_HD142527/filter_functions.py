#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:17:46 2022

@author: jeje
"""

import numpy as np



def e_func(x, a, b, c):

    return a * np.exp(-b * x) + c

def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianBeam(base, x_position, D0):
    rows, cols = base.shape
    for x in range(cols):
        base[:,x] = np.exp(((-distance((0,x), (0, x_position))**2)/(2*(D0**2))))
    return base

def gaussianSpyder(base, x_position, D0):
    rows, cols = base.shape
    y = 0
    while y < rows: 
        if D0 > 1:
            D0 -= 0.08
            
        for x in range(cols):
            base[y,x] = np.exp(((-distance((y,x), (y, x_position))**2)/(2*(D0**2))))
            
        y += 1
            
    return base
