#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:44:30 2021

@author: jeje
"""

import numpy as np

names = np.loadtxt("/home/jeje/Dokumente/Masterthesis/Programs/ghosts.txt", 
                  dtype = str, skiprows=1, max_rows=6)

#data = np.loadtxt("/home/jeje/Dokumente/Masterthesis/Programs/ghosts.txt", 
                  #skiprows=2, max_rows=6)
print(names[1][1])
